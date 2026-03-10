import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FaceDetector implements Closeable {

    public static class Detection {
        public float x1;
        public float y1;
        public float x2;
        public float y2;
        public float score;
        // 5 点关键点: [x0,y0,x1,y1,...,x4,y4]
        public final float[] kps = new float[10];
    }

    private static class AnchorGrid {
        final float[] cx;
        final float[] cy;

        AnchorGrid(float[] cx, float[] cy) {
            this.cx = cx;
            this.cy = cy;
        }
    }

    private static final int[] STRIDES = new int[]{8, 16, 32};
    private static final int ANCHOR_PER_LOC = 2;

    private final int inputWidth;
    private final int inputHeight;
    private final float inputMean;
    private final float inputStd;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    private final FloatBuffer inputTensorBuffer;
    private final OnnxTensor inputTensor;
    private final long[] inputShape;

    private final Map<Integer, AnchorGrid> anchorGrids = new HashMap<>();
    private final BufferedImage letterboxCanvas;

    public FaceDetector(String modelPath) throws OrtException {
        this(modelPath, 640, 640, 127.5f, 128.0f);
    }

    public FaceDetector(String modelPath, int inputWidth, int inputHeight, float inputMean, float inputStd) throws OrtException {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputMean = inputMean;
        this.inputStd = inputStd;

        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        this.session = env.createSession(modelPath, opts);
        this.inputName = session.getInputNames().iterator().next();

        this.inputShape = new long[]{1, 3, inputHeight, inputWidth};
        this.inputTensorBuffer = ByteBuffer
                .allocateDirect(1 * 3 * inputHeight * inputWidth * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        this.inputTensor = OnnxTensor.createTensor(env, inputTensorBuffer, inputShape);
        this.letterboxCanvas = new BufferedImage(inputWidth, inputHeight, BufferedImage.TYPE_3BYTE_BGR);

        for (int stride : STRIDES) {
            anchorGrids.put(stride, buildAnchorGrid(stride, inputWidth, inputHeight));
        }
    }

    private static AnchorGrid buildAnchorGrid(int stride, int inputW, int inputH) {
        int fmW = inputW / stride;
        int fmH = inputH / stride;
        int n = fmW * fmH * ANCHOR_PER_LOC;
        float[] cx = new float[n];
        float[] cy = new float[n];
        int idx = 0;
        for (int y = 0; y < fmH; y++) {
            float cyVal = y * stride;
            for (int x = 0; x < fmW; x++) {
                float cxVal = x * stride;
                for (int a = 0; a < ANCHOR_PER_LOC; a++) {
                    cx[idx] = cxVal;
                    cy[idx] = cyVal;
                    idx++;
                }
            }
        }
        return new AnchorGrid(cx, cy);
    }

    /**
     * 检测人脸，返回 NMS 后结果（按分数降序）。
     */
    public List<Detection> detect(BufferedImage image, float scoreThresh, float nmsIouThresh, int topK) throws OrtException {
        float[] meta = new float[3]; // [scale, padX, padY]
        preprocessToCHW(image, inputTensorBuffer, meta);
        inputTensorBuffer.rewind();

        List<Detection> candidates = new ArrayList<>();
        try (OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor))) {
            decodeOutputs(result, image.getWidth(), image.getHeight(), meta[0], meta[1], meta[2], scoreThresh, candidates);
        }
        return nms(candidates, nmsIouThresh, topK);
    }

    private void preprocessToCHW(BufferedImage src, FloatBuffer dst, float[] meta) {
        int srcW = src.getWidth();
        int srcH = src.getHeight();
        float scale = Math.min((float) inputWidth / srcW, (float) inputHeight / srcH);
        int newW = Math.max(1, Math.round(srcW * scale));
        int newH = Math.max(1, Math.round(srcH * scale));
        int padX = (inputWidth - newW) / 2;
        int padY = (inputHeight - newH) / 2;

        Graphics2D g = letterboxCanvas.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.fillRect(0, 0, inputWidth, inputHeight);
            g.drawImage(src, padX, padY, newW, newH, null);
        } finally {
            g.dispose();
        }

        dst.rewind();
        int planeSize = inputWidth * inputHeight;
        for (int i = 0; i < planeSize * 3; i++) {
            dst.put(0.0f);
        }

        for (int y = 0; y < inputHeight; y++) {
            for (int x = 0; x < inputWidth; x++) {
                int rgb = letterboxCanvas.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF);
                float gch = ((rgb >> 8) & 0xFF);
                float b = (rgb & 0xFF);

                // SCRFD 常见预处理： (pixel - 127.5) / 128
                float rn = (r - inputMean) / inputStd;
                float gn = (gch - inputMean) / inputStd;
                float bn = (b - inputMean) / inputStd;

                int idx = y * inputWidth + x;
                dst.put(idx, rn);
                dst.put(planeSize + idx, gn);
                dst.put(planeSize * 2 + idx, bn);
            }
        }

        meta[0] = scale;
        meta[1] = padX;
        meta[2] = padY;
    }

    private void decodeOutputs(
            OrtSession.Result result,
            int origW,
            int origH,
            float scale,
            float padX,
            float padY,
            float scoreThresh,
            List<Detection> out) throws OrtException {

        Map<Integer, float[][]> scoresByStride = new HashMap<>();
        Map<Integer, float[][]> bboxesByStride = new HashMap<>();
        Map<Integer, float[][]> kpsByStride = new HashMap<>();

        for (int i = 0; i < result.size(); i++) {
            OnnxValue value = result.get(i);
            if (!(value instanceof OnnxTensor)) {
                continue;
            }
            OnnxTensor tensor = (OnnxTensor) value;
            TensorInfo info = (TensorInfo) tensor.getInfo();
            long[] shape = info.getShape();
            if (shape.length != 2) {
                continue;
            }
            int n = (int) shape[0];
            int c = (int) shape[1];
            int stride = strideFromRows(n);
            if (stride <= 0) {
                continue;
            }
            float[][] data = (float[][]) tensor.getValue();
            if (c == 1) {
                scoresByStride.put(stride, data);
            } else if (c == 4) {
                bboxesByStride.put(stride, data);
            } else if (c == 10) {
                kpsByStride.put(stride, data);
            }
        }

        for (int stride : STRIDES) {
            float[][] scores = scoresByStride.get(stride);
            float[][] boxes = bboxesByStride.get(stride);
            float[][] kps = kpsByStride.get(stride);
            AnchorGrid grid = anchorGrids.get(stride);
            if (scores == null || boxes == null || kps == null || grid == null) {
                continue;
            }
            int n = scores.length;
            for (int i = 0; i < n; i++) {
                float score = toProbability(scores[i][0]);
                if (score < scoreThresh) {
                    continue;
                }

                float l = boxes[i][0] * stride;
                float t = boxes[i][1] * stride;
                float r = boxes[i][2] * stride;
                float b = boxes[i][3] * stride;

                float cx = grid.cx[i];
                float cy = grid.cy[i];

                float x1 = (cx - l - padX) / scale;
                float y1 = (cy - t - padY) / scale;
                float x2 = (cx + r - padX) / scale;
                float y2 = (cy + b - padY) / scale;

                Detection det = new Detection();
                det.x1 = clamp(x1, 0.0f, origW - 1.0f);
                det.y1 = clamp(y1, 0.0f, origH - 1.0f);
                det.x2 = clamp(x2, 0.0f, origW - 1.0f);
                det.y2 = clamp(y2, 0.0f, origH - 1.0f);
                det.score = score;

                for (int p = 0; p < 5; p++) {
                    float dx = kps[i][2 * p] * stride;
                    float dy = kps[i][2 * p + 1] * stride;
                    float kx = (cx + dx - padX) / scale;
                    float ky = (cy + dy - padY) / scale;
                    det.kps[2 * p] = clamp(kx, 0.0f, origW - 1.0f);
                    det.kps[2 * p + 1] = clamp(ky, 0.0f, origH - 1.0f);
                }
                out.add(det);
            }
        }
    }

    private static int strideFromRows(int rows) {
        if (rows == 12800) {
            return 8;
        }
        if (rows == 3200) {
            return 16;
        }
        if (rows == 800) {
            return 32;
        }
        return -1;
    }

    private static float toProbability(float v) {
        if (v >= 0.0f && v <= 1.0f) {
            return v;
        }
        return (float) (1.0 / (1.0 + Math.exp(-v)));
    }

    private static float clamp(float v, float low, float high) {
        if (v < low) {
            return low;
        }
        if (v > high) {
            return high;
        }
        return v;
    }

    private static float iou(Detection a, Detection b) {
        float xx1 = Math.max(a.x1, b.x1);
        float yy1 = Math.max(a.y1, b.y1);
        float xx2 = Math.min(a.x2, b.x2);
        float yy2 = Math.min(a.y2, b.y2);
        float w = Math.max(0.0f, xx2 - xx1);
        float h = Math.max(0.0f, yy2 - yy1);
        float inter = w * h;
        float areaA = Math.max(0.0f, a.x2 - a.x1) * Math.max(0.0f, a.y2 - a.y1);
        float areaB = Math.max(0.0f, b.x2 - b.x1) * Math.max(0.0f, b.y2 - b.y1);
        float union = areaA + areaB - inter;
        if (union <= 0.0f) {
            return 0.0f;
        }
        return inter / union;
    }

    private static List<Detection> nms(List<Detection> dets, float iouThresh, int topK) {
        if (dets.isEmpty()) {
            return dets;
        }
        dets.sort(Comparator.comparingDouble((Detection d) -> d.score).reversed());
        List<Detection> kept = new ArrayList<>();
        boolean[] removed = new boolean[dets.size()];
        for (int i = 0; i < dets.size(); i++) {
            if (removed[i]) {
                continue;
            }
            Detection di = dets.get(i);
            kept.add(di);
            if (topK > 0 && kept.size() >= topK) {
                break;
            }
            for (int j = i + 1; j < dets.size(); j++) {
                if (removed[j]) {
                    continue;
                }
                if (iou(di, dets.get(j)) > iouThresh) {
                    removed[j] = true;
                }
            }
        }
        return kept;
    }

    @Override
    public void close() throws IOException {
        try {
            inputTensor.close();
            session.close();
        } catch (OrtException e) {
            throw new IOException(e);
        }
    }
}
