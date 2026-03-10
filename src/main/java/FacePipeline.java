import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

/**
 * 检测 + 裁剪 + FaceNet embedding 的整合流水线。
 */
public class FacePipeline implements Closeable {

    private final FaceDetector detector;
    private final FaceNet faceNet;

    // FaceNet 推理资源（复用，避免重复创建）
    private final OrtEnvironment env;
    private final OrtSession session;
    private final FloatBuffer inputTensorBuffer;
    private final OnnxTensor inputTensor;
    private final FloatBuffer outputTensorBuffer;
    private final OnnxTensor outputTensor;
    private final float[][][][] batchArray = new float[1][3][160][160];
    private final float[][] embeddings = new float[1][512];
    private final byte[] embeddingBytes = new byte[512 * 4];

    // 动态缓存，裁剪尺寸变化时按需扩容
    private byte[][][] imageArray = new byte[160][160][3];

    public FacePipeline(String detModelPath, String facenetModelPath) throws OrtException {
        this.detector = new FaceDetector(detModelPath);
        this.faceNet = new FaceNet();
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(facenetModelPath, new OrtSession.SessionOptions());

        long[] inputShape = {1, 3, 160, 160};
        long[] outputShape = {1, 512};
        this.inputTensorBuffer = ByteBuffer.allocateDirect((int) (inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * 4))
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        this.outputTensorBuffer = ByteBuffer.allocateDirect((int) (outputShape[0] * outputShape[1] * 4))
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        this.inputTensor = OnnxTensor.createTensor(env, inputTensorBuffer, inputShape);
        this.outputTensor = OnnxTensor.createTensor(env, outputTensorBuffer, outputShape);
    }

    public static class PipelineResult {
        public boolean success;
        public float detScore;
        public byte[] embedding;
    }

    /**
     * 对单张图片执行：检测最大脸 -> 裁剪 -> embedding。
     */
    public PipelineResult extractLargestFaceEmbedding(BufferedImage image) throws OrtException {
        PipelineResult result = new PipelineResult();
        List<FaceDetector.Detection> dets = detector.detect(image, 0.45f, 0.4f, 1);
        if (dets.isEmpty()) {
            result.success = false;
            return result;
        }
        FaceDetector.Detection best = dets.get(0);
        result.detScore = best.score;

        BufferedImage crop = cropFace(image, best, 1.2f);
        int h = crop.getHeight();
        int w = crop.getWidth();
        ensureImageArrayCapacity(h, w);
        byte[] argb = imageToARGBBytesNoHeader(crop);

        faceNet.processFrame(
                session, w, h,
                argb, imageArray, batchArray,
                inputTensorBuffer, inputTensor,
                outputTensorBuffer, outputTensor,
                embeddings, embeddingBytes);

        result.success = true;
        result.embedding = embeddingBytes.clone();
        return result;
    }

    public float compareTwoImages(BufferedImage a, BufferedImage b) throws OrtException {
        PipelineResult ra = extractLargestFaceEmbedding(a);
        PipelineResult rb = extractLargestFaceEmbedding(b);
        if (!ra.success || !rb.success) {
            return -1.0f;
        }
        return faceNet.byte_cosine_similarity(ra.embedding, rb.embedding);
    }

    private void ensureImageArrayCapacity(int h, int w) {
        if (imageArray.length < h || imageArray[0].length < w) {
            imageArray = new byte[h][w][3];
        }
    }

    private static BufferedImage cropFace(BufferedImage src, FaceDetector.Detection det, float expandRatio) {
        float w = det.x2 - det.x1;
        float h = det.y2 - det.y1;
        float cx = (det.x1 + det.x2) * 0.5f;
        float cy = (det.y1 + det.y2) * 0.5f;
        float side = Math.max(w, h) * expandRatio;

        int x1 = Math.max(0, Math.round(cx - side * 0.5f));
        int y1 = Math.max(0, Math.round(cy - side * 0.5f));
        int x2 = Math.min(src.getWidth() - 1, Math.round(cx + side * 0.5f));
        int y2 = Math.min(src.getHeight() - 1, Math.round(cy + side * 0.5f));
        int cw = Math.max(1, x2 - x1);
        int ch = Math.max(1, y2 - y1);
        return src.getSubimage(x1, y1, cw, ch);
    }

    private static byte[] imageToARGBBytesNoHeader(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] bytes = new byte[width * height * 4];
        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = image.getRGB(x, y);
                bytes[index++] = (byte) ((pixel >> 24) & 0xFF);
                bytes[index++] = (byte) ((pixel >> 16) & 0xFF);
                bytes[index++] = (byte) ((pixel >> 8) & 0xFF);
                bytes[index++] = (byte) (pixel & 0xFF);
            }
        }
        return bytes;
    }

    @Override
    public void close() throws IOException {
        try {
            detector.close();
            inputTensor.close();
            outputTensor.close();
            session.close();
        } catch (OrtException e) {
            throw new IOException(e);
        }
    }

    /**
     * 用法：
     *   默认：直接使用项目内默认路径
     *   或：java FacePipeline <detModel> <facenetModel> <imgA> <imgB>
     */
    public static void main(String[] args) {
        String detModelPath = "/home/model/project/facenet/model/facedet.onnx";
        String facenetModelPath = "/home/model/project/facenet/model/facenet.onnx";
        String imageAPath = "/home/model/project/facenet/image_test/zhou_01.jpg";
        String imageBPath = "/home/model/project/facenet/image_test/zhou_02.jpg";

        if (args.length >= 4) {
            detModelPath = args[0];
            facenetModelPath = args[1];
            imageAPath = args[2];
            imageBPath = args[3];
        }

        try (FacePipeline pipeline = new FacePipeline(detModelPath, facenetModelPath)) {
            BufferedImage imgA = ImageIO.read(new File(imageAPath));
            BufferedImage imgB = ImageIO.read(new File(imageBPath));
            if (imgA == null || imgB == null) {
                System.out.println("输入图片读取失败");
                return;
            }

            PipelineResult ra = pipeline.extractLargestFaceEmbedding(imgA);
            PipelineResult rb = pipeline.extractLargestFaceEmbedding(imgB);
            if (!ra.success || !rb.success) {
                System.out.println("至少一张图片未检测到人脸");
                return;
            }

            float sim = pipeline.faceNet.byte_cosine_similarity(ra.embedding, rb.embedding);
            System.out.printf("A检测分数: %.4f, B检测分数: %.4f%n", ra.detScore, rb.detScore);
            System.out.printf("两张图片相似度: %.4f%n", sim);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
