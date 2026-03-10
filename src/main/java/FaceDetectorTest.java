import ai.onnxruntime.OrtException;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

public class FaceDetectorTest {

    private static BufferedImage drawDetections(BufferedImage src, List<FaceDetector.Detection> dets) {
        BufferedImage out = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        try {
            g.drawImage(src, 0, 0, null);
            g.setStroke(new BasicStroke(2.0f));
            for (FaceDetector.Detection d : dets) {
                int x = Math.max(0, Math.round(d.x1));
                int y = Math.max(0, Math.round(d.y1));
                int w = Math.max(1, Math.round(d.x2 - d.x1));
                int h = Math.max(1, Math.round(d.y2 - d.y1));
                g.setColor(Color.RED);
                g.drawRect(x, y, w, h);
                g.setColor(Color.GREEN);
                for (int i = 0; i < 5; i++) {
                    int px = Math.round(d.kps[2 * i]);
                    int py = Math.round(d.kps[2 * i + 1]);
                    g.fillOval(px - 2, py - 2, 4, 4);
                }
            }
        } finally {
            g.dispose();
        }
        return out;
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

    public static void main(String[] args) throws IOException, OrtException {
        String modelPath = "/home/model/project/facenet/model/facedet.onnx";
        String inputDirPath = "/home/model/project/facenet/image_test";
        String outputDirPath = "/home/model/project/facenet/image_test_out";

        File inputDir = new File(inputDirPath);
        File outputDir = new File(outputDirPath);
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IOException("创建输出目录失败: " + outputDirPath);
        }

        File[] files = inputDir.listFiles((dir, name) -> {
            String n = name.toLowerCase();
            return n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".png");
        });
        if (files == null || files.length == 0) {
            System.out.println("未找到测试图片: " + inputDirPath);
            return;
        }

        try (FaceDetector detector = new FaceDetector(modelPath)) {
            for (File f : files) {
                BufferedImage img = ImageIO.read(f);
                if (img == null) {
                    System.out.println("跳过无法读取的图片: " + f.getName());
                    continue;
                }
                long t0 = System.nanoTime();
                List<FaceDetector.Detection> dets = detector.detect(img, 0.45f, 0.4f, 50);
                long t1 = System.nanoTime();
                System.out.printf("%s 检测到 %d 张人脸 (%.2f ms)%n",
                        f.getName(), dets.size(), (t1 - t0) / 1e6);

                BufferedImage vis = drawDetections(img, dets);
                String stem = f.getName();
                int dot = stem.lastIndexOf('.');
                if (dot > 0) {
                    stem = stem.substring(0, dot);
                }

                File visOut = new File(outputDir, stem + "_det.jpg");
                ImageIO.write(vis, "jpg", visOut);

                for (int i = 0; i < dets.size(); i++) {
                    BufferedImage crop = cropFace(img, dets.get(i), 1.2f);
                    File cropOut = new File(outputDir, stem + "_face_" + i + ".jpg");
                    ImageIO.write(crop, "jpg", cropOut);
                }
            }
        }
        System.out.println("检测与裁剪结果已输出到: " + outputDirPath);
    }
}
