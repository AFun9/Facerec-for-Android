import ai.onnxruntime.OrtException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class FaceRegistryTest {

    private static BufferedImage readImage(String path) throws IOException {
        BufferedImage image = ImageIO.read(new File(path));
        if (image == null) {
            throw new IOException("无法读取图片: " + path);
        }
        return image;
    }

    public static void main(String[] args) throws IOException, OrtException {
        String detModelPath = "/home/model/project/facenet/model/facedet.onnx";
        String facenetModelPath = "/home/model/project/facenet/model/facenet.onnx";
        String registryPath = "/home/model/project/facenet/face_registry.bin";
        String registryJsonPath = "/home/model/project/facenet/face_registry.json";

        String zhou1 = "/home/model/project/facenet/image_test/zhou_01.jpg";
        String zhou2 = "/home/model/project/facenet/image_test/zhou_02.jpg";
        String zhou3 = "/home/model/project/facenet/image_test/zhou_03.jpg";
        String zhou4 = "/home/model/project/facenet/image_test/zhou_04.jpg";
        String jack1 = "/home/model/project/facenet/image_test/jack_01.jpg";

        try (FaceRegistry registry = new FaceRegistry(detModelPath, facenetModelPath)) {
            boolean ok1 = registry.register("zhou", readImage(zhou1));
            boolean ok2 = registry.register("zhou", readImage(zhou2));
            boolean ok3 = registry.register("jack", readImage(jack1));
            if (!ok1 || !ok2 || !ok3) {
                throw new RuntimeException("注册失败：存在图片未检测到可用人脸");
            }

            System.out.printf("注册完成: user=%d, embedding=%d%n", registry.userCount(), registry.embeddingCount());

            FaceRegistry.MatchResult m1 = registry.identify(readImage(zhou3), 0.45f);
            System.out.printf("识别 zhou_03 -> user=%s, sim=%.4f, det=%.4f%n", m1.userId, m1.similarity, m1.detScore);
            if (!"zhou".equals(m1.userId)) {
                throw new RuntimeException("识别失败: zhou_03 预期匹配 zhou");
            }

            FaceRegistry.MatchResult m2 = registry.identify(readImage(jack1), 0.45f);
            System.out.printf("识别 jack_01 -> user=%s, sim=%.4f, det=%.4f%n", m2.userId, m2.similarity, m2.detScore);
            if (!"jack".equals(m2.userId)) {
                throw new RuntimeException("识别失败: jack_01 预期匹配 jack");
            }

            registry.save(registryPath);
            System.out.println("人脸库已保存: " + registryPath);
            registry.saveAsJson(registryJsonPath);
            System.out.println("人脸库已导出JSON: " + registryJsonPath);
        }

        try (FaceRegistry loaded = new FaceRegistry(detModelPath, facenetModelPath)) {
            loaded.load(registryPath);
            System.out.printf("加载完成: user=%d, embedding=%d%n", loaded.userCount(), loaded.embeddingCount());

            FaceRegistry.MatchResult m3 = loaded.identify(readImage(zhou4), 0.45f);
            System.out.printf("识别 zhou_04(加载后) -> user=%s, sim=%.4f, det=%.4f%n",
                    m3.userId, m3.similarity, m3.detScore);
            if (!"zhou".equals(m3.userId)) {
                throw new RuntimeException("加载后识别失败: zhou_04 预期匹配 zhou");
            }
        }

        System.out.println("FaceRegistryTest 运行成功");
    }
}
