import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Main {
    // 将图片转换为包含宽高和 ARGB 字节数组，用来生成测试数据的
    public static byte[] imageToARGBBytes(String filePath) {
        try {
            BufferedImage image = ImageIO.read(new File(filePath));
            if (image == null) {
                return null;
            }
            int width = image.getWidth();
            int height = image.getHeight();
            int dataLength = width * height * 4;
            byte[] bytes = new byte[8 + dataLength]; // 前 8 字节: width(4) + height(4), 后 dataLength 字节: ARGB 数据
            int index = 0;

            bytes[index++] = (byte) ((width >> 24) & 0xFF);
            bytes[index++] = (byte) ((width >> 16) & 0xFF);
            bytes[index++] = (byte) ((width >> 8) & 0xFF);
            bytes[index++] = (byte) (width & 0xFF);

            bytes[index++] = (byte) ((height >> 24) & 0xFF);
            bytes[index++] = (byte) ((height >> 16) & 0xFF);
            bytes[index++] = (byte) ((height >> 8) & 0xFF);
            bytes[index++] = (byte) (height & 0xFF);

            // 写入 ARGB 数据
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pixel = image.getRGB(x, y);
                    bytes[index++] = (byte) ((pixel >> 24) & 0xFF); // A
                    bytes[index++] = (byte) ((pixel >> 16) & 0xFF); // R
                    bytes[index++] = (byte) ((pixel >> 8) & 0xFF); // G
                    bytes[index++] = (byte) (pixel & 0xFF); // B
                }
            }
            return bytes;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void main(String[] args) {
        String detModelPath = "/home/model/project/facenet/model/facedet.onnx";
        String facenetModelPath = "/home/model/project/facenet/model/facenet.onnx";
        String imagePath1 = "/home/model/project/facenet/image_test/zhou_01.jpg";
        String imagePath2 = "/home/model/project/facenet/image_test/zhou_02.jpg";

        if (args.length >= 4) {
            detModelPath = args[0];
            facenetModelPath = args[1];
            imagePath1 = args[2];
            imagePath2 = args[3];
        }

        try (FacePipeline pipeline = new FacePipeline(detModelPath, facenetModelPath)) {
            BufferedImage img1 = ImageIO.read(new File(imagePath1));
            BufferedImage img2 = ImageIO.read(new File(imagePath2));
            if (img1 == null || img2 == null) {
                System.out.println("输入图片读取失败");
                return;
            }

            FacePipeline.PipelineResult r1 = pipeline.extractLargestFaceEmbedding(img1);
            FacePipeline.PipelineResult r2 = pipeline.extractLargestFaceEmbedding(img2);
            if (!r1.success || !r2.success) {
                System.out.println("至少一张图片未检测到人脸");
                return;
            }

            FaceNet facenet = new FaceNet();
            float similarity = facenet.byte_cosine_similarity(r1.embedding, r2.embedding);
            System.out.printf("图1检测分数: %.4f, 图2检测分数: %.4f\n", r1.detScore, r2.detScore);
            System.out.printf("两张图片的相似度: %.4f\n", similarity);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static boolean read_image_to_img_array(String file_path, byte[][][] img_array) {
        // 初始化图片路径
        File image_file = new File(file_path);

        // 使用 ImageIO 读取图片
        BufferedImage image = null;
        try {
            image = ImageIO.read(image_file);
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        if (image == null) {
            return false;
        }

        // 拿到图片尺寸
        int width = image.getWidth();
        int height = image.getHeight();

        // 创建一个三维数组来保存图片数组
        // byte[][][] img_array = new byte[height][width][3]; 从外界传
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 获取 rgb 的值
                int pixel = image.getRGB(x, y);

                // 像素格式 0xAARRGGBB 32位
                // AA 位是 alpha 通道
                // RR 是红色分量
                // GG 是绿色分量
                // BB 是蓝色分量
                img_array[y][x][0] = (byte) ((pixel >> 16) & 0xFF);
                img_array[y][x][1] = (byte) ((pixel >> 8) & 0xFF);
                img_array[y][x][2] = (byte) (pixel & 0xFF);
            }
        }
        return true;
    }
}