import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collections;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import ai.onnxruntime.*;

public class FaceNet {

    private static final boolean DEBUG = true;

    // 以索引数组的方式进行索引
    public boolean read_image_from_bytes_direct(byte[] byteData, int width, int height, byte[][][] img_array) {
        /**
         * 这个 byteData 事实上是一个更大的 byte 数组，我们需要从这个数组中拿到部分数据，也就是 width*height*4 个字节的数据
         * 然后转换成 img_array 数组
         */
        int byteIndex = 0;
        // 按行读取像素数据
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // 每个像素 4 个字节: A, R, G, B
                // Java 的 byte 是有符号的，范围 -128 到 127，需要转换为 0-255
                byteIndex++; // 透明度分量直接跳过
                // 这里直接拷贝 byte 即可，后续使用时再 &0xFF 转 int
                img_array[y][x][0] = byteData[byteIndex++]; // 红色分量
                img_array[y][x][1] = byteData[byteIndex++]; // 绿色分量
                img_array[y][x][2] = byteData[byteIndex++]; // 蓝色分量
            }
        }
        return true;
    }

    public int mirror_clamp(int coord, int max) {
        // 使用镜像边界进行边界限制
        if (coord < 0) {
            return -coord - 1;
        } else if (coord >= max) {
            return 2 * max - coord - 1;
        } else {
            return coord;
        }
    }

    public int readN(InputStream in, byte[] b, int off, int len) throws IOException {
        int n = 0;
        while (n < len) {
            int count = in.read(b, off + n, len - n);
            if (count < 0) {
                if (n == 0) {
                    return count;
                } else {
                    break;
                }
            }
            n += count;
        }
        return n;
    }

    public int clamp(double value, int min, int max) {
        /**
         * 在插值计算的时候，有可能超出范围，需要对这个数值进行限制
         */
        return (int) Math.max(min, Math.min(value, max));
    }

    // bicubic 核参数（fast resize 版本回退后不再需要）
    // 兼容旧实现所需的临时数组（保留，但在新 fast 路径中不再使用）
    double alpha = -0.5;
    double[] weight_x = new double[4];
    double[] weight_y = new double[4];
    double[] temp_result = new double[3];

    // 传入一个 float[3] 数组来存储结果，直接归一化到 0-1
    public void get_pixel_for_bicubic_float(byte[][][] img_array, double x, double y,
            int width, int height, float[] result) {
        /**
         * 使用双三次插值函数获取对应的像素值，直接归一化到 0-1
         * img_array: 原始图像的RGB数组，格式是 [height][weight][3]
         * x，y: 浮点数坐标
         * width, height: 原始图像的宽和高
         * x，y: 这是一个浮点数坐标，由于对图片进行了缩放
         * width, height: 原始图像的宽和高
         * result: 用于存储结果的数组，长度为3，分别存储 R,G,B 分量，归一化到 0-1
         * 按照 python 缩放图片的方式进行缩放，保持一致
         * 在 python 中使用的是 PIL 库中的 resize 方法，需要传入的参数是
         * size: (width, height),表示需要缩放后的尺寸
         * resample: 表示的是重采样滤波器，默认使用的是 BICUBIC 插值算法进行重采样(双三次插值函数)
         * box： 表示需要裁减的区域，我们默认是整张图片，因为前端传入的是一张已经裁减好的人脸图片
         * reducing_gap：默认是 None,无优化
         * 我们需要实现一个重采样滤波器的方法， 使用BICUBIC插值算法进行重采样，需要定义一个和双三次核函数
         * 双三次插值函数，是这样一个公式
         * 1 - (alpha + 3) * |x|^2 + (alpha + 2) * |x|^3 |x| <= 1
         * S(x) = 4 * alpha - 8 * alpha |x| + 5* alpha * |x|^2-alpha * |x|^3 1 < |x|<2
         * 0 otherwise
         * 一般来说， alpha 取 -0.5 或者 -0.75
         *
         */

        int x_int = (int) Math.floor(x);
        int y_int = (int) Math.floor(y);

        // 将 BICUBIC 直接写在函数内，不再调用函数
        // double alpha = -0.5; 这是一个常量，定义到外面
        for (int i = -1; i <= 2; i++) {
            double dx = Math.abs(x - (x_int + i));
            if (dx <= 1) {
                weight_x[i + 1] = (alpha + 2) * dx * dx * dx - (alpha + 3) * dx * dx + 1;
            } else if (dx < 2) {
                weight_x[i + 1] = alpha * dx * dx * dx - 5 * alpha * dx * dx + 8 * alpha * dx - 4 * alpha;
            } else {
                weight_x[i + 1] = 0.0;
            }

            double dy = Math.abs(y - (y_int + i));
            if (dy <= 1) {
                weight_y[i + 1] = (alpha + 2) * dy * dy * dy - (alpha + 3) * dy * dy + 1;
            } else if (dy < 2) {
                weight_y[i + 1] = alpha * dy * dy * dy - 5 * alpha * dy * dy + 8 * alpha * dy - 4 * alpha;
            } else {
                weight_y[i + 1] = 0.0;
            }
        }

        temp_result[0] = 0.0;
        temp_result[1] = 0.0;
        temp_result[2] = 0.0;
        double total_weight = 0.0;

        // 求和
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                double weight = weight_x[i] * weight_y[j];
                total_weight += weight;

                int p_x = mirror_clamp(x_int + i - 1, width);
                int p_y = mirror_clamp(y_int + j - 1, height);

                for (int c = 0; c < 3; c++) {
                    temp_result[c] += (img_array[p_y][p_x][c] & 0xFF) * weight;
                }
            }
        }
        // 直接归一化到 0-1 ，先 clamp 确保值在 0-255 范围内
        for (int c = 0; c < 3; c++) {
            result[c] = (float) (clamp(temp_result[c] / total_weight, 0, 255) / 255.0);
        }
    }

    final int tar_height = 160;
    final int tar_width = 160;
    double[] src_xs = new double[tar_width];
    double[] src_ys = new double[tar_height];
    float[] pixel_value_float = new float[3]; // 用来存放映射后的每个像素的值

    /**
     * bilinear resize：直接输出归一化后的 float 到 batchArray（0~1）
     * 相比 bicubic 更快，且对“输入尺寸每帧变化”的场景更稳定。
     */
    public void resize_image_bilinear_float(byte[][][] img_array, int srcHeight, int srcWidth,
            float[][][][] batchArray) {
        float scaleX = (float) srcWidth / tar_width;
        float scaleY = (float) srcHeight / tar_height;

        // 预计算目标坐标映射到源坐标（与原 bicubic 保持同样的 “center-aligned” 取样方式）
        for (int x = 0; x < tar_width; x++) {
            double sx = (x + 0.5f) * scaleX - 0.5f;
            if (sx < 0)
                sx = 0;
            if (sx > srcWidth - 1)
                sx = srcWidth - 1;
            src_xs[x] = sx;
        }
        for (int y = 0; y < tar_height; y++) {
            double sy = (y + 0.5f) * scaleY - 0.5f;
            if (sy < 0)
                sy = 0;
            if (sy > srcHeight - 1)
                sy = srcHeight - 1;
            src_ys[y] = sy;
        }

        for (int y = 0; y < tar_height; y++) {
            float srcY = (float) src_ys[y];
            int y0 = (int) srcY;
            int y1 = y0 + 1;
            if (y1 >= srcHeight)
                y1 = srcHeight - 1;
            float dy = srcY - y0;

            for (int x = 0; x < tar_width; x++) {
                float srcX = (float) src_xs[x];
                int x0 = (int) srcX;
                int x1 = x0 + 1;
                if (x1 >= srcWidth)
                    x1 = srcWidth - 1;
                float dx = srcX - x0;

                // separable bilinear
                for (int c = 0; c < 3; c++) {
                    float p00 = (img_array[y0][x0][c] & 0xFF);
                    float p10 = (img_array[y0][x1][c] & 0xFF);
                    float p01 = (img_array[y1][x0][c] & 0xFF);
                    float p11 = (img_array[y1][x1][c] & 0xFF);

                    float top = p00 + dx * (p10 - p00);
                    float bot = p01 + dx * (p11 - p01);
                    float v = top + dy * (bot - top);

                    if (v < 0)
                        v = 0;
                    else if (v > 255)
                        v = 255;
                    batchArray[0][c][y][x] = v / 255.0f;
                }
            }
        }
    }

    // [tar_height][tar_width][3] 直接输出归一化后的 float 数组到 batchArray
    public void resize_image_bicubic_float(byte[][][] img_array, int srcHeight, int srcWidth,
            float[][][][] batchArray) {
        /**
         * 开始对图片中的所有像素值进行映射，直接归一化到 0-1 并写入 batchArray
         * img_array：原始图片的RGB数组
         * batchArray：输出batch数组 [1][3][height][width]
         * srcHeight：原始图片高度
         * srcWidth：原始图片宽度
         */

        // 计算缩放比例
        double scale_x = (double) srcWidth / tar_width;
        double scale_y = (double) srcHeight / tar_height;

        // 预计算所有 src_x 和 src_y 以减少每次 clamp

        for (int x = 0; x < tar_width; x++) {
            src_xs[x] = Math.max(0, Math.min((x + 0.5) * scale_x - 0.5, srcWidth - 1));
        }
        for (int y = 0; y < tar_height; y++) {
            src_ys[y] = Math.max(0, Math.min((y + 0.5) * scale_y - 0.5, srcHeight - 1));
        }

        // 对每个像素进行映射
        for (int y = 0; y < tar_height; y++) {
            for (int x = 0; x < tar_width; x++) {
                // 使用预计算的坐标
                double src_x = src_xs[x];
                double src_y = src_ys[y];

                // 使用插值算法得到映射后的像素值，直接归一化
                get_pixel_for_bicubic_float(img_array, src_x, src_y, srcWidth, srcHeight, pixel_value_float);

                for (int c = 0; c < 3; c++) {
                    batchArray[0][c][y][x] = pixel_value_float[c];
                }
            }
        }
    }

    // 需要传入一个一维数组进行保存 []
    public void l2_normalized(float[] emb) {
        /**
         * 按照python的写法，需要对推理出来的数组进行l2归一化
         * 首先计算这个数组的模长，然后将每个值除以这个模长
         */
        float norm = 0.0f;
        for (int i = 0; i < 512; i++) {
            norm += emb[i] * emb[i];
        }
        // 计算模长
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < 512; i++) {
            emb[i] = emb[i] / norm;
        }
    }

    public float cosine_similarity(float[] emb_1, float[] emb_2) {
        /**
         * 用于计算余弦相似度度量
         * 需要传入两个向量，也就是数组，[a1,a2,a3,a4,a5...],[b1,b2,b3,b4,b5...]
         * 这两个向量的维度是一致的，计算方法为
         * a1*b1 + a2*b2 + a3*b3 + ... = dot
         * ai^2 + a2^2 + a3^2 + ... = ai
         * math.sqrt(ai)
         * b1^2 + b2^2 + b3^2 + ... = bi
         * math.sqrt(bi)
         * result = dot/(math.sqrt(ai)*math.sqrt(bi))
         * 这个值在 0-1 之间，越接近1就表明这两个向量越相似
         */

        float dot = 0.0f;
        float norm_1 = 0.0f;
        float norm_2 = 0.0f;
        for (int i = 0; i < 512; i++) {
            dot += emb_1[i] * emb_2[i];
            norm_1 += emb_1[i] * emb_1[i];
            norm_2 += emb_2[i] * emb_2[i];
        }

        norm_1 = (float) Math.sqrt(norm_1);
        norm_2 = (float) Math.sqrt(norm_2);
        return dot / (norm_1 * norm_2);
    }

    // 填充 FloatBuffer
    public void flatten4D(float[][][][] batchArray, FloatBuffer buffer) {
        for (int b = 0; b < batchArray.length; b++) {
            for (int c = 0; c < batchArray[b].length; c++) {
                for (int h = 0; h < batchArray[b][c].length; h++) {
                    for (int w = 0; w < batchArray[b][c][h].length; w++) {
                        buffer.put(batchArray[b][c][h][w]);
                    }
                }
            }
        }
    }

    long[] input_shape = { 1, 3, 160, 160 };

    long[] output_shape = { 1, 512 };

    long getShapeSize(long[] shape) {
        long c = 1;
        for (int i = 0; i < shape.length; i++) {
            c *= shape[i];
        }
        return c;
    }

    /**
     * 执行 ONNX 模型推理，提取图像嵌入向量
     *
     * @param session      ONNX 会话对象
     * @param inputTensor  输入张量，包含预处理后的图像数据
     * @param inputName    输入张量的名称
     * @param outputTensor 输出张量，用于接收模型输出
     * @param outputName   输出张量的名称
     * @throws OrtException ONNX 运行时异常
     */
    public void getEmbedding(OrtSession session, OnnxTensor inputTensor, String inputName, OnnxTensor outputTensor,
            String outputName) throws OrtException {
        // 执行模型推理，将输入张量传递给模型，输出结果写入输出张量
        session.run(Collections.singletonMap(inputName, inputTensor),
                Collections.singletonMap(outputName, outputTensor));
        // 注意：推理结果已直接写入 outputTensor 的底层缓冲区，无需额外复制
    }

    /**
     * 将 FloatBuffer 中的数据重塑为二维 float 数组
     * 
     * @param input  输入的 FloatBuffer，包含一维数据
     * @param output 输出的二维数组
     */
    public void reshape2D(FloatBuffer input, float[][] output) {
        int rows = output.length;
        int cols = output[0].length;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][j] = input.get();
            }
        }
    }

    float[] byte_to_float_a = new float[512];
    float[] byte_to_float_b = new float[512];

    public float byte_cosine_similarity(byte[] byteDataA, byte[] byteDataB) {
        // 封装成一个使用 byte 数组计算余弦相似度的接口
        // 将 byte 数组转换为 float 数组

        byteToFloat(byteDataA, byte_to_float_a);
        byteToFloat(byteDataB, byte_to_float_b);

        // 计算余弦相似度
        return cosine_similarity(byte_to_float_a, byte_to_float_b);
    }

    public void byteToFloat(byte[] byteData, float[] floatData) {
        // 将 byte 数组转换成 float 类型数组
        for (int i = 0; i < floatData.length; i++) {
            int intBits = ((byteData[i * 4] & 0xFF) << 24) |
                    ((byteData[i * 4 + 1] & 0xFF) << 16) |
                    ((byteData[i * 4 + 2] & 0xFF) << 8) |
                    (byteData[i * 4 + 3] & 0xFF);
            floatData[i] = Float.intBitsToFloat(intBits);
        }
    }

    public void floatToByte(float[] floatData, byte[] byteData) {
        // 将 float 数组转换成 byte 类型数组
        for (int i = 0; i < floatData.length; i++) {
            int intBits = Float.floatToIntBits(floatData[i]);
            byteData[i * 4] = (byte) ((intBits >> 24) & 0xFF);
            byteData[i * 4 + 1] = (byte) ((intBits >> 16) & 0xFF);
            byteData[i * 4 + 2] = (byte) ((intBits >> 8) & 0xFF);
            byteData[i * 4 + 3] = (byte) (intBits & 0xFF);
        }
    }

    /**
     * 处理图像流，提取每张图像的嵌入向量并输出
     *
     * @param modelPath          ONNX 模型文件路径
     * @param previewImageWidth  预览图像宽度，用于缓冲区大小
     * @param previewImageHeight 预览图像高度，用于缓冲区大小
     * @param inputStream        输入流，包含图像数据
     * @param outputStream       输出流，用于写入嵌入向量
     */
    public void process(String modelPath, int previewImageWidth, int previewImageHeight, InputStream inputStream,
            OutputStream outputStream) {

        // 初始化 ONNX 环境和会话选项
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        try {
            // 创建 ONNX 会话
            OrtSession session = env.createSession(modelPath, opts);

            ByteBuffer lengthBuffer = ByteBuffer.allocate(4);
            byte[] inputByteBuffer = new byte[previewImageWidth * previewImageHeight * 4]; // ARGB 格式输入缓冲区
            byte[][][] imageArray = new byte[previewImageHeight][previewImageWidth][3]; // 图像数组
            float[][][][] batchArray = new float[1][3][tar_height][tar_width]; // 批处理数组 [1][3][160][160]
            // 创建输入缓冲区和张量
            FloatBuffer inputTensorBuffer = ByteBuffer.allocateDirect((int) getShapeSize(input_shape) * 4)
                    .order(ByteOrder.nativeOrder()).asFloatBuffer();
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputTensorBuffer, input_shape);
            // 创建输出缓冲区和张量
            FloatBuffer outputTensorBuffer = ByteBuffer.allocateDirect((int) getShapeSize(output_shape) * 4)
                    .order(ByteOrder.nativeOrder()).asFloatBuffer();
            OnnxTensor outputTensor = OnnxTensor.createTensor(env, outputTensorBuffer, output_shape);
            //
            float[][] embeddings = new float[1][512]; // 嵌入向量数组 [1][512]
            //
            byte[] outputByteBuffer = new byte[512 * 4]; // 输出字节缓冲区

            try {
                while (true) {
                    // 读取图像宽度
                    int rW = readN(inputStream, lengthBuffer.array(), 0, lengthBuffer.limit());
                    if (rW < 0)
                        break;
                    int imageWidth = lengthBuffer.getInt();
                    lengthBuffer.rewind();

                    // 读取图像高度
                    int rH = readN(inputStream, lengthBuffer.array(), 0, lengthBuffer.limit());
                    if (rH < 0)
                        break;
                    int imageHeight = lengthBuffer.getInt();
                    lengthBuffer.rewind();

                    int need = imageWidth * imageHeight * 4;

                    // 读取图像数据
                    int bytesRead = readN(inputStream, inputByteBuffer, 0, need);
                    if (bytesRead < 0)
                        break;
                    if (bytesRead < need)
                        break;
                    if (DEBUG) {
                        System.out.printf("读取到图片数据: 宽=%d, 高=%d, 字节数=%d\n", imageWidth, imageHeight, bytesRead);
                    }

                    // 处理单帧图像
                    processFrame(
                            session, imageWidth, imageHeight,
                            inputByteBuffer, imageArray, batchArray,
                            inputTensorBuffer, inputTensor,
                            outputTensorBuffer, outputTensor,
                            embeddings, outputByteBuffer);

                    // 将嵌入向量写入输出流
                    outputStream.write(outputByteBuffer);

                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }

    public void processFrame() {

    }

    /**
     * 处理单帧图像：从字节数据解码、缩放、推理、归一化并输出
     *
     * @param session            ONNX 会话
     * @param imageWidth         图像宽度
     * @param imageHeight        图像高度
     * @param input              输入字节数组（ARGB 格式）
     * @param output             输出字节数组（嵌入向量）
     * @param inputTensor        输入张量
     * @param embeddings         嵌入向量数组
     * @param imageArray         图像像素数组
     * @param batchArray         批处理数组
     * @param outputTensor       输出张量
     * @param outputTensorBuffer 输出缓冲区
     * @param inputTensorBuffer  输入缓冲区
     */
    void processFrame(
            // ---- 输入（原始数据/中间图像/模型输入）----
            OrtSession session,
            int imageWidth, int imageHeight,
            byte[] input,
            byte[][][] imageArray,
            float[][][][] batchArray,
            FloatBuffer inputTensorBuffer, OnnxTensor inputTensor,
            // ---- 输出（模型输出/结果）----
            FloatBuffer outputTensorBuffer, OnnxTensor outputTensor,
            float[][] embeddings,
            byte[] output) throws OrtException {

        // 从字节数据解码图像到数组
        read_image_from_bytes_direct(input, imageWidth, imageHeight, imageArray);

        // 缩放图像到目标尺寸：切换为 bilinear（更快）
        long resizeStart = System.nanoTime();
        // 使用双三次插值进行缩放resize_image_bicubic_float
        // 使用双线性插值进行缩放resize_image_bilinear_float
        resize_image_bilinear_float(imageArray, imageHeight, imageWidth, batchArray);
        long resizeTime = System.nanoTime();
        if (DEBUG) {
            System.out.printf("图片缩放完成 (%.3f ms)\n", (resizeTime - resizeStart) / 1e6);
        }

        // 填充输入缓冲区
        flatten4D(batchArray, inputTensorBuffer);
        inputTensorBuffer.rewind();

        // 执行推理
        long inferenceStart = System.nanoTime();
        getEmbedding(session, inputTensor, "input", outputTensor, "embedding");
        reshape2D(outputTensorBuffer, embeddings);
        outputTensorBuffer.rewind();
        long inferenceTime = System.nanoTime();
        if (DEBUG) {
            System.out.printf("特征提取完成 (%.3f ms)\n", (inferenceTime - inferenceStart) / 1e6);
        }

        // 对嵌入向量进行 L2 归一化
        long l2Start = System.nanoTime();
        l2_normalized(embeddings[0]);
        long l2Time = System.nanoTime();
        if (DEBUG) {
            System.out.printf("L2归一化完成 (%.3f ms)\n", (l2Time - l2Start) / 1e6);
        }

        // 将归一化后的 float 数组转换为字节数组
        floatToByte(embeddings[0], output);
    }

}