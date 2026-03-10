import ai.onnxruntime.OrtException;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 基础人脸库：
 * 1) 注册：图片 -> embedding
 * 2) 识别：查询 embedding 与库中向量做余弦匹配
 * 3) 持久化：二进制文件 save/load
 */
public class FaceRegistry implements Closeable {

    public static class MatchResult {
        public boolean detected;
        public String userId;
        public float similarity;
        public float detScore;
    }

    private static final byte[] MAGIC = new byte[]{'F', 'R', 'E', 'G', '1'};
    private static final int EMBEDDING_BYTES = 512 * 4;

    private final FacePipeline pipeline;
    private final FaceNet faceNet = new FaceNet();
    private final Map<String, List<byte[]>> db = new LinkedHashMap<>();

    public FaceRegistry(String detModelPath, String facenetModelPath) throws OrtException {
        this.pipeline = new FacePipeline(detModelPath, facenetModelPath);
    }

    /**
     * 注册一张人脸到指定用户。
     * 检测不到人脸时返回 false。
     */
    public boolean register(String userId, BufferedImage image) throws OrtException {
        if (userId == null || userId.isEmpty() || image == null) {
            return false;
        }
        FacePipeline.PipelineResult r = pipeline.extractLargestFaceEmbedding(image);
        if (!r.success || r.embedding == null || r.embedding.length != EMBEDDING_BYTES) {
            return false;
        }
        List<byte[]> embs = db.computeIfAbsent(userId, k -> new ArrayList<>());
        embs.add(r.embedding.clone());
        return true;
    }

    /**
     * 识别：返回与库中最相似的用户。
     * similarity < threshold 时 userId 为空字符串。
     */
    public MatchResult identify(BufferedImage image, float threshold) throws OrtException {
        MatchResult result = new MatchResult();
        result.userId = "";
        result.similarity = -1.0f;
        result.detected = false;

        FacePipeline.PipelineResult query = pipeline.extractLargestFaceEmbedding(image);
        if (!query.success || query.embedding == null) {
            return result;
        }
        result.detected = true;
        result.detScore = query.detScore;

        for (Map.Entry<String, List<byte[]>> entry : db.entrySet()) {
            String userId = entry.getKey();
            List<byte[]> embs = entry.getValue();
            for (byte[] emb : embs) {
                float sim = faceNet.byte_cosine_similarity(query.embedding, emb);
                if (sim > result.similarity) {
                    result.similarity = sim;
                    result.userId = userId;
                }
            }
        }

        if (result.similarity < threshold) {
            result.userId = "";
        }
        return result;
    }

    public int userCount() {
        return db.size();
    }

    public int embeddingCount() {
        int count = 0;
        for (List<byte[]> embs : db.values()) {
            count += embs.size();
        }
        return count;
    }

    public void clear() {
        db.clear();
    }

    /**
     * 二进制格式：
     * MAGIC(5) + userCount(int)
     *   userId(UTF) + embCount(int)
     *     embLen(int) + embBytes
     */
    public void save(String filePath) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath)))) {
            out.write(MAGIC);
            out.writeInt(db.size());
            for (Map.Entry<String, List<byte[]>> entry : db.entrySet()) {
                out.writeUTF(entry.getKey());
                List<byte[]> embs = entry.getValue();
                out.writeInt(embs.size());
                for (byte[] emb : embs) {
                    if (emb == null) {
                        out.writeInt(0);
                        continue;
                    }
                    out.writeInt(emb.length);
                    out.write(emb);
                }
            }
            out.flush();
        }
    }

    public void load(String filePath) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            byte[] magic = new byte[MAGIC.length];
            in.readFully(magic);
            for (int i = 0; i < MAGIC.length; i++) {
                if (magic[i] != MAGIC[i]) {
                    throw new IOException("人脸库文件格式错误");
                }
            }

            db.clear();
            int userSize = in.readInt();
            for (int i = 0; i < userSize; i++) {
                String userId = in.readUTF();
                int embCount = in.readInt();
                List<byte[]> embs = new ArrayList<>(embCount);
                for (int j = 0; j < embCount; j++) {
                    int len = in.readInt();
                    if (len <= 0) {
                        embs.add(new byte[0]);
                        continue;
                    }
                    byte[] emb = new byte[len];
                    in.readFully(emb);
                    embs.add(emb);
                }
                db.put(userId, embs);
            }
        }
    }

    /**
     * 导出可读 JSON，便于查看存储内容。
     * embedding 以 512 维 float 数组输出。
     */
    public void saveAsJson(String filePath) throws IOException {
        StringBuilder sb = new StringBuilder(1024);
        sb.append("{\n");
        sb.append("  \"version\": \"FREG1\",\n");
        sb.append("  \"userCount\": ").append(userCount()).append(",\n");
        sb.append("  \"embeddingCount\": ").append(embeddingCount()).append(",\n");
        sb.append("  \"users\": [\n");

        boolean firstUser = true;
        for (Map.Entry<String, List<byte[]>> entry : db.entrySet()) {
            if (!firstUser) {
                sb.append(",\n");
            }
            firstUser = false;

            String userId = entry.getKey();
            List<byte[]> embs = entry.getValue();
            sb.append("    {\n");
            sb.append("      \"userId\": \"").append(escapeJson(userId)).append("\",\n");
            sb.append("      \"embeddings\": [\n");

            for (int i = 0; i < embs.size(); i++) {
                if (i > 0) {
                    sb.append(",\n");
                }
                byte[] embBytes = embs.get(i);
                float[] emb = new float[512];
                if (embBytes != null && embBytes.length >= EMBEDDING_BYTES) {
                    faceNet.byteToFloat(embBytes, emb);
                }
                sb.append("        [");
                for (int j = 0; j < emb.length; j++) {
                    if (j > 0) {
                        sb.append(", ");
                    }
                    sb.append(emb[j]);
                }
                sb.append("]");
            }
            sb.append("\n");
            sb.append("      ]\n");
            sb.append("    }");
        }

        sb.append("\n");
        sb.append("  ]\n");
        sb.append("}\n");

        try (Writer writer = new OutputStreamWriter(new FileOutputStream(filePath), StandardCharsets.UTF_8)) {
            writer.write(sb.toString());
            writer.flush();
        }
    }

    private static String escapeJson(String s) {
        if (s == null) {
            return "";
        }
        StringBuilder out = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '\"':
                    out.append("\\\"");
                    break;
                case '\\':
                    out.append("\\\\");
                    break;
                case '\b':
                    out.append("\\b");
                    break;
                case '\f':
                    out.append("\\f");
                    break;
                case '\n':
                    out.append("\\n");
                    break;
                case '\r':
                    out.append("\\r");
                    break;
                case '\t':
                    out.append("\\t");
                    break;
                default:
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
            }
        }
        return out.toString();
    }

    @Override
    public void close() throws IOException {
        pipeline.close();
    }
}
