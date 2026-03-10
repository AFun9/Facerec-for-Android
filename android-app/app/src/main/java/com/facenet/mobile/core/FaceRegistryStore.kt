package com.facenet.mobile.core

import android.content.Context
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

class FaceRegistryStore(context: Context) {

    private val db: LinkedHashMap<String, MutableList<FloatArray>> = linkedMapOf()
    private val magic = byteArrayOf('F'.code.toByte(), 'R'.code.toByte(), 'E'.code.toByte(), 'G'.code.toByte(), '1'.code.toByte())
    private val baseDir: File = File(context.filesDir, "facereg").apply { mkdirs() }

    fun register(userId: String, embedding: FloatArray): Boolean {
        if (userId.isBlank() || embedding.size != 512) return false
        val list = db.getOrPut(userId) { mutableListOf() }
        list += embedding.copyOf()
        return true
    }

    /**
     * 识别人脸。
     * 由于 embedding 已经过 L2 归一化，余弦相似度 = 点积（分母恒为 1.0），
     * 省去两次 sqrt，大幅降低运算量。
     */
    fun identify(query: FloatArray, threshold: Float = 0.45f): MatchResult {
        if (query.size != 512) return MatchResult(false, "", -1f)
        var bestUser = ""
        var bestScore = -1f
        db.forEach { (user, embs) ->
            embs.forEach { emb ->
                val sim = dotProduct(query, emb)
                if (sim > bestScore) {
                    bestScore = sim
                    bestUser = user
                }
            }
        }
        if (bestScore < threshold) bestUser = ""
        return MatchResult(detected = true, userId = bestUser, similarity = bestScore)
    }

    /** 删除指定用户的所有注册数据，返回是否存在该用户。磁盘写入由调用方负责调度到 IO 线程。 */
    fun deleteUser(userId: String): Boolean {
        return db.remove(userId) != null
    }

    fun userCount(): Int = db.size
    fun embeddingCount(): Int = db.values.sumOf { it.size }

    fun saveBin(fileName: String = "registry.bin"): File {
        val f = File(baseDir, fileName)
        DataOutputStream(BufferedOutputStream(f.outputStream())).use { out ->
            out.write(magic)
            out.writeInt(db.size)
            db.forEach { (userId, embs) ->
                out.writeUTF(userId)
                out.writeInt(embs.size)
                embs.forEach { emb ->
                    out.writeInt(emb.size)
                    emb.forEach { out.writeFloat(it) }
                }
            }
            out.flush()
        }
        return f
    }

    fun loadBin(fileName: String = "registry.bin"): Boolean {
        val f = File(baseDir, fileName)
        if (!f.exists()) return false
        DataInputStream(BufferedInputStream(f.inputStream())).use { input ->
            val m = ByteArray(magic.size)
            input.readFully(m)
            if (!m.contentEquals(magic)) return false
            db.clear()
            val users = input.readInt()
            repeat(users) {
                val user = input.readUTF()
                val embCount = input.readInt()
                val list = mutableListOf<FloatArray>()
                repeat(embCount) {
                    val len = input.readInt()
                    val emb = FloatArray(len)
                    for (i in 0 until len) emb[i] = input.readFloat()
                    list += emb
                }
                db[user] = list
            }
        }
        return true
    }

    fun saveJson(fileName: String = "registry.json"): File {
        val f = File(baseDir, fileName)
        val sb = StringBuilder(1024)
        sb.append("{\n")
        sb.append("  \"version\": \"FREG1\",\n")
        sb.append("  \"userCount\": ").append(userCount()).append(",\n")
        sb.append("  \"embeddingCount\": ").append(embeddingCount()).append(",\n")
        sb.append("  \"users\": [\n")

        var firstUser = true
        db.forEach { (userId, embs) ->
            if (!firstUser) sb.append(",\n")
            firstUser = false
            sb.append("    {\n")
            sb.append("      \"userId\": \"").append(escapeJson(userId)).append("\",\n")
            sb.append("      \"embeddings\": [\n")
            embs.forEachIndexed { i, emb ->
                if (i > 0) sb.append(",\n")
                sb.append("        [")
                emb.forEachIndexed { j, v ->
                    if (j > 0) sb.append(", ")
                    sb.append(v)
                }
                sb.append("]")
            }
            sb.append("\n")
            sb.append("      ]\n")
            sb.append("    }")
        }

        sb.append("\n  ]\n}\n")
        f.writeText(sb.toString())
        return f
    }

    fun listRegisteredUsers(): List<String> = db.keys.toList()

    /** 两个 L2 归一化向量的点积，等价于余弦相似度但无需开方运算 */
    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    private fun escapeJson(s: String): String {
        val out = StringBuilder(s.length + 8)
        for (c in s) {
            when (c) {
                '\"'       -> out.append("\\\"")
                '\\'       -> out.append("\\\\")
                '\b'       -> out.append("\\b")
                '\u000C'   -> out.append("\\f")
                '\n'       -> out.append("\\n")
                '\r'       -> out.append("\\r")
                '\t'       -> out.append("\\t")
                else -> {
                    if (c.code < 0x20) out.append(String.format("\\u%04x", c.code))
                    else out.append(c)
                }
            }
        }
        return out.toString()
    }
}
