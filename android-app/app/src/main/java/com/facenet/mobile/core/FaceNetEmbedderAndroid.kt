package com.facenet.mobile.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.sqrt

class FaceNetEmbedderAndroid(
    private val context: Context,
    private val modelAssetName: String = "facenet.onnx"
) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val outputName: String
    private val inputShape = longArrayOf(1, 3, 160, 160)
    private val inputBuffer: FloatBuffer =
        ByteBuffer.allocateDirect(1 * 3 * 160 * 160 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val inputTensor: OnnxTensor

    /** 像素读取缓冲，避免每次 embed 分配 IntArray(25600) */
    private val pixelBuf = IntArray(160 * 160)

    /**
     * 预分配 160×160 Bitmap + Canvas + Matrix，消除 BitmapUtils.resizeBitmap
     * 在 embed() 每次调用时的 Bitmap.createBitmap(160,160) 分配。
     */
    private val resizedBitmap = Bitmap.createBitmap(160, 160, Bitmap.Config.ARGB_8888)
    private val resizedCanvas = Canvas(resizedBitmap)
    private val resizeMatrix  = Matrix()
    private val resizeSrcRect = RectF()
    private val resizeDstRect = RectF(0f, 0f, 160f, 160f)  // 固定目标，永不变

    /** 除法转乘法：1/255 预计算，每像素省 1 次 fdiv */
    private val inv255 = 1f / 255f

    /** 推理输出 Set 缓存，避免每次 setOf(outputName) 分配 */
    private lateinit var outputSet: Set<String>

    /** 推理输入 Map 缓存，避免每次 Collections.singletonMap 分配 */
    private lateinit var inputMap: Map<String, OnnxTensor>

    init {
        val modelFile = materializeAsset(context, modelAssetName)
        // SessionOptions 实现 AutoCloseable，必须 use {} 关闭否则泄漏 native 内存
        session = OrtSession.SessionOptions().use { opts ->
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setIntraOpNumThreads(2)
            env.createSession(modelFile.absolutePath, opts)
        }
        inputName = session.inputNames.first()
        outputName = session.outputNames.first()
        inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)
        outputSet = setOf(outputName)
        inputMap = Collections.singletonMap(inputName, inputTensor)
    }

    fun embed(faceBitmap: Bitmap): FloatArray {
        // 直接绘入预分配 Bitmap，消除每次 BitmapUtils.resizeBitmap 的 Bitmap.createBitmap 分配
        resizeSrcRect.set(0f, 0f, faceBitmap.width.toFloat(), faceBitmap.height.toFloat())
        resizeMatrix.setRectToRect(resizeSrcRect, resizeDstRect, Matrix.ScaleToFit.FILL)
        resizedCanvas.drawBitmap(faceBitmap, resizeMatrix, null)
        preprocess(resizedBitmap, inputBuffer)
        session.run(inputMap, outputSet).use { result ->
            val tensor = result[0] as OnnxTensor
            val emb = FloatArray(512)
            val fb = tensor.floatBuffer
            fb.rewind()
            fb.get(emb)
            l2Normalize(emb)
            return emb
        }
    }

    private fun preprocess(bitmap: Bitmap, dst: FloatBuffer) {
        bitmap.getPixels(pixelBuf, 0, 160, 0, 0, 160, 160)
        val pixels = pixelBuf
        val plane = 160 * 160
        // idx 单调递增计数器：省去每像素 y*160+x 乘法；除法改乘法省去 fdiv
        var idx = 0
        for (y in 0 until 160) {
            for (x in 0 until 160) {
                val argb = pixels[idx]
                dst.put(idx,             ((argb shr 16) and 0xFF) * inv255)
                dst.put(plane + idx,     ((argb shr 8)  and 0xFF) * inv255)
                dst.put(2 * plane + idx, (argb           and 0xFF) * inv255)
                idx++
            }
        }
    }

    private fun l2Normalize(emb: FloatArray) {
        var norm = 0f
        for (v in emb) norm += v * v
        norm = sqrt(norm).coerceAtLeast(1e-12f)
        for (i in emb.indices) emb[i] /= norm
    }

    companion object {
        fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
            var dot = 0f
            var na = 0f
            var nb = 0f
            for (i in a.indices) {
                dot += a[i] * b[i]
                na += a[i] * a[i]
                nb += b[i] * b[i]
            }
            val denom = sqrt(na).coerceAtLeast(1e-12f) * sqrt(nb).coerceAtLeast(1e-12f)
            return dot / denom
        }
    }

    override fun close() {
        inputTensor.close()
        session.close()
        resizedBitmap.recycle()
    }
}
