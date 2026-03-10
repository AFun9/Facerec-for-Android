package com.facenet.mobile.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class FaceDetectorAndroid(
    private val context: Context,
    private val modelAssetName: String = "facedet.onnx",
    private val inputWidth: Int = 640,
    private val inputHeight: Int = 640,
    private val inputMean: Float = 127.5f,
    private val inputStd: Float = 128f
) : AutoCloseable {

    /** 除法转乘法：1/inputStd 预先计算，hot loop 每像素省 1 次 fdiv */
    private val invStd = 1f / inputStd

    private data class AnchorGrid(val cx: FloatArray, val cy: FloatArray)

    // strides[0]=8, strides[1]=16, strides[2]=32
    private val strides = intArrayOf(8, 16, 32)
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    private val inputShape = longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
    private val inputBuffer: FloatBuffer =
        ByteBuffer.allocateDirect(1 * 3 * inputHeight * inputWidth * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val inputTensor: OnnxTensor

    // ── 预分配缓冲：消除每帧 GC ───────────────────────────────────────────────

    /** 640×640 像素读取缓冲，避免每帧 new IntArray(409600) */
    private val pixelBuf = IntArray(inputWidth * inputHeight)

    /** preprocess 结果：[scale, padX, padY] */
    private val preprocessMeta = FloatArray(3)

    /** 推理输入 Map 缓存 */
    private lateinit var inputMap: Map<String, OnnxTensor>

    /** anchor 网格，数组索引（0=stride8, 1=stride16, 2=stride32）替换 HashMap */
    private val anchorGridArr = Array(3) { AnchorGrid(FloatArray(0), FloatArray(0)) }

    /** ONNX 输出引用，按 stride 索引存储，替换每帧三次 HashMap 创建 */
    private val scoresRef = arrayOfNulls<Array<FloatArray>>(3)
    private val boxesRef  = arrayOfNulls<Array<FloatArray>>(3)

    /** 候选检测框列表，每帧 clear() 后复用 */
    private val detCandidates = ArrayList<FaceDetection>(128)

    /** NMS 标记数组，预分配避免每次 new BooleanArray */
    private val nmsRemoved = BooleanArray(256)

    /** kps 在当前 pipeline 中未使用，所有检测框共享此空数组 */
    private val emptyKps = FloatArray(10)

    // ── 画布：预分配 Matrix 和 RectF 避免 createScaledBitmap ──────────────────

    private val letterbox = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas = Canvas(letterbox)
    private val blackPaint = Paint().apply { color = Color.BLACK }
    /** 预分配 Matrix 和 RectF，彻底消除 Bitmap.createScaledBitmap 分配 */
    private val preprocessMatrix = Matrix()
    private val srcRectF = RectF()
    private val dstRectF = RectF()

    init {
        val modelFile = materializeAsset(context, modelAssetName)
        // SessionOptions 实现 AutoCloseable，必须关闭否则会泄漏 native 内存
        session = OrtSession.SessionOptions().use { opts ->
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setIntraOpNumThreads(2)
            env.createSession(modelFile.absolutePath, opts)
        }
        inputName = session.inputNames.first()
        inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)
        inputMap = Collections.singletonMap(inputName, inputTensor)
        for (i in strides.indices) {
            anchorGridArr[i] = buildAnchorGrid(strides[i], inputWidth, inputHeight)
        }
    }

    fun detect(
        bitmap: Bitmap,
        scoreThresh: Float = 0.45f,
        nmsIouThresh: Float = 0.4f,
        topK: Int = 50
    ): List<FaceDetection> {
        preprocess(bitmap)
        detCandidates.clear()
        session.run(inputMap).use { result ->
            decodeOutputs(result, bitmap.width, bitmap.height, scoreThresh, detCandidates)
        }
        return nms(detCandidates, nmsIouThresh, topK)
    }

    private fun preprocess(src: Bitmap) {
        val srcW = src.width
        val srcH = src.height
        val scale = min(inputWidth.toFloat() / srcW, inputHeight.toFloat() / srcH)
        val newW = max(1, (srcW * scale).toInt())
        val newH = max(1, (srcH * scale).toInt())
        val padX = (inputWidth - newW) / 2
        val padY = (inputHeight - newH) / 2

        // 用预分配的 Matrix 直接在 letterboxCanvas 上缩放绘制，不再创建中间 Bitmap
        letterboxCanvas.drawRect(0f, 0f, inputWidth.toFloat(), inputHeight.toFloat(), blackPaint)
        srcRectF.set(0f, 0f, srcW.toFloat(), srcH.toFloat())
        dstRectF.set(padX.toFloat(), padY.toFloat(), (padX + newW).toFloat(), (padY + newH).toFloat())
        preprocessMatrix.setRectToRect(srcRectF, dstRectF, Matrix.ScaleToFit.FILL)
        letterboxCanvas.drawBitmap(src, preprocessMatrix, null)

        val plane = inputWidth * inputHeight
        letterbox.getPixels(pixelBuf, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        // idx 单调递增计数器替代 y*width+x，省去每像素一次乘法
        var idx = 0
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val argb = pixelBuf[idx]
                val r = ((argb shr 16) and 0xFF).toFloat()
                val g = ((argb shr 8) and 0xFF).toFloat()
                val b = (argb and 0xFF).toFloat()
                inputBuffer.put(idx,             (r - inputMean) * invStd)
                inputBuffer.put(plane + idx,     (g - inputMean) * invStd)
                inputBuffer.put(2 * plane + idx, (b - inputMean) * invStd)
                idx++
            }
        }

        preprocessMeta[0] = scale
        preprocessMeta[1] = padX.toFloat()
        preprocessMeta[2] = padY.toFloat()
    }

    private fun decodeOutputs(
        result: OrtSession.Result,
        origW: Int,
        origH: Int,
        scoreThresh: Float,
        out: MutableList<FaceDetection>
    ) {
        // 只提取 scores 和 boxes，kps 不使用（跳过 kpsRef 赋值和检查）
        for (i in 0 until result.size()) {
            val value = result[i].value
            if (value !is Array<*>) continue
            @Suppress("UNCHECKED_CAST")
            val data = value as Array<FloatArray>
            if (data.isEmpty()) continue
            val si = when (data.size) { 12800 -> 0; 3200 -> 1; 800 -> 2; else -> -1 }
            if (si < 0) continue
            when (data[0].size) {
                1 -> scoresRef[si] = data
                4 -> boxesRef[si]  = data
            }
        }

        val scale  = preprocessMeta[0]
        val padX   = preprocessMeta[1]
        val padY   = preprocessMeta[2]
        val origWf = (origW - 1).toFloat()
        val origHf = (origH - 1).toFloat()

        for (si in 0..2) {
            val scores = scoresRef[si] ?: continue
            val boxes  = boxesRef[si]  ?: continue
            val grid   = anchorGridArr[si]
            val stride = strides[si]

            for (i in scores.indices) {
                val score = toProbability(scores[i][0])
                if (score < scoreThresh) continue
                val cx = grid.cx[i]
                val cy = grid.cy[i]
                // 缓存内层数组引用，避免 boxes[i] 重复二次解引用
                val boxRow = boxes[i]
                val l = boxRow[0] * stride
                val t = boxRow[1] * stride
                val r = boxRow[2] * stride
                val b = boxRow[3] * stride
                val x1 = ((cx - l) - padX) / scale
                val y1 = ((cy - t) - padY) / scale
                val x2 = ((cx + r) - padX) / scale
                val y2 = ((cy + b) - padY) / scale
                out += FaceDetection(
                    x1 = x1.coerceIn(0f, origWf),
                    y1 = y1.coerceIn(0f, origHf),
                    x2 = x2.coerceIn(0f, origWf),
                    y2 = y2.coerceIn(0f, origHf),
                    score = score,
                    kps = emptyKps
                )
            }
        }

        // 释放引用，允许 ONNX 输出 GC
        scoresRef.fill(null)
        boxesRef.fill(null)
    }

    private fun buildAnchorGrid(stride: Int, inputW: Int, inputH: Int): AnchorGrid {
        val fmW = inputW / stride
        val fmH = inputH / stride
        val n = fmW * fmH * 2
        val cx = FloatArray(n)
        val cy = FloatArray(n)
        var idx = 0
        for (y in 0 until fmH) {
            val cyVal = (y * stride).toFloat()
            for (x in 0 until fmW) {
                val cxVal = (x * stride).toFloat()
                repeat(2) {
                    cx[idx] = cxVal
                    cy[idx] = cyVal
                    idx++
                }
            }
        }
        return AnchorGrid(cx, cy)
    }

    /** 使用 Float 精度的 exp；显式比较替代 in 0f..1f，避免 ClosedFloatingPointRange 对象创建 */
    private fun toProbability(v: Float): Float {
        if (v >= 0f && v <= 1f) return v
        return 1f / (1f + exp(-v))
    }

    private fun iou(a: FaceDetection, b: FaceDetection): Float {
        val xx1 = max(a.x1, b.x1)
        val yy1 = max(a.y1, b.y1)
        val xx2 = min(a.x2, b.x2)
        val yy2 = min(a.y2, b.y2)
        val w = max(0f, xx2 - xx1)
        val h = max(0f, yy2 - yy1)
        val inter = w * h
        val union = a.area + b.area - inter
        if (union <= 0f) return 0f
        return inter / union
    }

    private fun nms(
        candidates: ArrayList<FaceDetection>,
        iouThresh: Float,
        topK: Int
    ): List<FaceDetection> {
        if (candidates.isEmpty()) return emptyList()
        // 用显式 Comparator 替代 sortByDescending { it.score }，避免 Float 装箱
        candidates.sortWith { a, b -> java.lang.Float.compare(b.score, a.score) }
        val n = candidates.size
        nmsRemoved.fill(false, 0, n.coerceAtMost(nmsRemoved.size))
        val kept = ArrayList<FaceDetection>(topK.coerceAtMost(n))
        for (i in 0 until n) {
            if (i < nmsRemoved.size && nmsRemoved[i]) continue
            val di = candidates[i]
            kept += di
            if (topK > 0 && kept.size >= topK) break
            for (j in i + 1 until n) {
                if (j < nmsRemoved.size && !nmsRemoved[j] && iou(di, candidates[j]) > iouThresh) {
                    nmsRemoved[j] = true
                }
            }
        }
        return kept
    }

    override fun close() {
        inputTensor.close()
        session.close()
    }
}
