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
    private companion object {
        private const val ROTATION_90 = 90
        private const val ROTATION_180 = 180
        private const val ROTATION_270 = 270
    }

    /** 除法转乘法：1/inputStd 预先计算，hot loop 每像素省 1 次 fdiv */
    private val invStd = 1f / inputStd
    private val blackNorm = (0f - inputMean) * invStd

    private data class AnchorGrid(val cx: FloatArray, val cy: FloatArray)
    private data class FrameSize(val width: Int, val height: Int)

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
    private val inputMap: Map<String, OnnxTensor>

    /** anchor 网格，数组索引（0=stride8, 1=stride16, 2=stride32）替换 HashMap */
    private val anchorGridArr = Array(3) { AnchorGrid(FloatArray(0), FloatArray(0)) }

    /** ONNX 输出引用，按 stride 索引存储，替换每帧多次 HashMap 创建 */
    private val scoresRef = arrayOfNulls<Array<FloatArray>>(3)
    private val boxesRef  = arrayOfNulls<Array<FloatArray>>(3)
    private val kpsRef    = arrayOfNulls<Array<FloatArray>>(3)

    /** 候选框工作区：NMS 前只保留原始数值，避免为所有候选框创建对象 */
    private var candX1 = FloatArray(256)
    private var candY1 = FloatArray(256)
    private var candX2 = FloatArray(256)
    private var candY2 = FloatArray(256)
    private var candScore = FloatArray(256)
    private var candKps = FloatArray(256 * 10)
    private var nmsRemoved = BooleanArray(256)

    /** RGBA 采样缓存：把每帧重复的坐标映射和偏移计算搬到数组里 */
    private val sampleOffsetX = IntArray(inputWidth)
    private val sampleOffsetY = IntArray(inputHeight)
    private var cachedSampleWidth = -1
    private var cachedSampleHeight = -1
    private var cachedRotation = Int.MIN_VALUE
    private var cachedPadX = -1
    private var cachedPadY = -1
    private var cachedNewW = -1
    private var cachedNewH = -1
    private var cachedRowStride = -1
    private var cachedPixelStride = -1

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
        scoreThresh: Float,
        nmsIouThresh: Float,
        topK: Int
    ): List<FaceDetection> {
        preprocess(bitmap)
        val candidateCount = session.run(inputMap).use { result ->
            decodeOutputs(result, bitmap.width, bitmap.height, scoreThresh)
        }
        return nms(candidateCount, nmsIouThresh, topK)
    }

    fun detectRgba(
        rgbaBuffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int,
        rotationDegrees: Int,
        scoreThresh: Float,
        nmsIouThresh: Float,
        topK: Int
    ): List<FaceDetection> {
        preprocessRgba(rgbaBuffer, width, height, rowStride, pixelStride, rotationDegrees)
        val outputSize = orientedSize(width, height, rotationDegrees)
        val candidateCount = session.run(inputMap).use { result ->
            decodeOutputs(result, outputSize.width, outputSize.height, scoreThresh)
        }
        return nms(candidateCount, nmsIouThresh, topK)
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

    private fun preprocessRgba(
        rgbaBuffer: ByteBuffer,
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int,
        rotationDegrees: Int
    ) {
        val sourceSize = orientedSize(width, height, rotationDegrees)
        val srcW = sourceSize.width
        val srcH = sourceSize.height
        val scale = min(inputWidth.toFloat() / srcW, inputHeight.toFloat() / srcH)
        val newW = max(1, (srcW * scale).toInt())
        val newH = max(1, (srcH * scale).toInt())
        val padX = (inputWidth - newW) / 2
        val padY = (inputHeight - newH) / 2
        val plane = inputWidth * inputHeight

        prepareRgbaSampling(
            width = width,
            height = height,
            rowStride = rowStride,
            pixelStride = pixelStride,
            rotationDegrees = rotationDegrees,
            srcW = srcW,
            srcH = srcH,
            scale = scale,
            padX = padX,
            padY = padY,
            newW = newW,
            newH = newH
        )
        val buffer = rgbaBuffer.duplicate()
        fillInputBufferWithBlack(plane)
        val contentBottom = padY + newH
        val contentRight = padX + newW
        for (y in padY until contentBottom) {
            val yOffset = sampleOffsetY[y]
            var idx = y * inputWidth + padX
            for (x in padX until contentRight) {
                val base = yOffset + sampleOffsetX[x]
                val r = (buffer.get(base).toInt() and 0xFF).toFloat()
                val g = (buffer.get(base + 1).toInt() and 0xFF).toFloat()
                val b = (buffer.get(base + 2).toInt() and 0xFF).toFloat()
                inputBuffer.put(idx, (r - inputMean) * invStd)
                inputBuffer.put(plane + idx, (g - inputMean) * invStd)
                inputBuffer.put(2 * plane + idx, (b - inputMean) * invStd)
                idx++
            }
        }

        preprocessMeta[0] = scale
        preprocessMeta[1] = padX.toFloat()
        preprocessMeta[2] = padY.toFloat()
    }

    private fun fillInputBufferWithBlack(plane: Int) {
        for (i in 0 until plane) {
            inputBuffer.put(i, blackNorm)
            inputBuffer.put(plane + i, blackNorm)
            inputBuffer.put(2 * plane + i, blackNorm)
        }
    }

    private fun prepareRgbaSampling(
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int,
        rotationDegrees: Int,
        srcW: Int,
        srcH: Int,
        scale: Float,
        padX: Int,
        padY: Int,
        newW: Int,
        newH: Int
    ) {
        if (isSamplingCacheValid(width, height, rowStride, pixelStride, rotationDegrees, padX, padY, newW, newH)) {
            return
        }

        val invScale = 1f / scale
        val contentRight = padX + newW
        val contentBottom = padY + newH

        for (x in 0 until inputWidth) {
            sampleOffsetX[x] = 0
        }
        for (y in 0 until inputHeight) {
            sampleOffsetY[y] = 0
        }

        when (rotationDegrees) {
            ROTATION_90 -> {
                for (x in padX until contentRight) {
                    val srcX = (((x - padX) + 0.5f) * invScale).toInt().coerceIn(0, srcW - 1)
                    sampleOffsetX[x] = (height - 1 - srcX) * rowStride
                }
                for (y in padY until contentBottom) {
                    val srcY = (((y - padY) + 0.5f) * invScale).toInt().coerceIn(0, srcH - 1)
                    sampleOffsetY[y] = srcY * pixelStride
                }
            }
            ROTATION_180 -> {
                for (x in padX until contentRight) {
                    val srcX = (((x - padX) + 0.5f) * invScale).toInt().coerceIn(0, srcW - 1)
                    sampleOffsetX[x] = (width - 1 - srcX) * pixelStride
                }
                for (y in padY until contentBottom) {
                    val srcY = (((y - padY) + 0.5f) * invScale).toInt().coerceIn(0, srcH - 1)
                    sampleOffsetY[y] = (height - 1 - srcY) * rowStride
                }
            }
            ROTATION_270 -> {
                for (x in padX until contentRight) {
                    val srcX = (((x - padX) + 0.5f) * invScale).toInt().coerceIn(0, srcW - 1)
                    sampleOffsetX[x] = srcX * rowStride
                }
                for (y in padY until contentBottom) {
                    val srcY = (((y - padY) + 0.5f) * invScale).toInt().coerceIn(0, srcH - 1)
                    sampleOffsetY[y] = (width - 1 - srcY) * pixelStride
                }
            }
            else -> {
                for (x in padX until contentRight) {
                    val srcX = (((x - padX) + 0.5f) * invScale).toInt().coerceIn(0, srcW - 1)
                    sampleOffsetX[x] = srcX * pixelStride
                }
                for (y in padY until contentBottom) {
                    val srcY = (((y - padY) + 0.5f) * invScale).toInt().coerceIn(0, srcH - 1)
                    sampleOffsetY[y] = srcY * rowStride
                }
            }
        }

        updateSamplingCache(width, height, rowStride, pixelStride, rotationDegrees, padX, padY, newW, newH)
    }

    private fun orientedSize(width: Int, height: Int, rotationDegrees: Int): FrameSize {
        return when (rotationDegrees) {
            ROTATION_90, ROTATION_270 -> FrameSize(width = height, height = width)
            else -> FrameSize(width = width, height = height)
        }
    }

    private fun isSamplingCacheValid(
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int,
        rotationDegrees: Int,
        padX: Int,
        padY: Int,
        newW: Int,
        newH: Int
    ): Boolean {
        return cachedSampleWidth == width &&
            cachedSampleHeight == height &&
            cachedRotation == rotationDegrees &&
            cachedPadX == padX &&
            cachedPadY == padY &&
            cachedNewW == newW &&
            cachedNewH == newH &&
            cachedRowStride == rowStride &&
            cachedPixelStride == pixelStride
    }

    private fun updateSamplingCache(
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int,
        rotationDegrees: Int,
        padX: Int,
        padY: Int,
        newW: Int,
        newH: Int
    ) {
        cachedSampleWidth = width
        cachedSampleHeight = height
        cachedRotation = rotationDegrees
        cachedPadX = padX
        cachedPadY = padY
        cachedNewW = newW
        cachedNewH = newH
        cachedRowStride = rowStride
        cachedPixelStride = pixelStride
    }

    private fun decodeOutputs(
        result: OrtSession.Result,
        origW: Int,
        origH: Int,
        scoreThresh: Float
    ): Int {
        // 提取 scores / boxes / kps，stride 通过每层输出行数区分
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
                10 -> kpsRef[si]   = data
            }
        }

        val scale  = preprocessMeta[0]
        val padX   = preprocessMeta[1]
        val padY   = preprocessMeta[2]
        val origWf = (origW - 1).toFloat()
        val origHf = (origH - 1).toFloat()
        var candidateCount = 0

        for (si in 0..2) {
            val scores = scoresRef[si] ?: continue
            val boxes  = boxesRef[si]  ?: continue
            val kps    = kpsRef[si]    ?: continue
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
                val kpsRow = kps[i]
                ensureCandidateCapacity(candidateCount + 1)
                val base = candidateCount * 10
                for (p in 0 until 5) {
                    val dx = kpsRow[2 * p] * stride
                    val dy = kpsRow[2 * p + 1] * stride
                    val kx = ((cx + dx) - padX) / scale
                    val ky = ((cy + dy) - padY) / scale
                    candKps[base + 2 * p] = kx.coerceIn(0f, origWf)
                    candKps[base + 2 * p + 1] = ky.coerceIn(0f, origHf)
                }

                candX1[candidateCount] = x1.coerceIn(0f, origWf)
                candY1[candidateCount] = y1.coerceIn(0f, origHf)
                candX2[candidateCount] = x2.coerceIn(0f, origWf)
                candY2[candidateCount] = y2.coerceIn(0f, origHf)
                candScore[candidateCount] = score
                candidateCount++
            }
        }

        // 释放引用，允许 ONNX 输出 GC
        scoresRef.fill(null)
        boxesRef.fill(null)
        kpsRef.fill(null)
        return candidateCount
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

    private fun ensureCandidateCapacity(required: Int) {
        if (candX1.size >= required) return
        val newSize = max(required, candX1.size * 2)
        candX1 = candX1.copyOf(newSize)
        candY1 = candY1.copyOf(newSize)
        candX2 = candX2.copyOf(newSize)
        candY2 = candY2.copyOf(newSize)
        candScore = candScore.copyOf(newSize)
        candKps = candKps.copyOf(newSize * 10)
        nmsRemoved = nmsRemoved.copyOf(newSize)
    }

    private fun iou(indexA: Int, indexB: Int): Float {
        val xx1 = max(candX1[indexA], candX1[indexB])
        val yy1 = max(candY1[indexA], candY1[indexB])
        val xx2 = min(candX2[indexA], candX2[indexB])
        val yy2 = min(candY2[indexA], candY2[indexB])
        val w = max(0f, xx2 - xx1)
        val h = max(0f, yy2 - yy1)
        val inter = w * h
        val areaA = (candX2[indexA] - candX1[indexA]).coerceAtLeast(0f) * (candY2[indexA] - candY1[indexA]).coerceAtLeast(0f)
        val areaB = (candX2[indexB] - candX1[indexB]).coerceAtLeast(0f) * (candY2[indexB] - candY1[indexB]).coerceAtLeast(0f)
        val union = areaA + areaB - inter
        if (union <= 0f) return 0f
        return inter / union
    }

    private fun nms(
        candidateCount: Int,
        iouThresh: Float,
        topK: Int
    ): List<FaceDetection> {
        if (candidateCount <= 0) return emptyList()
        nmsRemoved.fill(false, 0, candidateCount)
        val keepLimit = if (topK > 0) topK.coerceAtMost(candidateCount) else candidateCount
        val kept = ArrayList<FaceDetection>(keepLimit)
        repeat(keepLimit) {
            var bestIndex = -1
            var bestScore = Float.NEGATIVE_INFINITY
            for (i in 0 until candidateCount) {
                if (!nmsRemoved[i] && candScore[i] > bestScore) {
                    bestScore = candScore[i]
                    bestIndex = i
                }
            }
            if (bestIndex < 0) return kept

            val kps = FloatArray(10)
            System.arraycopy(candKps, bestIndex * 10, kps, 0, 10)
            kept += FaceDetection(
                x1 = candX1[bestIndex],
                y1 = candY1[bestIndex],
                x2 = candX2[bestIndex],
                y2 = candY2[bestIndex],
                score = candScore[bestIndex],
                kps = kps
            )
            nmsRemoved[bestIndex] = true
            for (j in 0 until candidateCount) {
                if (!nmsRemoved[j] && iou(bestIndex, j) > iouThresh) {
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
