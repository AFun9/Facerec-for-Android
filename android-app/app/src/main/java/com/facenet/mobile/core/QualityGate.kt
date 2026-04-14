package com.facenet.mobile.core

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.graphics.RectF

class QualityGate(
    private val minDetScore: Float = 0.45f,
    private val minFaceAreaRatio: Float = 0.015f,
    private val minSharpness: Float = 3.0f
) : AutoCloseable {
    /** 清晰度计算的像素缓冲，避免每次 check 分配 IntArray(25600) */
    private val sharpnessBuf = IntArray(160 * 160)
    private val normalizedBitmap = Bitmap.createBitmap(160, 160, Bitmap.Config.ARGB_8888)
    private val normalizedCanvas = Canvas(normalizedBitmap)
    private val cropSrcRect = Rect()
    private val normalizedDstRect = RectF(0f, 0f, 160f, 160f)

    data class CheckResult(
        val pass: Boolean,
        val reason: String,
        val ratio: Float,
        val sharpness: Float
    )

    fun check(bitmap: Bitmap, det: FaceDetection): CheckResult {
        if (det.score < minDetScore) {
            return CheckResult(false, "检测分数过低", 0f, 0f)
        }
        val ratio = det.area / (bitmap.width.toFloat() * bitmap.height.toFloat())
        if (ratio < minFaceAreaRatio) {
            return CheckResult(false, "人脸过小", ratio, 0f)
        }

        // 统一到固定分辨率再估计清晰度，减少不同拍照分辨率带来的偏差
        BitmapUtils.fillFaceSquareRect(bitmap, det, 1.0f, cropSrcRect)
        normalizedCanvas.drawBitmap(bitmap, cropSrcRect, normalizedDstRect, null)
        val sharp = sharpness(normalizedBitmap)

        if (sharp < minSharpness) {
            // 高置信度且人脸占比足够时，放宽清晰度门限，避免误拒绝
            if (det.score >= 0.75f && ratio >= 0.02f) {
                return CheckResult(true, "清晰度偏低但已放行", ratio, sharp)
            }
            return CheckResult(false, "清晰度不足", ratio, sharp)
        }
        return CheckResult(true, "ok", ratio, sharp)
    }

    // 基础清晰度估计：灰度一阶梯度均值
    private fun sharpness(bitmap: Bitmap): Float {
        val w = bitmap.width
        val h = bitmap.height
        if (w < 2 || h < 2) return 0f
        val pixels = sharpnessBuf
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        var sum = 0f
        var cnt = 0
        for (y in 0 until h - 1) {
            val row = y * w
            for (x in 0 until w - 1) {
                val p = pixels[row + x]
                val pX = pixels[row + x + 1]
                val pY = pixels[row + x + w]
                val g = gray(p)
                val gx = kotlin.math.abs(gray(pX) - g)
                val gy = kotlin.math.abs(gray(pY) - g)
                sum += gx + gy
                cnt++
            }
        }
        return if (cnt == 0) 0f else sum / cnt
    }

    private fun gray(argb: Int): Float {
        val r = (argb shr 16) and 0xFF
        val g = (argb shr 8) and 0xFF
        val b = argb and 0xFF
        return 0.299f * r + 0.587f * g + 0.114f * b
    }

    override fun close() {
        normalizedBitmap.recycle()
    }
}
