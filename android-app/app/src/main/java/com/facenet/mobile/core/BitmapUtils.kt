package com.facenet.mobile.core

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.RectF

object BitmapUtils {

    // 预分配：resizeBitmap 是单线程路径（realtimeLock 保护），可安全复用
    private val resizeMatrix  = Matrix()
    private val resizeSrcRect = RectF()
    private val resizeDstRect = RectF()

    fun cropFaceSquare(bitmap: Bitmap, det: FaceDetection, expandRatio: Float = 1.2f): Bitmap {
        val w = det.width
        val h = det.height
        val cx = (det.x1 + det.x2) * 0.5f
        val cy = (det.y1 + det.y2) * 0.5f
        val side = maxOf(w, h) * expandRatio
        val x1 = (cx - side * 0.5f).toInt().coerceIn(0, bitmap.width - 1)
        val y1 = (cy - side * 0.5f).toInt().coerceIn(0, bitmap.height - 1)
        val x2 = (cx + side * 0.5f).toInt().coerceIn(1, bitmap.width)
        val y2 = (cy + side * 0.5f).toInt().coerceIn(1, bitmap.height)
        val cw = (x2 - x1).coerceAtLeast(1)
        val ch = (y2 - y1).coerceAtLeast(1)
        return Bitmap.createBitmap(bitmap, x1, y1, cw, ch)
    }

    fun resizeBitmap(src: Bitmap, width: Int, height: Int): Bitmap {
        val out = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        resizeSrcRect.set(0f, 0f, src.width.toFloat(), src.height.toFloat())
        resizeDstRect.set(0f, 0f, width.toFloat(), height.toFloat())
        resizeMatrix.setRectToRect(resizeSrcRect, resizeDstRect, Matrix.ScaleToFit.FILL)
        canvas.drawBitmap(src, resizeMatrix, null)
        return out
    }
}
