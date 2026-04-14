package com.facenet.mobile.core

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF

object BitmapUtils {

    // 预分配：resizeBitmap 是单线程路径（realtimeLock 保护），可安全复用
    private val resizeMatrix  = Matrix()
    private val resizeSrcRect = RectF()
    private val resizeDstRect = RectF()
    private val alignValues = FloatArray(9)
    private val ataWorkspace = Array(4) { FloatArray(4) }
    private val atbWorkspace = FloatArray(4)
    private val augWorkspace = Array(4) { FloatArray(5) }
    private val paramsWorkspace = FloatArray(4)

    fun fillFaceSquareRect(bitmap: Bitmap, det: FaceDetection, expandRatio: Float, outRect: Rect) {
        val w = det.width
        val h = det.height
        val cx = (det.x1 + det.x2) * 0.5f
        val cy = (det.y1 + det.y2) * 0.5f
        val side = maxOf(w, h) * expandRatio
        val x1 = (cx - side * 0.5f).toInt().coerceIn(0, bitmap.width - 1)
        val y1 = (cy - side * 0.5f).toInt().coerceIn(0, bitmap.height - 1)
        val x2 = (cx + side * 0.5f).toInt().coerceIn(1, bitmap.width)
        val y2 = (cy + side * 0.5f).toInt().coerceIn(1, bitmap.height)
        outRect.set(x1, y1, x2.coerceAtLeast(x1 + 1), y2.coerceAtLeast(y1 + 1))
    }

    fun alignFaceFivePointsInto(
        src: Bitmap,
        kps: FloatArray,
        dst: Bitmap,
        canvas: Canvas,
        matrix: Matrix,
        reference: FloatArray
    ) {
        require(kps.size >= 10) { "Expected 5-point landmarks as 10 floats." }
        canvas.drawColor(Color.BLACK)

        val params = estimateSimilarityTransform(kps, reference)
        if (params == null) {
            resizeSrcRect.set(0f, 0f, src.width.toFloat(), src.height.toFloat())
            resizeDstRect.set(0f, 0f, dst.width.toFloat(), dst.height.toFloat())
            matrix.setRectToRect(resizeSrcRect, resizeDstRect, Matrix.ScaleToFit.FILL)
            canvas.drawBitmap(src, matrix, null)
            return
        }

        alignValues[0] = params[0]
        alignValues[1] = -params[1]
        alignValues[2] = params[2]
        alignValues[3] = params[1]
        alignValues[4] = params[0]
        alignValues[5] = params[3]
        alignValues[6] = 0f
        alignValues[7] = 0f
        alignValues[8] = 1f

        matrix.setValues(alignValues)
        canvas.drawBitmap(src, matrix, null)
    }

    private fun scaledReference(outputSize: Int): FloatArray {
        val ratio = outputSize / 112f
        val out = FloatArray(REFERENCE_ALIGNMENT_112.size)
        for (i in REFERENCE_ALIGNMENT_112.indices step 2) {
            out[i] = REFERENCE_ALIGNMENT_112[i] * ratio
            out[i + 1] = REFERENCE_ALIGNMENT_112[i + 1] * ratio
        }
        return out
    }

    /**
     * Solve similarity transform:
     *   u = a*x - b*y + tx
     *   v = b*x + a*y + ty
     * with least squares over 5 landmark pairs.
     */
    private fun estimateSimilarityTransform(src: FloatArray, dst: FloatArray): FloatArray? {
        val ata = ataWorkspace
        val atb = atbWorkspace
        for (r in 0 until 4) {
            atb[r] = 0f
            for (c in 0 until 4) {
                ata[r][c] = 0f
            }
        }

        for (i in 0 until 5) {
            val x = src[2 * i]
            val y = src[2 * i + 1]
            val u = dst[2 * i]
            val v = dst[2 * i + 1]

            accumulateNormalEquation(x, -y, 1f, 0f, u, ata, atb)
            accumulateNormalEquation(y, x, 0f, 1f, v, ata, atb)
        }

        return solve4x4(ata, atb)
    }

    private fun accumulateNormalEquation(
        r0: Float,
        r1: Float,
        r2: Float,
        r3: Float,
        value: Float,
        ata: Array<FloatArray>,
        atb: FloatArray
    ) {
        atb[0] += r0 * value
        atb[1] += r1 * value
        atb[2] += r2 * value
        atb[3] += r3 * value

        ata[0][0] += r0 * r0
        ata[0][1] += r0 * r1
        ata[0][2] += r0 * r2
        ata[0][3] += r0 * r3
        ata[1][0] += r1 * r0
        ata[1][1] += r1 * r1
        ata[1][2] += r1 * r2
        ata[1][3] += r1 * r3
        ata[2][0] += r2 * r0
        ata[2][1] += r2 * r1
        ata[2][2] += r2 * r2
        ata[2][3] += r2 * r3
        ata[3][0] += r3 * r0
        ata[3][1] += r3 * r1
        ata[3][2] += r3 * r2
        ata[3][3] += r3 * r3
    }

    private fun solve4x4(a: Array<FloatArray>, b: FloatArray): FloatArray? {
        val aug = augWorkspace
        for (r in 0 until 4) {
            for (c in 0 until 4) {
                aug[r][c] = a[r][c]
            }
            aug[r][4] = b[r]
        }

        for (col in 0 until 4) {
            var pivot = col
            var maxAbs = kotlin.math.abs(aug[col][col])
            for (row in col + 1 until 4) {
                val absValue = kotlin.math.abs(aug[row][col])
                if (absValue > maxAbs) {
                    maxAbs = absValue
                    pivot = row
                }
            }
            if (maxAbs < 1e-6f) return null

            if (pivot != col) {
                val tmp = aug[col]
                aug[col] = aug[pivot]
                aug[pivot] = tmp
            }

            val pivotValue = aug[col][col]
            for (j in col until 5) {
                aug[col][j] /= pivotValue
            }

            for (row in 0 until 4) {
                if (row == col) continue
                val factor = aug[row][col]
                if (factor == 0f) continue
                for (j in col until 5) {
                    aug[row][j] -= factor * aug[col][j]
                }
            }
        }

        for (i in 0 until 4) {
            paramsWorkspace[i] = aug[i][4]
        }
        return paramsWorkspace
    }

    private val REFERENCE_ALIGNMENT_112 = floatArrayOf(
        38.2946f, 51.6963f,
        73.5318f, 51.5014f,
        56.0252f, 71.7366f,
        41.5493f, 92.3655f,
        70.7299f, 92.2041f
    )
}
