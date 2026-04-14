package com.facenet.mobile.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Collections
import kotlin.math.exp

class EmotionClassifierAndroid(
    context: Context,
    private val modelAssetName: String = "affecnet7.onnx"
) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val outputName: String
    private val inputShape = longArrayOf(1, 3, 112, 112)
    private val inputBuffer: FloatBuffer =
        ByteBuffer.allocateDirect(1 * 3 * 112 * 112 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val inputTensor: OnnxTensor
    private val pixelBuf = IntArray(112 * 112)
    private val inputMap: Map<String, OnnxTensor>
    private val outputSet: Set<String>
    private val alignedBitmap = Bitmap.createBitmap(112, 112, Bitmap.Config.ARGB_8888)
    private val alignedCanvas = Canvas(alignedBitmap)
    private val alignMatrix = Matrix()
    private val alignedReference = floatArrayOf(
        38.2946f, 51.6963f,
        73.5318f, 51.5014f,
        56.0252f, 71.7366f,
        41.5493f, 92.3655f,
        70.7299f, 92.2041f
    )
    private val inv255 = 1f / 255f
    private val invStd0 = 1f / IMAGENET_STD[0]
    private val invStd1 = 1f / IMAGENET_STD[1]
    private val invStd2 = 1f / IMAGENET_STD[2]
    private val logitsBuf = FloatArray(EMOTION_LABELS.size)

    init {
        val modelFile = materializeAsset(context, modelAssetName)
        session = OrtSession.SessionOptions().use { opts ->
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            opts.setIntraOpNumThreads(2)
            env.createSession(modelFile.absolutePath, opts)
        }
        inputName = session.inputNames.first()
        outputName = session.outputNames.first()
        inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)
        inputMap = Collections.singletonMap(inputName, inputTensor)
        outputSet = setOf(outputName)
    }

    fun predict(image: Bitmap, kps: FloatArray): EmotionResult {
        BitmapUtils.alignFaceFivePointsInto(
            src = image,
            kps = kps,
            dst = alignedBitmap,
            canvas = alignedCanvas,
            matrix = alignMatrix,
            reference = alignedReference
        )
        return predictAlignedFace(alignedBitmap)
    }

    fun predictAlignedFace(alignedFace: Bitmap): EmotionResult {
        preprocess(alignedFace, inputBuffer)
        session.run(inputMap, outputSet).use { result ->
            val tensor = result[0] as OnnxTensor
            val fb = tensor.floatBuffer
            fb.rewind()
            fb.get(logitsBuf)
            return postprocess(logitsBuf)
        }
    }

    private fun preprocess(bitmap: Bitmap, dst: FloatBuffer) {
        bitmap.getPixels(pixelBuf, 0, 112, 0, 0, 112, 112)
        val plane = 112 * 112
        var idx = 0
        for (y in 0 until 112) {
            for (x in 0 until 112) {
                val argb = pixelBuf[idx]
                val r = ((argb shr 16) and 0xFF) * inv255
                val g = ((argb shr 8) and 0xFF) * inv255
                val b = (argb and 0xFF) * inv255
                dst.put(idx,             (r - IMAGENET_MEAN[0]) * invStd0)
                dst.put(plane + idx,     (g - IMAGENET_MEAN[1]) * invStd1)
                dst.put(2 * plane + idx, (b - IMAGENET_MEAN[2]) * invStd2)
                idx++
            }
        }
    }

    private fun postprocess(logits: FloatArray): EmotionResult {
        val maxLogit = logits.maxOrNull() ?: 0f
        var sum = 0f
        var bestIndex = 0
        var bestUnnormalized = 0f
        for (i in logits.indices) {
            val v = exp((logits[i] - maxLogit).toDouble()).toFloat()
            sum += v
            if (i == 0 || v > bestUnnormalized) {
                bestUnnormalized = v
                bestIndex = i
            }
        }
        val bestScore = if (sum > 0f) bestUnnormalized / sum else 0f
        return EmotionResult(
            label = EMOTION_LABELS[bestIndex],
            confidence = bestScore
        )
    }

    override fun close() {
        inputTensor.close()
        session.close()
        alignedBitmap.recycle()
    }

    companion object {
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        val EMOTION_LABELS = arrayOf(
            "Neutral",
            "Happy",
            "Sad",
            "Surprise",
            "Fear",
            "Disgust",
            "Angry"
        )
    }
}
