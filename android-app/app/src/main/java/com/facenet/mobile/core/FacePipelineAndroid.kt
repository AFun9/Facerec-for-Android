package com.facenet.mobile.core

import android.content.Context
import android.graphics.Bitmap

class FacePipelineAndroid(
    context: Context,
    detModelAsset: String = "facedet.onnx",
    facenetModelAsset: String = "facenet.onnx"
) : AutoCloseable {

    private val detector = FaceDetectorAndroid(context, detModelAsset)
    private val embedder = FaceNetEmbedderAndroid(context, facenetModelAsset)
    private val qualityGate = QualityGate()

    fun extractLargestFaceEmbedding(bitmap: Bitmap, enforceQuality: Boolean = true): EmbeddingResult {
        val dets = detector.detect(bitmap, scoreThresh = 0.45f, nmsIouThresh = 0.4f, topK = 1)
        if (dets.isEmpty()) return EmbeddingResult(false, failReason = "未检测到人脸")
        val best = dets[0]
        if (enforceQuality) {
            val check = qualityGate.check(bitmap, best)
            if (!check.pass) {
                return EmbeddingResult(false, detScore = best.score, failReason = check.reason)
            }
        }
        val crop = BitmapUtils.cropFaceSquare(bitmap, best, expandRatio = 1.2f)
        val embedding = embedder.embed(crop)
        crop.recycle()
        return EmbeddingResult(success = true, detScore = best.score, embedding = embedding)
    }

    override fun close() {
        detector.close()
        embedder.close()
    }
}
