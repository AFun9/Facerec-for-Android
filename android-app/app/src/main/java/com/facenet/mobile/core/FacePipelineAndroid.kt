package com.facenet.mobile.core

import android.content.Context
import android.graphics.Bitmap

class FacePipelineAndroid(
    context: Context,
    detModelAsset: String = "facedet.onnx",
    facenetModelAsset: String = "facenet.onnx",
    emotionModelAsset: String = "affecnet7.onnx"
) : AutoCloseable {

    private val detector = FaceDetectorAndroid(context, detModelAsset)
    private val embedder = FaceNetEmbedderAndroid(context, facenetModelAsset)
    private val emotion = EmotionClassifierAndroid(context, emotionModelAsset)
    private val qualityGate = QualityGate()
    private val inferenceLock = Any()

    fun extractLargestFaceEmbedding(bitmap: Bitmap, enforceQuality: Boolean = true): EmbeddingResult {
        synchronized(inferenceLock) {
            val dets = detector.detect(bitmap, scoreThresh = 0.45f, nmsIouThresh = 0.4f, topK = 1)
            if (dets.isEmpty()) return EmbeddingResult(false, failReason = "未检测到人脸")
            return extractFaceEmbeddingLocked(bitmap, dets[0], enforceQuality)
        }
    }

    fun extractFaceEmbedding(
        bitmap: Bitmap,
        detection: FaceDetection,
        enforceQuality: Boolean = true
    ): EmbeddingResult = synchronized(inferenceLock) {
        extractFaceEmbeddingLocked(bitmap, detection, enforceQuality)
    }

    fun analyzeRealtimeFace(
        bitmap: Bitmap,
        detection: FaceDetection,
        outEmbedding: FloatArray
    ): FaceAnalysisResult = synchronized(inferenceLock) {
        if (outEmbedding.size != 512) {
            return FaceAnalysisResult(success = false, failReason = "embedding buffer size mismatch")
        }
        embedder.fillFaceRegionEmbedding(bitmap, detection, outEmbedding, expandRatio = 1.2f)
        val emotionResult = if (detection.kps.any { it != 0f }) {
            emotion.predict(bitmap, detection.kps)
        } else {
            EmotionResult(label = "", confidence = 0f)
        }
        FaceAnalysisResult(
            success = true,
            detScore = detection.score,
            emotionLabel = emotionResult.label,
            emotionConfidence = emotionResult.confidence
        )
    }

    private fun extractFaceEmbeddingLocked(
        bitmap: Bitmap,
        detection: FaceDetection,
        enforceQuality: Boolean
    ): EmbeddingResult {
        if (enforceQuality) {
            val check = qualityGate.check(bitmap, detection)
            if (!check.pass) {
                return EmbeddingResult(false, detScore = detection.score, failReason = check.reason)
            }
        }
        val embedding = embedder.embedFaceRegion(bitmap, detection, expandRatio = 1.2f)
        val emotionResult = if (detection.kps.any { it != 0f }) {
            emotion.predict(bitmap, detection.kps)
        } else {
            EmotionResult(label = "", confidence = 0f)
        }
        return EmbeddingResult(
            success = true,
            detScore = detection.score,
            embedding = embedding,
            emotionLabel = emotionResult.label,
            emotionConfidence = emotionResult.confidence
        )
    }

    override fun close() {
        detector.close()
        embedder.close()
        emotion.close()
        qualityGate.close()
    }
}
