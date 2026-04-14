package com.facenet.mobile.core

@Suppress("ArrayInDataClass")  // kps 为共享空数组，不参与 equals/hashCode/copy
data class FaceDetection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val kps: FloatArray = FloatArray(10)
) {
    val width:  Float get() = (x2 - x1).coerceAtLeast(0f)
    val height: Float get() = (y2 - y1).coerceAtLeast(0f)
    val area:   Float get() = width * height
}

@Suppress("ArrayInDataClass")  // embedding 不参与 equals/hashCode
data class EmbeddingResult(
    val success: Boolean,
    val detScore: Float = 0f,
    val embedding: FloatArray = FloatArray(512),
    val failReason: String = "",
    val emotionLabel: String = "",
    val emotionConfidence: Float = 0f
)

data class MatchResult(
    val detected: Boolean,
    val userId: String,
    val similarity: Float
    // detScore 已移除：FaceRegistryStore.identify 始终返回 0f，为死字段
)

@Suppress("ArrayInDataClass")
data class EmotionResult(
    val label: String,
    val confidence: Float,
    val probabilities: FloatArray = FloatArray(0)
)

data class FaceAnalysisResult(
    val success: Boolean,
    val detScore: Float = 0f,
    val failReason: String = "",
    val emotionLabel: String = "",
    val emotionConfidence: Float = 0f
)

data class RealtimeFrameDetection(
    val bitmap: android.graphics.Bitmap,
    val detection: FaceDetection,
    val sourceWidth: Int = bitmap.width,
    val sourceHeight: Int = bitmap.height
)
