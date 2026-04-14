package com.facenet.mobile.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.facenet.mobile.core.EmbeddingResult
import com.facenet.mobile.core.FaceAnalysisResult
import com.facenet.mobile.core.FacePipelineAndroid
import com.facenet.mobile.core.FaceRegistryStore
import com.facenet.mobile.core.MatchResult
import com.facenet.mobile.core.RealtimeFrameDetection
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

data class EmotionThresholds(
    val neutral: Float = 0.30f,
    val happy: Float = 0.30f,
    val sad: Float = 0.30f,
    val surprise: Float = 0.30f,
    val fear: Float = 0.30f,
    val disgust: Float = 0.30f,
    val angry: Float = 0.30f
) {
    fun thresholdFor(label: String): Float {
        return when (label) {
            'N' + "eutral" -> neutral
            'H' + "appy" -> happy
            'S' + "ad" -> sad
            'S' + "urprise" -> surprise
            'F' + "ear" -> fear
            'D' + "isgust" -> disgust
            'A' + "ngry" -> angry
            else -> 0.0f
        }
    }
}

data class AppSettings(
    val realtimeEnabled: Boolean = false,
    val identifyThreshold: Float = 0.45f,
    val detectionIntervalMs: Int = 280,
    val emotionThresholds: EmotionThresholds = EmotionThresholds()
)

data class AppUiState(
    val registerName: String = "",
    val registerProgress: Int = 0,
    val registerTarget: Int = 3,
    val status: String = "准备就绪",
    val users: List<String> = emptyList(),
    val verifyResult: String = "",
    val isBusy: Boolean = false,
    val settings: AppSettings = AppSettings()
)

class AppViewModel(app: Application) : AndroidViewModel(app) {
    private val _uiState = MutableStateFlow(AppUiState())
    val uiState: StateFlow<AppUiState> = _uiState.asStateFlow()

    private val pipeline = FacePipelineAndroid(app.applicationContext)
    private val registry = FaceRegistryStore(app.applicationContext)
    private var currentRegisterUser: String = ""
    private var currentShots = 0
    private var lastRealtimeResult: String = ""
    private val realtimeEmbedding = FloatArray(512)

    /** 实时识别并发锁：CAS 保证只有一个协程在处理，丢帧时立即 recycle */
    private val realtimeLock = AtomicBoolean(false)

    init {
        viewModelScope.launch(Dispatchers.IO) {
            registry.loadBin()
            withContext(Dispatchers.Main) { updateUsers() }
        }
    }

    fun setRegisterName(name: String) {
        _uiState.value = _uiState.value.copy(registerName = name)
    }

    fun setRegisterTarget(target: Int) {
        _uiState.value = _uiState.value.copy(registerTarget = target.coerceIn(1, 3))
    }

    fun captureForRegister(bitmap: Bitmap?) {
        if (bitmap == null) {
            _uiState.value = _uiState.value.copy(status = "拍照失败，请重试")
            return
        }
        val user = _uiState.value.registerName.trim()
        if (user.isEmpty()) {
            _uiState.value = _uiState.value.copy(status = "请先输入用户名称")
            return
        }
        if (currentRegisterUser != user) {
            currentRegisterUser = user
            currentShots = 0
            _uiState.value = _uiState.value.copy(registerProgress = 0)
        }
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isBusy = true, status = "正在提取人脸特征...")
            val emb = try {
                withContext(Dispatchers.Default) {
                    pipeline.extractLargestFaceEmbedding(bitmap, enforceQuality = true)
                }
            } finally {
                bitmap.recycle()
            }
            if (!emb.success) {
                val msg = if (emb.failReason.isNotBlank()) emb.failReason else "请正视镜头、确保光线充足"
                _uiState.value = _uiState.value.copy(isBusy = false, status = "质量未通过：$msg")
                return@launch
            }
            val ok = registry.register(user, emb.embedding)
            if (!ok) {
                _uiState.value = _uiState.value.copy(isBusy = false, status = "注册失败，请重试")
                return@launch
            }
            currentShots++
            _uiState.value = _uiState.value.copy(
                isBusy = false,
                registerProgress = currentShots,
                status = "采集成功 (${currentShots}/${_uiState.value.registerTarget}) 分数=${"%.3f".format(emb.detScore)}"
            )
            if (currentShots >= _uiState.value.registerTarget) {
                finishRegister()
            }
        }
    }

    fun finishRegister() {
        if (currentShots <= 0 || currentRegisterUser.isEmpty()) {
            _uiState.value = _uiState.value.copy(status = "至少采集 1 张后才能完成注册")
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            registry.saveBin()
            registry.saveJson()
        }
        updateUsers()
        _uiState.value = _uiState.value.copy(
            status = "注册完成：$currentRegisterUser，共 $currentShots 张",
            registerProgress = currentShots
        )
        currentShots = 0
        currentRegisterUser = ""
    }

    fun captureForVerify(bitmap: Bitmap?) {
        if (bitmap == null) {
            _uiState.value = _uiState.value.copy(verifyResult = "拍照失败", status = "拍照失败")
            return
        }
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isBusy = true, status = "正在识别...")
            val emb = try {
                withContext(Dispatchers.Default) {
                    pipeline.extractLargestFaceEmbedding(bitmap, enforceQuality = false)
                }
            } finally {
                bitmap.recycle()
            }
            if (!emb.success) {
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    verifyResult = "未检测到有效人脸",
                    status = "识别失败"
                )
                return@launch
            }
            val threshold = _uiState.value.settings.identifyThreshold
            val result = buildVerifyLine(emb, threshold)
            lastRealtimeResult = result
            _uiState.value = _uiState.value.copy(isBusy = false, verifyResult = result, status = "识别完成")
        }
    }

    fun realtimeIdentify(frame: RealtimeFrameDetection?) {
        if (frame == null) {
            if (lastRealtimeResult != "未找到人脸") {
                lastRealtimeResult = "未找到人脸"
                _uiState.value = _uiState.value.copy(verifyResult = lastRealtimeResult)
            }
            return
        }
        if (!realtimeLock.compareAndSet(false, true)) {
            return
        }
        try {
            val analysis = pipeline.analyzeRealtimeFace(frame.bitmap, frame.detection, realtimeEmbedding)
            val threshold = _uiState.value.settings.identifyThreshold
            val result = buildRealtimeLine(analysis, threshold) ?: return
            if (result != lastRealtimeResult) {
                lastRealtimeResult = result
                _uiState.value = _uiState.value.copy(verifyResult = result)
            }
        } finally {
            realtimeLock.set(false)
        }
    }

    fun deleteUser(userId: String) {
        val existed = registry.deleteUser(userId)
        if (!existed) return
        viewModelScope.launch(Dispatchers.IO) {
            registry.saveBin()
            registry.saveJson()
        }
        updateUsers()
        _uiState.value = _uiState.value.copy(status = "已删除：$userId")
    }

    fun updateSettings(settings: AppSettings) {
        _uiState.value = _uiState.value.copy(settings = settings)
    }

    private fun updateUsers() {
        _uiState.value = _uiState.value.copy(users = registry.listRegisteredUsers())
    }

    private fun formatEmotionSuffix(emb: EmbeddingResult): String {
        return formatEmotionSuffix(emb.emotionLabel, emb.emotionConfidence)
    }

    private fun formatEmotionSuffix(emotionLabel: String, emotionConfidence: Float): String {
        if (emotionLabel.isBlank()) return ""
        val thresholds = _uiState.value.settings.emotionThresholds
        val threshold = thresholds.thresholdFor(emotionLabel)
        if (emotionConfidence >= threshold) {
            return " | $emotionLabel ${"%.3f".format(emotionConfidence)}"
        }
        return " | 待定($emotionLabel ${"%.3f".format(emotionConfidence)} < ${"%.2f".format(threshold)})"
    }

    private fun buildVerifyLine(emb: EmbeddingResult, threshold: Float): String {
        val match: MatchResult = registry.identify(emb.embedding, threshold = threshold)
        val identifyPart = if (match.userId.isBlank()) {
            "未识别（${"%.3f".format(match.similarity)}）"
        } else {
            "✓ ${match.userId}（${"%.3f".format(match.similarity)}）"
        }
        return "$identifyPart${formatEmotionSuffix(emb)}"
    }

    private fun buildRealtimeLine(analysis: FaceAnalysisResult, threshold: Float): String? {
        if (!analysis.success) return null
        val match: MatchResult = registry.identify(realtimeEmbedding, threshold = threshold)
        val identifyPart = if (match.userId.isBlank()) {
            "未识别 ${"%.2f".format(match.similarity)}"
        } else {
            "✓ ${match.userId} ${"%.2f".format(match.similarity)}"
        }
        val emotionPart = formatEmotionSuffix(analysis.emotionLabel, analysis.emotionConfidence)
        return "$identifyPart$emotionPart"
    }

    override fun onCleared() {
        super.onCleared()
        pipeline.close()
    }
}
