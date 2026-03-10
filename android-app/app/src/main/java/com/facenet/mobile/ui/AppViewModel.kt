package com.facenet.mobile.ui

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.facenet.mobile.core.FacePipelineAndroid
import com.facenet.mobile.core.FaceRegistryStore
import com.facenet.mobile.core.MatchResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

data class AppSettings(
    val realtimeEnabled: Boolean = false,
    val identifyThreshold: Float = 0.45f,
    val detectionIntervalMs: Int = 280
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

    /** 实时识别并发锁：CAS 保证只有一个协程在处理，丢帧时立即 recycle */
    private val realtimeLock = AtomicBoolean(false)

    init {
        // loadBin 读取磁盘，移至 IO 线程避免阻塞主线程
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
            // try-finally 保证 bitmap 在任何情况下（包括异常/取消）都能被 recycle
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
                status = "采集成功 (${currentShots}/${_uiState.value.registerTarget})  分数=${"%.3f".format(emb.detScore)}"
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
            val m: MatchResult = registry.identify(emb.embedding, threshold = threshold)
            val result = if (m.userId.isBlank()) {
                "未识别（${"%.3f".format(m.similarity)}）"
            } else {
                "✓ ${m.userId}（${"%.3f".format(m.similarity)}）"
            }
            _uiState.value = _uiState.value.copy(isBusy = false, verifyResult = result, status = "识别完成")
        }
    }

    fun realtimeIdentify(bitmap: Bitmap) {
        // CAS 保证只有一个协程在运行，当前忙则丢帧并立即回收 bitmap
        if (!realtimeLock.compareAndSet(false, true)) {
            bitmap.recycle()
            return
        }
        viewModelScope.launch {
            try {
                val emb = withContext(Dispatchers.Default) {
                    pipeline.extractLargestFaceEmbedding(bitmap, enforceQuality = false)
                }
                bitmap.recycle()
                if (!emb.success) return@launch
                val threshold = _uiState.value.settings.identifyThreshold
                val m: MatchResult = registry.identify(emb.embedding, threshold = threshold)
                val result = if (m.userId.isBlank()) "" else "✓ ${m.userId} (${"%.2f".format(m.similarity)})"
                _uiState.value = _uiState.value.copy(verifyResult = result)
            } finally {
                realtimeLock.set(false)
            }
        }
    }

    fun deleteUser(userId: String) {
        val existed = registry.deleteUser(userId)
        if (!existed) return
        // 磁盘写入移至 IO 线程，避免阻塞主线程
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

    override fun onCleared() {
        super.onCleared()
        pipeline.close()
    }
}
