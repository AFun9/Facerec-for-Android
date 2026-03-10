package com.facenet.mobile.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.facenet.mobile.camera.CameraCaptureView

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppScreen(vm: AppViewModel = viewModel()) {
    val state by vm.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { hasPermission = it }

    LaunchedEffect(Unit) {
        if (!hasPermission) launcher.launch(Manifest.permission.CAMERA)
    }

    if (!hasPermission) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("需要相机权限才能使用", fontSize = 16.sp)
                Spacer(Modifier.height(12.dp))
                Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) {
                    Text("授予权限")
                }
            }
        }
        return
    }

    var captureAction by remember {
        mutableStateOf<(onBitmap: (Bitmap?) -> Unit) -> Unit>({ it(null) })
    }
    var showRegisterSheet by remember { mutableStateOf(false) }
    var showSettingsSheet by remember { mutableStateOf(false) }

    Box(modifier = Modifier.fillMaxSize()) {
        // 全屏摄像头 + 人脸框覆盖层
        CameraCaptureView(
            modifier = Modifier.fillMaxSize(),
            detectionIntervalMs = state.settings.detectionIntervalMs.toLong(),
            onFrameForIdentify = if (state.settings.realtimeEnabled) {
                { bitmap -> vm.realtimeIdentify(bitmap) }
            } else null,
            onReady = { captureAction = it }
        )

        // 实时识别结果浮层（顶部居中）
        if (state.verifyResult.isNotBlank()) {
            Text(
                text = state.verifyResult,
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 48.dp)
                    .background(Color.Black.copy(alpha = 0.65f), RoundedCornerShape(12.dp))
                    .padding(horizontal = 20.dp, vertical = 10.dp),
                color = Color.White,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }

        // 底部三个功能按钮
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 36.dp),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Button(
                onClick = { showRegisterSheet = true },
                modifier = Modifier.height(52.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0))
            ) { Text("注  册", fontSize = 16.sp) }

            Button(
                onClick = { captureAction { bitmap -> vm.captureForVerify(bitmap) } },
                modifier = Modifier.height(52.dp),
                enabled = !state.isBusy,
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
            ) { Text("识  别", fontSize = 16.sp) }

            Button(
                onClick = { showSettingsSheet = true },
                modifier = Modifier.height(52.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF6A1B9A))
            ) { Text("设  置", fontSize = 16.sp) }
        }
    }

    // 注册底部弹出面板
    if (showRegisterSheet) {
        ModalBottomSheet(
            onDismissRequest = { showRegisterSheet = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
        ) {
            RegisterSheet(
                state = state,
                vm = vm,
                captureAction = captureAction,
                onDismiss = { showRegisterSheet = false }
            )
        }
    }

    // 设置底部弹出面板
    if (showSettingsSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSettingsSheet = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
        ) {
            SettingsSheet(
                state = state,
                vm = vm,
                onDismiss = { showSettingsSheet = false }
            )
        }
    }
}

@Composable
private fun RegisterSheet(
    state: AppUiState,
    vm: AppViewModel,
    captureAction: (onBitmap: (Bitmap?) -> Unit) -> Unit,
    onDismiss: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .padding(bottom = 36.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Text("人脸注册", fontSize = 22.sp, fontWeight = FontWeight.Bold)

        OutlinedTextField(
            value = state.registerName,
            onValueChange = vm::setRegisterName,
            label = { Text("用户名称") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )

        // 采集张数选择
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(
                onClick = { vm.setRegisterTarget(1) },
                colors = if (state.registerTarget == 1)
                    ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0))
                else
                    ButtonDefaults.buttonColors(containerColor = Color(0xFF90A4AE))
            ) { Text("单张") }
            Button(
                onClick = { vm.setRegisterTarget(3) },
                colors = if (state.registerTarget == 3)
                    ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0))
                else
                    ButtonDefaults.buttonColors(containerColor = Color(0xFF90A4AE))
            ) { Text("推荐3张") }
        }

        Text(
            text = "采集进度：${state.registerProgress} / ${state.registerTarget}",
            fontSize = 14.sp,
            color = Color.Gray
        )

        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Button(
                onClick = { captureAction { bitmap -> vm.captureForRegister(bitmap) } },
                enabled = !state.isBusy,
                modifier = Modifier.weight(1f)
            ) { Text("拍摄采集") }

            Button(
                onClick = {
                    vm.finishRegister()
                    onDismiss()
                },
                enabled = !state.isBusy,
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
            ) { Text("完成注册") }
        }

        Text(
            text = "状态：${state.status}",
            fontSize = 13.sp,
            color = Color(0xFF616161)
        )

        if (state.users.isNotEmpty()) {
            Text("已注册用户：", fontWeight = FontWeight.Medium, fontSize = 14.sp)
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                state.users.forEach { user ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text("• $user", fontSize = 14.sp, modifier = Modifier.weight(1f))
                        Button(
                            onClick = { vm.deleteUser(user) },
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFC62828)),
                            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp, vertical = 4.dp),
                            modifier = Modifier.height(32.dp)
                        ) {
                            Text("删除", fontSize = 12.sp)
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(4.dp))
    }
}

@Composable
private fun SettingsSheet(
    state: AppUiState,
    vm: AppViewModel,
    onDismiss: () -> Unit
) {
    val s = state.settings
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .padding(bottom = 36.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {
        Text("设置", fontSize = 22.sp, fontWeight = FontWeight.Bold)

        // 实时识别开关
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("实时识别", fontWeight = FontWeight.Medium, fontSize = 15.sp)
                Text("自动对预览帧持续识别人脸", fontSize = 12.sp, color = Color.Gray)
            }
            Switch(
                checked = s.realtimeEnabled,
                onCheckedChange = { vm.updateSettings(s.copy(realtimeEnabled = it)) }
            )
        }

        // 识别阈值滑块
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("识别阈值", fontWeight = FontWeight.Medium, fontSize = 15.sp)
                Text("${"%.2f".format(s.identifyThreshold)}", fontSize = 15.sp, color = Color(0xFF1565C0))
            }
            Text("越高越严格，建议 0.40～0.65", fontSize = 12.sp, color = Color.Gray)
            Slider(
                value = s.identifyThreshold,
                onValueChange = { vm.updateSettings(s.copy(identifyThreshold = it)) },
                valueRange = 0.30f..0.80f,
                steps = 9
            )
        }

        // 帧处理间隔滑块
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("帧处理间隔", fontWeight = FontWeight.Medium, fontSize = 15.sp)
                Text("${s.detectionIntervalMs} ms", fontSize = 15.sp, color = Color(0xFF1565C0))
            }
            Text("间隔越小越实时，但性能消耗更高", fontSize = 12.sp, color = Color.Gray)
            Slider(
                value = s.detectionIntervalMs.toFloat(),
                onValueChange = { vm.updateSettings(s.copy(detectionIntervalMs = it.toInt())) },
                valueRange = 100f..2000f,
                steps = 18
            )
        }

        Button(
            onClick = onDismiss,
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF455A64))
        ) { Text("关  闭") }

        Spacer(Modifier.height(4.dp))
    }
}
