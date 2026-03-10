package com.facenet.mobile.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.exifinterface.media.ExifInterface
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.facenet.mobile.core.FaceDetection
import com.facenet.mobile.core.FaceDetectorAndroid
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@Composable
fun CameraCaptureView(
    modifier: Modifier = Modifier,
    detectionIntervalMs: Long = 280L,
    onFrameForIdentify: ((Bitmap) -> Unit)? = null,
    onReady: (capture: (onBitmap: (Bitmap?) -> Unit) -> Unit) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val previewViewState = remember { mutableStateOf<PreviewView?>(null) }
    val imageCaptureState = remember { mutableStateOf<ImageCapture?>(null) }
    val faceBoxesState = remember { mutableStateOf<List<FaceDetection>>(emptyList()) }
    // 用独立 Int state 替代 Pair，消除每帧 Pair 对象分配
    val srcWidthState  = remember { mutableStateOf(0) }
    val srcHeightState = remember { mutableStateOf(0) }
    val detector = remember(context) { FaceDetectorAndroid(context) }
    val scope = rememberCoroutineScope()

    val currentInterval by rememberUpdatedState(detectionIntervalMs)
    val currentOnFrameForIdentify by rememberUpdatedState(onFrameForIdentify)

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            detector.close()
        }
    }

    LaunchedEffect(previewViewState.value) {
        val previewView = previewViewState.value ?: return@LaunchedEffect
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val cameraProvider = cameraProviderFuture.get()
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }
        val imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
        imageCaptureState.value = imageCapture
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_FRONT_CAMERA,
            preview,
            imageCapture
        )
    }

    // remember 包裹：lambda 捕获的都是 remember 出的稳定对象，recomposition 不重建
    val captureFunc: (onBitmap: (Bitmap?) -> Unit) -> Unit = remember(imageCaptureState, cameraExecutor, scope) {
        { onBitmap ->
            val imageCapture = imageCaptureState.value
            if (imageCapture == null) {
                onBitmap(null)
            } else {
                captureToBitmap(context, imageCapture, cameraExecutor) { bitmap ->
                    scope.launch { onBitmap(bitmap) }
                }
            }
        }
    }

    LaunchedEffect(imageCaptureState.value) {
        onReady(captureFunc)
    }

    // 预览帧检测：定时从 PreviewView 抓一帧做人脸检测，若有实时识别回调则同步触发
    LaunchedEffect(previewViewState.value) {
        val previewView = previewViewState.value ?: return@LaunchedEffect
        while (isActive) {
            val frame = previewView.bitmap
            if (frame != null && frame.width > 0 && frame.height > 0) {
                val boxes = withContext(Dispatchers.Default) {
                    detector.detect(frame, scoreThresh = 0.45f, nmsIouThresh = 0.4f, topK = 5)
                }
                srcWidthState.value  = frame.width
                srcHeightState.value = frame.height
                faceBoxesState.value = boxes
                val identifyCb = currentOnFrameForIdentify
                if (identifyCb != null && boxes.isNotEmpty()) {
                    // 所有权转移给回调，回调方负责 recycle
                    identifyCb(frame)
                } else {
                    frame.recycle()
                }
            } else {
                frame?.recycle()
                faceBoxesState.value = emptyList()
            }
            delay(currentInterval)
        }
    }

    Box(modifier = modifier) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).also {
                    it.scaleType = PreviewView.ScaleType.FILL_CENTER
                    previewViewState.value = it
                }
            }
        )
        FaceBoxOverlay(
            boxes = faceBoxesState.value,
            srcWidth = srcWidthState.value,
            srcHeight = srcHeightState.value
        )
    }
}

private fun captureToBitmap(
    context: Context,
    imageCapture: ImageCapture,
    executor: ExecutorService,
    onResult: (Bitmap?) -> Unit
) {
    val tmp = File.createTempFile("capture_", ".jpg", context.cacheDir)
    val output = ImageCapture.OutputFileOptions.Builder(tmp).build()
    imageCapture.takePicture(
        output,
        executor,
        object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                val bitmap = BitmapFactory.decodeFile(tmp.absolutePath)
                val rotated = bitmap?.let { fixOrientation(it, tmp.absolutePath) }
                tmp.delete()
                onResult(rotated)
            }

            override fun onError(exception: ImageCaptureException) {
                tmp.delete()
                onResult(null)
            }
        }
    )
}

private fun fixOrientation(src: Bitmap, filePath: String): Bitmap {
    val exif = ExifInterface(filePath)
    val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
    val degree = when (orientation) {
        ExifInterface.ORIENTATION_ROTATE_90 -> 90f
        ExifInterface.ORIENTATION_ROTATE_180 -> 180f
        ExifInterface.ORIENTATION_ROTATE_270 -> 270f
        else -> 0f
    }
    if (degree == 0f) return src
    val matrix = Matrix().apply { postRotate(degree) }
    return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
}

// 文件级常量：Color 和 Stroke 只计算一次，避免在绘制循环内重复创建
private val FaceBoxColor  = Color(0xFF00E676)
private val FaceBoxStroke = Stroke(width = 4f)

@Composable
private fun FaceBoxOverlay(
    boxes: List<FaceDetection>,
    srcWidth: Int,
    srcHeight: Int
) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        if (srcWidth <= 0 || srcHeight <= 0) return@Canvas
        // 与 PreviewView.FILL_CENTER 对齐的坐标映射
        val scale = maxOf(size.width / srcWidth.toFloat(), size.height / srcHeight.toFloat())
        val scaledW = srcWidth * scale
        val scaledH = srcHeight * scale
        val dx = (scaledW - size.width) * 0.5f
        val dy = (scaledH - size.height) * 0.5f

        boxes.forEach { b ->
            val x1 = b.x1 * scale - dx
            val y1 = b.y1 * scale - dy
            val x2 = b.x2 * scale - dx
            val y2 = b.y2 * scale - dy
            val w = (x2 - x1).coerceAtLeast(1f)
            val h = (y2 - y1).coerceAtLeast(1f)

            drawRect(
                color = FaceBoxColor,
                topLeft = Offset(x1, y1),
                size = Size(w, h),
                style = FaceBoxStroke
            )
            drawLine(
                color = FaceBoxColor,
                start = Offset(x1, y1),
                end = Offset(x1 + 26f, y1),
                strokeWidth = 6f,
                cap = StrokeCap.Round
            )
            drawLine(
                color = FaceBoxColor,
                start = Offset(x1, y1),
                end = Offset(x1, y1 + 26f),
                strokeWidth = 6f,
                cap = StrokeCap.Round
            )
        }
    }
}
