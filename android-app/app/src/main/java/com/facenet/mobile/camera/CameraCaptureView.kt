package com.facenet.mobile.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color as AndroidColor
import android.graphics.Matrix
import android.graphics.Rect
import android.os.SystemClock
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.facenet.mobile.core.FaceDetection
import com.facenet.mobile.core.FaceDetectorAndroid
import com.facenet.mobile.core.RealtimeFrameDetection
import kotlinx.coroutines.launch
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@Composable
fun CameraCaptureView(
    modifier: Modifier = Modifier,
    detectionIntervalMs: Long = 280L,
    onFrameForIdentify: ((RealtimeFrameDetection?) -> Unit)? = null,
    onReady: (capture: (onBitmap: (Bitmap?) -> Unit) -> Unit) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val previewViewState = remember { mutableStateOf<PreviewView?>(null) }
    val imageCaptureState = remember { mutableStateOf<ImageCapture?>(null) }
    val faceBoxesState = remember { mutableStateOf<List<FaceDetection>>(emptyList()) }
    val srcWidthState = remember { mutableStateOf(0) }
    val srcHeightState = remember { mutableStateOf(0) }
    val detector = remember(context) { FaceDetectorAndroid(context) }
    val frameConverter = remember { AnalysisFrameBitmapConverter() }
    val scope = androidx.compose.runtime.rememberCoroutineScope()
    val lastAnalyzeAtMs = remember { longArrayOf(0L) }
    val overlayCache = remember { OverlayStateCache() }

    val currentInterval by rememberUpdatedState(detectionIntervalMs)
    val currentOnFrameForIdentify by rememberUpdatedState(onFrameForIdentify)

    DisposableEffect(detector, cameraExecutor, frameConverter) {
        onDispose {
            cameraExecutor.shutdown()
            detector.close()
            frameConverter.close()
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
        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        imageAnalysis.targetRotation = previewView.display.rotation
        imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
            val now = SystemClock.elapsedRealtime()
            if (now - lastAnalyzeAtMs[0] < currentInterval) {
                imageProxy.close()
                return@setAnalyzer
            }
            lastAnalyzeAtMs[0] = now
            analyzeFrame(
                imageProxy = imageProxy,
                detector = detector,
                frameConverter = frameConverter,
                onBoxesReady = { boxes, width, height ->
                    if (overlayCache.updateIfChanged(boxes, width, height)) {
                        scope.launch {
                            srcWidthState.value = width
                            srcHeightState.value = height
                            faceBoxesState.value = boxes
                        }
                    }
                },
                onFrameForIdentify = currentOnFrameForIdentify
            )
        }
        imageCaptureState.value = imageCapture
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_FRONT_CAMERA,
            preview,
            imageCapture,
            imageAnalysis
        )
    }

    val captureFunc: (onBitmap: (Bitmap?) -> Unit) -> Unit = remember(imageCaptureState, cameraExecutor) {
        { onBitmap ->
            val imageCapture = imageCaptureState.value
            if (imageCapture == null) {
                onBitmap(null)
            } else {
                captureToBitmap(context, imageCapture, cameraExecutor, onBitmap)
            }
        }
    }
    LaunchedEffect(Unit) {
        onReady(captureFunc)
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
            srcHeight = srcHeightState.value,
            mirrorX = true
        )
    }
}

private fun analyzeFrame(
    imageProxy: ImageProxy,
    detector: FaceDetectorAndroid,
    frameConverter: AnalysisFrameBitmapConverter,
    onBoxesReady: (List<FaceDetection>, Int, Int) -> Unit,
    onFrameForIdentify: ((RealtimeFrameDetection?) -> Unit)?
) {
    try {
        val rotation = imageProxy.imageInfo.rotationDegrees
        val plane = imageProxy.planes[0]
        val frameWidth = if (rotation == 90 || rotation == 270) imageProxy.height else imageProxy.width
        val frameHeight = if (rotation == 90 || rotation == 270) imageProxy.width else imageProxy.height
        val boxes = detector.detectRgba(
            rgbaBuffer = plane.buffer,
            width = imageProxy.width,
            height = imageProxy.height,
            rowStride = plane.rowStride,
            pixelStride = plane.pixelStride,
            rotationDegrees = rotation,
            scoreThresh = 0.45f,
            nmsIouThresh = 0.4f,
            topK = 5
        )
        onBoxesReady(boxes, frameWidth, frameHeight)
        if (onFrameForIdentify != null) {
            val frame = if (boxes.isEmpty()) null else frameConverter.convertFaceRoi(imageProxy, boxes[0])
            onFrameForIdentify(frame)
        }
    } finally {
        imageProxy.close()
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

private class OverlayStateCache {
    private var width: Int = 0
    private var height: Int = 0
    private val boxes = mutableListOf<FaceDetection>()

    fun updateIfChanged(currentBoxes: List<FaceDetection>, currentWidth: Int, currentHeight: Int): Boolean {
        if (isSameState(currentBoxes, currentWidth, currentHeight)) return false
        width = currentWidth
        height = currentHeight
        boxes.clear()
        currentBoxes.forEach { box ->
            boxes += FaceDetection(
                x1 = box.x1,
                y1 = box.y1,
                x2 = box.x2,
                y2 = box.y2,
                score = box.score,
                kps = box.kps.copyOf()
            )
        }
        return true
    }

    private fun isSameState(currentBoxes: List<FaceDetection>, currentWidth: Int, currentHeight: Int): Boolean {
        if (width != currentWidth || height != currentHeight) return false
        if (boxes.size != currentBoxes.size) return false
        for (i in boxes.indices) {
            if (!isSameDetection(boxes[i], currentBoxes[i])) return false
        }
        return true
    }

    private fun isSameDetection(a: FaceDetection, b: FaceDetection, epsilon: Float = 1.5f): Boolean {
        if (kotlin.math.abs(a.x1 - b.x1) > epsilon) return false
        if (kotlin.math.abs(a.y1 - b.y1) > epsilon) return false
        if (kotlin.math.abs(a.x2 - b.x2) > epsilon) return false
        if (kotlin.math.abs(a.y2 - b.y2) > epsilon) return false
        if (kotlin.math.abs(a.score - b.score) > 0.02f) return false
        if (a.kps.size != b.kps.size) return false
        for (i in a.kps.indices) {
            if (kotlin.math.abs(a.kps[i] - b.kps[i]) > epsilon) return false
        }
        return true
    }
}

private val FaceBoxColor = Color(0xFF00E676)
private val FaceBoxStroke = Stroke(width = 4f)

@Composable
private fun FaceBoxOverlay(
    boxes: List<FaceDetection>,
    srcWidth: Int,
    srcHeight: Int,
    mirrorX: Boolean
) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        if (srcWidth <= 0 || srcHeight <= 0) return@Canvas
        val scale = maxOf(size.width / srcWidth.toFloat(), size.height / srcHeight.toFloat())
        val scaledW = srcWidth * scale
        val scaledH = srcHeight * scale
        val dx = (scaledW - size.width) * 0.5f
        val dy = (scaledH - size.height) * 0.5f

        boxes.forEach { b ->
            val mappedX1 = b.x1 * scale - dx
            val y1 = b.y1 * scale - dy
            val mappedX2 = b.x2 * scale - dx
            val y2 = b.y2 * scale - dy
            val x1 = if (mirrorX) size.width - mappedX2 else mappedX1
            val x2 = if (mirrorX) size.width - mappedX1 else mappedX2
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
            drawLine(
                color = FaceBoxColor,
                start = Offset(x2, y1),
                end = Offset(x2 - 26f, y1),
                strokeWidth = 6f,
                cap = StrokeCap.Round
            )
            drawLine(
                color = FaceBoxColor,
                start = Offset(x2, y1),
                end = Offset(x2, y1 + 26f),
                strokeWidth = 6f,
                cap = StrokeCap.Round
            )
            drawLine(
                color = FaceBoxColor,
                start = Offset(x1, y2),
                end = Offset(x1 + 26f, y2),
                strokeWidth = 6f,
                cap = StrokeCap.Round
            )
            drawLine(
                color = FaceBoxColor,
                start = Offset(x1, y2),
                end = Offset(x1, y2 - 26f),
                strokeWidth = 6f,
                cap = StrokeCap.Round
            )
            drawLine(
                color = FaceBoxColor,
                start = Offset(x2, y2),
                end = Offset(x2 - 26f, y2),
                strokeWidth = 6f,
                cap = StrokeCap.Round
            )
            drawLine(
                color = FaceBoxColor,
                start = Offset(x2, y2),
                end = Offset(x2, y2 - 26f),
                strokeWidth = 6f,
                cap = StrokeCap.Round
            )
        }
    }
}

private class AnalysisFrameBitmapConverter : AutoCloseable {
    private var rgbaBitmap: Bitmap? = null
    private var rotatedBitmap: Bitmap? = null
    private var rotatedCanvas: Canvas? = null
    private var faceBitmap: Bitmap? = null
    private var packedRgba = ByteArray(0)
    private var packedBuffer: ByteBuffer = ByteBuffer.wrap(packedRgba)
    private val rotateMatrix = Matrix()
    private val faceRoiRect = Rect()

    fun convert(imageProxy: ImageProxy): Bitmap {
        val width = imageProxy.width
        val height = imageProxy.height
        val srcBitmap = ensureRgbaBitmap(width, height)
        copyRgbaPlane(imageProxy, width, height)
        packedBuffer.rewind()
        srcBitmap.copyPixelsFromBuffer(packedBuffer)

        val rotation = imageProxy.imageInfo.rotationDegrees
        if (rotation == 0) return srcBitmap

        val dstWidth = if (rotation == 90 || rotation == 270) height else width
        val dstHeight = if (rotation == 90 || rotation == 270) width else height
        val dstBitmap = ensureRotatedBitmap(dstWidth, dstHeight)
        val canvas = rotatedCanvas ?: Canvas(dstBitmap).also { rotatedCanvas = it }
        canvas.drawColor(AndroidColor.BLACK)

        rotateMatrix.reset()
        when (rotation) {
            90 -> {
                rotateMatrix.postRotate(90f)
                rotateMatrix.postTranslate(dstWidth.toFloat(), 0f)
            }
            180 -> {
                rotateMatrix.postRotate(180f)
                rotateMatrix.postTranslate(dstWidth.toFloat(), dstHeight.toFloat())
            }
            270 -> {
                rotateMatrix.postRotate(270f)
                rotateMatrix.postTranslate(0f, dstHeight.toFloat())
            }
        }
        canvas.drawBitmap(srcBitmap, rotateMatrix, null)
        return dstBitmap
    }

    fun convertFaceRoi(imageProxy: ImageProxy, detection: FaceDetection): RealtimeFrameDetection? {
        val rotation = imageProxy.imageInfo.rotationDegrees
        val frameWidth = if (rotation == 90 || rotation == 270) imageProxy.height else imageProxy.width
        val frameHeight = if (rotation == 90 || rotation == 270) imageProxy.width else imageProxy.height
        if (!buildFaceRoiRect(frameWidth, frameHeight, detection, faceRoiRect)) return null

        val roiWidth = faceRoiRect.width()
        val roiHeight = faceRoiRect.height()
        if (roiWidth <= 0 || roiHeight <= 0) return null

        val dstBitmap = ensureFaceBitmap(roiWidth, roiHeight)
        copyRgbaRoi(imageProxy, faceRoiRect, rotation, roiWidth, roiHeight)
        packedBuffer.rewind()
        dstBitmap.copyPixelsFromBuffer(packedBuffer)

        return RealtimeFrameDetection(
            bitmap = dstBitmap,
            detection = offsetDetection(detection, faceRoiRect.left.toFloat(), faceRoiRect.top.toFloat()),
            sourceWidth = roiWidth,
            sourceHeight = roiHeight
        )
    }

    private fun ensureRgbaBitmap(width: Int, height: Int): Bitmap {
        val current = rgbaBitmap
        if (current != null && current.width == width && current.height == height) return current
        current?.recycle()
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { rgbaBitmap = it }
    }

    private fun ensureRotatedBitmap(width: Int, height: Int): Bitmap {
        val current = rotatedBitmap
        if (current != null && current.width == width && current.height == height) return current
        current?.recycle()
        rotatedCanvas = null
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { rotatedBitmap = it }
    }

    private fun ensureFaceBitmap(width: Int, height: Int): Bitmap {
        val current = faceBitmap
        if (current != null && current.width == width && current.height == height) return current
        current?.recycle()
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { faceBitmap = it }
    }

    private fun copyRgbaPlane(imageProxy: ImageProxy, width: Int, height: Int) {
        val required = width * height * 4
        if (packedRgba.size != required) {
            packedRgba = ByteArray(required)
            packedBuffer = ByteBuffer.wrap(packedRgba)
        }
        val plane = imageProxy.planes[0]
        val src = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val rowBytes = width * 4

        if (rowStride == rowBytes) {
            src.rewind()
            src.get(packedRgba, 0, required)
            return
        }

        for (row in 0 until height) {
            src.position(row * rowStride)
            src.get(packedRgba, row * rowBytes, rowBytes)
        }
    }

    private fun copyRgbaRoi(
        imageProxy: ImageProxy,
        roi: Rect,
        rotationDegrees: Int,
        dstWidth: Int,
        dstHeight: Int
    ) {
        val required = dstWidth * dstHeight * 4
        if (packedRgba.size != required) {
            packedRgba = ByteArray(required)
            packedBuffer = ByteBuffer.wrap(packedRgba)
        }
        val plane = imageProxy.planes[0]
        val src = plane.buffer.duplicate()
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val rawWidth = imageProxy.width
        val rawHeight = imageProxy.height

        var out = 0
        for (y in 0 until dstHeight) {
            val frameY = roi.top + y
            for (x in 0 until dstWidth) {
                val frameX = roi.left + x
                val rawX: Int
                val rawY: Int
                when (rotationDegrees) {
                    90 -> {
                        rawX = frameY
                        rawY = rawHeight - 1 - frameX
                    }
                    180 -> {
                        rawX = rawWidth - 1 - frameX
                        rawY = rawHeight - 1 - frameY
                    }
                    270 -> {
                        rawX = rawWidth - 1 - frameY
                        rawY = frameX
                    }
                    else -> {
                        rawX = frameX
                        rawY = frameY
                    }
                }
                val base = rawY * rowStride + rawX * pixelStride
                packedRgba[out] = src.get(base)
                packedRgba[out + 1] = src.get(base + 1)
                packedRgba[out + 2] = src.get(base + 2)
                packedRgba[out + 3] = if (pixelStride >= 4) src.get(base + 3) else 0xFF.toByte()
                out += 4
            }
        }
    }

    private fun buildFaceRoiRect(frameWidth: Int, frameHeight: Int, detection: FaceDetection, outRect: Rect): Boolean {
        if (frameWidth <= 0 || frameHeight <= 0) return false
        var left = detection.x1
        var top = detection.y1
        var right = detection.x2
        var bottom = detection.y2
        var hasLandmarks = false
        for (i in 0 until detection.kps.size step 2) {
            val x = detection.kps[i]
            val y = detection.kps[i + 1]
            if (x == 0f && y == 0f) continue
            hasLandmarks = true
            left = minOf(left, x)
            top = minOf(top, y)
            right = maxOf(right, x)
            bottom = maxOf(bottom, y)
        }
        val side = maxOf(right - left, bottom - top, detection.width, detection.height) * 1.45f
        val cx = if (hasLandmarks) (left + right) * 0.5f else (detection.x1 + detection.x2) * 0.5f
        val cy = if (hasLandmarks) (top + bottom) * 0.5f else (detection.y1 + detection.y2) * 0.5f
        val x1 = (cx - side * 0.5f).toInt().coerceIn(0, frameWidth - 1)
        val y1 = (cy - side * 0.5f).toInt().coerceIn(0, frameHeight - 1)
        val x2 = (cx + side * 0.5f).toInt().coerceIn(x1 + 1, frameWidth)
        val y2 = (cy + side * 0.5f).toInt().coerceIn(y1 + 1, frameHeight)
        outRect.set(x1, y1, x2, y2)
        return true
    }

    private fun offsetDetection(detection: FaceDetection, dx: Float, dy: Float): FaceDetection {
        val shiftedKps = FloatArray(detection.kps.size)
        for (i in shiftedKps.indices step 2) {
            shiftedKps[i] = detection.kps[i] - dx
            shiftedKps[i + 1] = detection.kps[i + 1] - dy
        }
        return FaceDetection(
            x1 = detection.x1 - dx,
            y1 = detection.y1 - dy,
            x2 = detection.x2 - dx,
            y2 = detection.y2 - dy,
            score = detection.score,
            kps = shiftedKps
        )
    }

    override fun close() {
        rgbaBitmap?.recycle()
        rotatedBitmap?.recycle()
        faceBitmap?.recycle()
    }
}
