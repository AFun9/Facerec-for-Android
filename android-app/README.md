# Face Register Android MVP

## 功能

- 摄像头预览（CameraX）
- 注册页：默认 3 张采集，支持 1 张完成注册
- 识别页：拍照识别用户与相似度
- 本地推理：`facedet.onnx` + `facenet.onnx` + `affecnet7.onnx`
- 本地存储：`filesDir/facereg/registry.bin` 与 `registry.json`

## 运行前准备

1. 在 `app/src/main/assets/` 放入：
   - `facedet.onnx`
   - `facenet.onnx`
   - `affecnet7.onnx`
2. 用 Android Studio 打开 `android-app` 目录并同步 Gradle。
3. 连接真机（建议 Android 8.0+）运行。

## 阈值建议

- 初始识别阈值：`0.45`
- 注册建议：同一人采集 3 张（光照和角度略有变化）
