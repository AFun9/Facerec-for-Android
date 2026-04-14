Put model files here before running the app on device:

- facedet.onnx
- facenet.onnx
- affecnet7.onnx

These names must match the defaults used by:
- FaceDetectorAndroid(modelAssetName = "facedet.onnx")
- FaceNetEmbedderAndroid(modelAssetName = "facenet.onnx")
- EmotionClassifierAndroid(modelAssetName = "affecnet7.onnx")
