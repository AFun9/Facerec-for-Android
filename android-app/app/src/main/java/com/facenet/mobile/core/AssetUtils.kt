package com.facenet.mobile.core

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * 将 assets 中的模型文件解压到 filesDir，首次调用后缓存。
 *
 * 完整性策略：
 *  - 若 asset 未压缩（build.gradle 设置了 noCompress "onnx"），使用 openFd 获取精确大小做完整性校验，
 *    能检测上次 App 崩溃导致的部分写入。
 *  - 若 asset 被压缩（旧包或其他格式），openFd 抛 IOException，此时退化为：文件存在且非空即命中缓存。
 */
internal fun materializeAsset(context: Context, assetName: String): File {
    val outFile = File(context.filesDir, assetName)

    val assetSize: Long = try {
        context.assets.openFd(assetName).use { it.length }
    } catch (_: IOException) {
        // asset 已被压缩，openFd 不可用，退化为存在性检查
        -1L
    }

    val cached = when {
        assetSize >= 0 -> outFile.exists() && outFile.length() == assetSize  // 精确校验
        else           -> outFile.exists() && outFile.length() > 0           // 宽松校验
    }
    if (cached) return outFile

    // 文件不存在或大小不匹配（写入不完整），重新拷贝
    context.assets.open(assetName).use { input ->
        FileOutputStream(outFile).use { output ->
            input.copyTo(output)
        }
    }
    return outFile
}
