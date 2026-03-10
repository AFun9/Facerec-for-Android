package com.facenet.mobile.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceMathTest {
    @Test
    fun cosineSimilarity_returnsExpectedRange() {
        val a = FloatArray(512) { if (it % 2 == 0) 1f else 0f }
        val b = FloatArray(512) { if (it % 2 == 0) 1f else 0f }
        val c = FloatArray(512) { if (it % 2 == 0) -1f else 0f }

        val same = FaceNetEmbedderAndroid.cosineSimilarity(a, b)
        val opposite = FaceNetEmbedderAndroid.cosineSimilarity(a, c)
        assertEquals(1.0f, same, 1e-5f)
        assertTrue(opposite < 0f)
    }
}
