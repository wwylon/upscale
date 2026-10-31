package com.upscale.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable

/**
 * Placeholder entry — wire to remote `upscale serve` or NCNN runtime later.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                UpscalePlaceholder()
            }
        }
    }
}

@Composable
fun UpscalePlaceholder() {
    Text("Upscale Android skeleton — see android/README.md")
}
