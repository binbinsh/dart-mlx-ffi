package com.example.dart_inference_example

import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        registerRuntimeSmokeChannel(flutterEngine, "dart_mlx_ffi/runtime_smoke")
        registerRuntimeSmokeChannel(flutterEngine, "dart_inference/runtime_smoke")
    }

    private fun registerRuntimeSmokeChannel(flutterEngine: FlutterEngine, name: String) {
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, name).setMethodCallHandler {
            call,
            result ->
            if (call.method != "logLine") {
                result.notImplemented()
                return@setMethodCallHandler
            }
            val line = call.arguments as? String
            if (line == null) {
                result.error("invalid_argument", "logLine expects a string.", null)
                return@setMethodCallHandler
            }
            Log.i(runtimeSmokeTag(line), line)
            result.success(null)
        }
    }

    private fun runtimeSmokeTag(line: String): String {
        return when {
            line.startsWith("DINF_RUNTIME_SMOKE") -> "DINF_RUNTIME_SMOKE"
            line.startsWith("DMF_RUNTIME_SMOKE") -> "DMF_RUNTIME_SMOKE"
            else -> "DMF_RUNTIME_SMOKE"
        }
    }
}
