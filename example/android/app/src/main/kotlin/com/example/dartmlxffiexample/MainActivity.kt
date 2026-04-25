package com.example.dartmlxffiexample

import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

private const val runtimeSmokeChannel = "dart_mlx_ffi/runtime_smoke"
private const val runtimeSmokeTag = "DMF_RUNTIME_SMOKE"

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, runtimeSmokeChannel)
            .setMethodCallHandler { call, result ->
                if (call.method != "logLine") {
                    result.notImplemented()
                    return@setMethodCallHandler
                }
                val line = call.arguments as? String
                if (line.isNullOrBlank()) {
                    result.error("invalid_args", "Expected non-empty log line.", null)
                    return@setMethodCallHandler
                }
                Log.i(runtimeSmokeTag, line)
                result.success(null)
            }
    }
}
