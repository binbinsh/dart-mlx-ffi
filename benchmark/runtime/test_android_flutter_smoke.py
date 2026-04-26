from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import android_flutter_smoke
from flutter_smoke_common import MarkerParser, extract_marker_payload


class AndroidFlutterSmokeTest(unittest.TestCase):
    def test_extract_marker_payload(self) -> None:
        line = (
            "DINF_RUNTIME_SMOKE_RESULT:"
            '{"passed":true,"engine":"litert","model_id":"qwen2_5"}'
        )
        payload = extract_marker_payload(line)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["engine"], "litert")

    def test_marker_parser_chunked_payload(self) -> None:
        parser = MarkerParser()
        self.assertIsNone(parser.feed("DINF_RUNTIME_SMOKE_RESULT_BEGIN:2"))
        self.assertIsNone(
            parser.feed("DINF_RUNTIME_SMOKE_RESULT_CHUNK:1/2:eyJwYXNzZWQiOnRydWUs")
        )
        payload = parser.feed("DINF_RUNTIME_SMOKE_RESULT_CHUNK:2/2:ImVuZ2luZSI6ImxpdGVydCJ9")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["engine"], "litert")
        self.assertIsNone(parser.feed("DINF_RUNTIME_SMOKE_RESULT_END"))

    def test_marker_parser_chunked_payload_without_end(self) -> None:
        parser = MarkerParser()
        self.assertIsNone(parser.feed("DINF_RUNTIME_SMOKE_RESULT_BEGIN:1"))
        payload = parser.feed("DINF_RUNTIME_SMOKE_RESULT_CHUNK:1/1:eyJwYXNzZWQiOnRydWV9")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])

    def test_detect_android_device_id(self) -> None:
        devices = [
            {"id": "ios-device", "targetPlatform": "ios", "isSupported": True},
            {
                "id": "android-device",
                "targetPlatform": "android-arm64",
                "isSupported": True,
            },
        ]
        completed = mock.Mock(returncode=0, stdout=json.dumps(devices))
        with mock.patch("android_flutter_smoke.subprocess.run", return_value=completed):
            device_id = android_flutter_smoke.detect_android_device_id()
        self.assertEqual(device_id, "android-device")

    def test_detect_android_abi(self) -> None:
        completed = mock.Mock(returncode=0, stdout="arm64-v8a\n", stderr="")
        with mock.patch("android_flutter_smoke._run_adb", return_value=completed):
            abi = android_flutter_smoke.detect_android_abi("android-device")
        self.assertEqual(abi, "arm64-v8a")

    def test_flutter_command_includes_artifact_define(self) -> None:
        command = android_flutter_smoke.flutter_command(
            device_id="android-device",
            device_user="0",
            model_id="qwen2_5",
            engine="litert",
            artifact="hf://litert-community/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct_seq128_q8_ekv1280.tflite",
            build_mode="release",
        )
        self.assertIn("--device-user=0", command)
        self.assertIn(
            "--dart-define=DINF_RUNTIME_SMOKE_ARTIFACT=hf://litert-community/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct_seq128_q8_ekv1280.tflite",
            command,
        )
        self.assertIn("--release", command)

    def test_logcat_follow_command(self) -> None:
        command = android_flutter_smoke.logcat_follow_command(device_id="android-device")
        self.assertEqual(command[:4], ["adb", "-s", "android-device", "logcat"])
        self.assertIn("DINF_RUNTIME_SMOKE:I", command)
        self.assertEqual(command[-2:], ["flutter:I", "*:S"])

    def test_parse_hf_uri(self) -> None:
        repo_id, path = android_flutter_smoke.parse_hf_uri(
            "hf://onnx-community/silero-vad/onnx/model.onnx"
        )
        self.assertEqual(repo_id, "onnx-community/silero-vad")
        self.assertEqual(path, "onnx/model.onnx")

    def test_artifact_permission_paths_for_default_android_root(self) -> None:
        paths = android_flutter_smoke._artifact_permission_paths(
            package_name="com.example.app",
            remote_root="/sdcard/Android/data/com.example.app/files/dart_inference_runtime_smoke",
            remote_dir="/sdcard/Android/data/com.example.app/files/dart_inference_runtime_smoke/silero_vad/onnx",
        )
        self.assertIn("/sdcard/Android/data/com.example.app", paths)
        self.assertIn("/sdcard/Android/data/com.example.app/files", paths)
        self.assertIn(
            "/sdcard/Android/data/com.example.app/files/dart_inference_runtime_smoke",
            paths,
        )
        self.assertIn(
            "/sdcard/Android/data/com.example.app/files/dart_inference_runtime_smoke/silero_vad",
            paths,
        )
        self.assertIn(
            "/sdcard/Android/data/com.example.app/files/dart_inference_runtime_smoke/silero_vad/onnx",
            paths,
        )

    def test_push_artifact_to_device_applies_permissions(self) -> None:
        local = Path("/tmp/model.onnx")
        calls: list[list[str]] = []

        def run_adb(
            device_id: str,
            args: list[str],
            *,
            check: bool,
        ) -> mock.Mock:
            calls.append(args)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch("android_flutter_smoke._run_adb", side_effect=run_adb):
            remote = android_flutter_smoke.push_artifact_to_device(
                device_id="android-device",
                local_path=local,
                package_name="com.example.app",
                model_id="silero_vad",
                engine="onnx",
                device_artifact_dir=None,
            )

        self.assertEqual(
            remote,
            "/sdcard/Android/data/com.example.app/files/dart_inference_runtime_smoke/silero_vad/onnx/model.onnx",
        )
        mkdir_calls = [entry for entry in calls if entry[:2] == ["shell", "mkdir"]]
        self.assertEqual(len(mkdir_calls), 1)
        push_calls = [entry for entry in calls if entry and entry[0] == "push"]
        self.assertEqual(len(push_calls), 1)
        chmod_calls = [entry for entry in calls if entry[:2] == ["shell", "chmod"]]
        self.assertGreaterEqual(len(chmod_calls), 2)

    def test_resolve_android_artifact_prefetch_success(self) -> None:
        local = Path("/tmp/model.onnx")
        with (
            mock.patch(
                "android_flutter_smoke.download_hf_artifact",
                return_value=local,
            ),
            mock.patch(
                "android_flutter_smoke.push_artifact_to_device",
                return_value="/sdcard/Android/data/com.example.app/files/runtime/model.onnx",
            ),
        ):
            resolved, prepare = android_flutter_smoke.resolve_android_artifact(
                artifact="hf://onnx-community/silero-vad/onnx/model.onnx",
                device_id="android-device",
                package_name="com.example.app",
                model_id="silero_vad",
                engine="onnx",
                hf_cache_dir=Path("/tmp/cache"),
                device_artifact_dir=None,
                host_prefetch=True,
            )
        self.assertEqual(
            resolved,
            "/sdcard/Android/data/com.example.app/files/runtime/model.onnx",
        )
        self.assertEqual(prepare["mode"], "host_prefetch")

    def test_resolve_android_artifact_prefetch_fallback(self) -> None:
        with mock.patch(
            "android_flutter_smoke.download_hf_artifact",
            side_effect=RuntimeError("network down"),
        ):
            resolved, prepare = android_flutter_smoke.resolve_android_artifact(
                artifact="hf://onnx-community/silero-vad/onnx/model.onnx",
                device_id="android-device",
                package_name="com.example.app",
                model_id="silero_vad",
                engine="onnx",
                hf_cache_dir=Path("/tmp/cache"),
                device_artifact_dir=None,
                host_prefetch=True,
            )
        self.assertEqual(resolved, "hf://onnx-community/silero-vad/onnx/model.onnx")
        self.assertEqual(prepare["mode"], "direct_fallback")

    def test_capture_runtime_smoke_android_reads_logcat_marker(self) -> None:
        flutter_proc = subprocess.Popen(
            [sys.executable, "-c", "import time; print('flutter boot', flush=True); time.sleep(1)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        logcat_script = textwrap.dedent(
            """
            import time
            print("DINF_RUNTIME_SMOKE_RESULT_BEGIN:1", flush=True)
            print("DINF_RUNTIME_SMOKE_RESULT_CHUNK:1/1:eyJwYXNzZWQiOnRydWV9", flush=True)
            print("DINF_RUNTIME_SMOKE_RESULT_END", flush=True)
            time.sleep(1)
            """
        )
        logcat_proc = subprocess.Popen(
            [sys.executable, "-c", logcat_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        parser = MarkerParser()
        try:
            payload, timed_out, flutter_lines, logcat_lines = (
                android_flutter_smoke.capture_runtime_smoke_android(
                    flutter_process=flutter_proc,
                    logcat_process=logcat_proc,
                    timeout_seconds=5,
                    parser=parser,
                )
            )
        finally:
            android_flutter_smoke.terminate_process(flutter_proc)
            android_flutter_smoke.terminate_process(logcat_proc)

        self.assertFalse(timed_out)
        self.assertIn("flutter boot", flutter_lines)
        self.assertGreaterEqual(len(logcat_lines), 2)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])

    def test_flutter_env_repairs_invalid_java_home_on_macos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = mock.Mock(returncode=0, stdout=f"{tmp}\n")
            with (
                mock.patch.dict(os.environ, {"JAVA_HOME": "/invalid/java/home"}, clear=False),
                mock.patch("android_flutter_smoke.sys.platform", "darwin"),
                mock.patch("android_flutter_smoke.subprocess.run", return_value=completed),
            ):
                env = android_flutter_smoke._flutter_env()
        self.assertEqual(env["JAVA_HOME"], tmp)


if __name__ == "__main__":
    unittest.main()
