from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import ios_flutter_smoke


class IosFlutterSmokeTest(unittest.TestCase):
    def test_extract_marker_payload(self) -> None:
        line = (
            "DINF_RUNTIME_SMOKE_RESULT:"
            '{"passed":true,"engine":"coreml","model_id":"silero_vad"}'
        )
        payload = ios_flutter_smoke.extract_marker_payload(line)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["engine"], "coreml")

    def test_extract_marker_payload_invalid_json(self) -> None:
        payload = ios_flutter_smoke.extract_marker_payload(
            "DINF_RUNTIME_SMOKE_RESULT:{invalid"
        )
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["error"], "Invalid marker JSON")

    def test_extract_marker_payload_accepts_legacy_dmf_marker(self) -> None:
        payload = ios_flutter_smoke.extract_marker_payload(
            'DMF_RUNTIME_SMOKE_RESULT:{"passed":true,"engine":"coreml"}'
        )
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["engine"], "coreml")

    def test_marker_parser_chunked_payload(self) -> None:
        parser = ios_flutter_smoke.MarkerParser()
        self.assertIsNone(parser.feed("DINF_RUNTIME_SMOKE_RESULT_BEGIN:2"))
        self.assertIsNone(
            parser.feed("DINF_RUNTIME_SMOKE_RESULT_CHUNK:1/2:eyJwYXNzZWQiOnRydWUs")
        )
        payload = parser.feed("DINF_RUNTIME_SMOKE_RESULT_CHUNK:2/2:ImVuZ2luZSI6ImNvcmVtbCJ9")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["engine"], "coreml")
        self.assertIsNone(parser.feed("DINF_RUNTIME_SMOKE_RESULT_END"))

    def test_marker_parser_incomplete_chunks(self) -> None:
        parser = ios_flutter_smoke.MarkerParser()
        self.assertIsNone(parser.feed("DINF_RUNTIME_SMOKE_RESULT_BEGIN:2"))
        self.assertIsNone(
            parser.feed("DINF_RUNTIME_SMOKE_RESULT_CHUNK:1/2:eyJwYXNzZWQiOnRydWV9")
        )
        payload = parser.feed("DINF_RUNTIME_SMOKE_RESULT_END")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["error"], "Incomplete marker chunks")

    def test_detect_ios_device_id(self) -> None:
        devices = [
            {"id": "android-device", "targetPlatform": "android", "isSupported": True},
            {"id": "ios-device", "targetPlatform": "ios", "isSupported": True},
        ]
        completed = mock.Mock(returncode=0, stdout=json.dumps(devices))
        with mock.patch("ios_flutter_smoke.subprocess.run", return_value=completed):
            device_id = ios_flutter_smoke.detect_ios_device_id()
        self.assertEqual(device_id, "ios-device")

    def test_flutter_command_includes_artifact_define(self) -> None:
        command = ios_flutter_smoke.flutter_command(
            device_id="ios-device",
            model_id="silero_vad",
            engine="coreml",
            artifact="hf://FluidInference/silero-vad-coreml/silero-vad-unified-v6.0.0.mlmodelc",
        )
        self.assertIn("--target", command)
        self.assertIn("lib/main.dart", command)
        self.assertIn(
            "--dart-define=DINF_RUNTIME_SMOKE_ARTIFACT="
            "hf://FluidInference/silero-vad-coreml/silero-vad-unified-v6.0.0.mlmodelc",
            command,
        )

    def test_ios_syslog_command_uses_udid(self) -> None:
        with mock.patch(
            "ios_flutter_smoke.shutil.which",
            return_value="/opt/homebrew/bin/idevicesyslog",
        ):
            command = ios_flutter_smoke.ios_syslog_command(device_id="ios-device")
        self.assertEqual(command, ["idevicesyslog", "-u", "ios-device"])

    def test_capture_runtime_smoke_ios_reads_syslog_marker(self) -> None:
        flutter_proc = subprocess.Popen(
            [sys.executable, "-c", "import time; print('flutter boot', flush=True); time.sleep(1)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        syslog_script = textwrap.dedent(
            """
            import time
            print("DMF_RUNTIME_SMOKE_RESULT_BEGIN:1", flush=True)
            print("DMF_RUNTIME_SMOKE_RESULT_CHUNK:1/1:eyJwYXNzZWQiOnRydWV9", flush=True)
            print("DMF_RUNTIME_SMOKE_RESULT_END", flush=True)
            time.sleep(1)
            """
        )
        syslog_proc = subprocess.Popen(
            [sys.executable, "-c", syslog_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            payload, timed_out, flutter_lines, syslog_lines = (
                ios_flutter_smoke.capture_runtime_smoke_ios(
                    flutter_process=flutter_proc,
                    syslog_process=syslog_proc,
                    timeout_seconds=5,
                    parser=ios_flutter_smoke.MarkerParser(),
                )
            )
        finally:
            ios_flutter_smoke.terminate_process(flutter_proc)
            ios_flutter_smoke.terminate_process(syslog_proc)

        self.assertFalse(timed_out)
        self.assertIn("flutter boot", flutter_lines)
        self.assertGreaterEqual(len(syslog_lines), 2)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])

    def test_capture_runtime_smoke_ios_exits_after_flutter_failure_grace(self) -> None:
        flutter_proc = subprocess.Popen(
            [sys.executable, "-c", "print('build failed', flush=True)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        syslog_proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            payload, timed_out, flutter_lines, _ = (
                ios_flutter_smoke.capture_runtime_smoke_ios(
                    flutter_process=flutter_proc,
                    syslog_process=syslog_proc,
                    timeout_seconds=5,
                    parser=ios_flutter_smoke.MarkerParser(),
                    post_flutter_exit_grace_seconds=0.1,
                )
            )
        finally:
            ios_flutter_smoke.terminate_process(flutter_proc)
            ios_flutter_smoke.terminate_process(syslog_proc)

        self.assertFalse(timed_out)
        self.assertIn("build failed", flutter_lines)
        self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
