from __future__ import annotations

import json
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

import ios_flutter_smoke


class IosFlutterSmokeTest(unittest.TestCase):
    def test_extract_marker_payload(self) -> None:
        line = (
            "DMF_RUNTIME_SMOKE_RESULT:"
            '{"passed":true,"engine":"coreml","model_id":"silero_vad"}'
        )
        payload = ios_flutter_smoke.extract_marker_payload(line)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["engine"], "coreml")

    def test_extract_marker_payload_invalid_json(self) -> None:
        payload = ios_flutter_smoke.extract_marker_payload(
            "DMF_RUNTIME_SMOKE_RESULT:{invalid"
        )
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["error"], "Invalid marker JSON")

    def test_marker_parser_chunked_payload(self) -> None:
        parser = ios_flutter_smoke.MarkerParser()
        self.assertIsNone(parser.feed("DMF_RUNTIME_SMOKE_RESULT_BEGIN:2"))
        self.assertIsNone(
            parser.feed("DMF_RUNTIME_SMOKE_RESULT_CHUNK:1/2:eyJwYXNzZWQiOnRydWUs")
        )
        payload = parser.feed("DMF_RUNTIME_SMOKE_RESULT_CHUNK:2/2:ImVuZ2luZSI6ImNvcmVtbCJ9")
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["engine"], "coreml")
        self.assertIsNone(parser.feed("DMF_RUNTIME_SMOKE_RESULT_END"))

    def test_marker_parser_incomplete_chunks(self) -> None:
        parser = ios_flutter_smoke.MarkerParser()
        self.assertIsNone(parser.feed("DMF_RUNTIME_SMOKE_RESULT_BEGIN:2"))
        self.assertIsNone(
            parser.feed("DMF_RUNTIME_SMOKE_RESULT_CHUNK:1/2:eyJwYXNzZWQiOnRydWV9")
        )
        payload = parser.feed("DMF_RUNTIME_SMOKE_RESULT_END")
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
            build_mode="release",
            artifact="hf://FluidInference/silero-vad-coreml/silero-vad-unified-v6.0.0.mlmodelc",
        )
        self.assertIn("--release", command)
        self.assertIn("--dart-define=DMF_RUNTIME_SMOKE_ARTIFACT=hf://FluidInference/silero-vad-coreml/silero-vad-unified-v6.0.0.mlmodelc", command)

    def test_release_timeout_without_marker_retries_debug_by_default(self) -> None:
        self.assertTrue(
            ios_flutter_smoke._should_retry_debug_after_release_timeout(
                {"status": "timeout", "runtime_smoke": None},
                build_mode="release",
                disabled=False,
            )
        )
        self.assertFalse(
            ios_flutter_smoke._should_retry_debug_after_release_timeout(
                {"status": "timeout", "runtime_smoke": None},
                build_mode="debug",
                disabled=False,
            )
        )
        self.assertFalse(
            ios_flutter_smoke._should_retry_debug_after_release_timeout(
                {"status": "timeout", "runtime_smoke": {"passed": False}},
                build_mode="release",
                disabled=False,
            )
        )

    def test_resolve_ios_artifact_hf_uri_stays_direct(self) -> None:
        resolved, prepare = ios_flutter_smoke.resolve_ios_artifact(
            artifact="hf://FluidInference/silero-vad-coreml/silero-vad-unified-v6.0.0.mlmodelc",
            device_id="ios-device",
            bundle_id="com.example.dartMlxFfiProbe",
            model_id="silero_vad",
            engine="coreml",
            device_artifact_dir="Documents/dart_mlx_ffi_runtime_smoke",
            device_push=True,
            device_copy_retries=2,
            direct_fallback_on_push_failure=False,
        )
        self.assertEqual(
            resolved,
            "hf://FluidInference/silero-vad-coreml/silero-vad-unified-v6.0.0.mlmodelc",
        )
        self.assertEqual(prepare["mode"], "direct")
        self.assertEqual(prepare["reason"], "hf_uri")

    def test_resolve_ios_artifact_local_push(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "model.mlmodelc"
            local.mkdir()
            resolved, prepare = ios_flutter_smoke.resolve_ios_artifact(
                artifact=str(local),
                device_id="ios-device",
                bundle_id="com.example.dartMlxFfiProbe",
                model_id="silero_vad",
                engine="coreml",
                device_artifact_dir="Documents/dart_mlx_ffi_runtime_smoke",
                device_push=True,
                device_copy_retries=2,
                direct_fallback_on_push_failure=False,
            )
        self.assertEqual(
            str(resolved),
            "Documents/dart_mlx_ffi_runtime_smoke/silero_vad/coreml/model.mlmodelc",
        )
        self.assertEqual(prepare["mode"], "local_push_deferred")
        self.assertIn("copy_source", prepare)
        self.assertIn("copy_destination", prepare)

    def test_resolve_ios_artifact_relative_path_resolves_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "benchmark" / "artifacts" / "converted" / "model.mlmodelc"
            local.mkdir(parents=True)
            with mock.patch.object(ios_flutter_smoke, "ROOT", root):
                resolved, prepare = ios_flutter_smoke.resolve_ios_artifact(
                    artifact="benchmark/artifacts/converted/model.mlmodelc",
                    device_id="ios-device",
                    bundle_id="com.example.dartMlxFfiProbe",
                    model_id="silero_vad",
                    engine="coreml",
                    device_artifact_dir="Documents/dart_mlx_ffi_runtime_smoke",
                    device_push=True,
                    device_copy_retries=2,
                    direct_fallback_on_push_failure=False,
                )
        self.assertEqual(
            str(resolved),
            "Documents/dart_mlx_ffi_runtime_smoke/silero_vad/coreml/model.mlmodelc",
        )
        self.assertEqual(prepare["mode"], "local_push_deferred")

    def test_resolve_ios_artifact_no_device_push_stays_direct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "model.mlmodelc"
            local.mkdir()
            resolved, prepare = ios_flutter_smoke.resolve_ios_artifact(
                artifact=str(local),
                device_id="ios-device",
                bundle_id="com.example.dartMlxFfiProbe",
                model_id="silero_vad",
                engine="coreml",
                device_artifact_dir="Documents/dart_mlx_ffi_runtime_smoke",
                device_push=False,
                device_copy_retries=2,
                direct_fallback_on_push_failure=False,
            )
        self.assertEqual(resolved, str(local))
        self.assertEqual(prepare["mode"], "direct")
        self.assertEqual(prepare["reason"], "device_push_disabled")

    def test_resolve_ios_artifact_sets_deferred_fallback_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "model.mlmodelc"
            local.mkdir()
            resolved, prepare = ios_flutter_smoke.resolve_ios_artifact(
                artifact=str(local),
                device_id="ios-device",
                bundle_id="com.example.dartMlxFfiProbe",
                model_id="silero_vad",
                engine="coreml",
                device_artifact_dir="Documents/dart_mlx_ffi_runtime_smoke",
                device_push=True,
                device_copy_retries=2,
                direct_fallback_on_push_failure=True,
            )
        self.assertEqual(
            resolved,
            "Documents/dart_mlx_ffi_runtime_smoke/silero_vad/coreml/model.mlmodelc",
        )
        self.assertEqual(prepare["mode"], "local_push_deferred_fallback")

    def test_deferred_push_mode_helper_accepts_fallback_variant(self) -> None:
        self.assertTrue(ios_flutter_smoke._is_deferred_push_mode("local_push_deferred"))
        self.assertTrue(
            ios_flutter_smoke._is_deferred_push_mode("local_push_deferred_fallback")
        )
        self.assertFalse(ios_flutter_smoke._is_deferred_push_mode("direct"))

    def test_push_artifact_pipeline_uses_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp)
            pipeline = local_dir / "pipeline.json"
            pipeline.write_text(
                json.dumps({"format": "dart_mlx_ffi.coreml_pipeline.v1", "stages": []}),
                encoding="utf-8",
            )
            with mock.patch(
                "ios_flutter_smoke._devicectl_copy_to",
                return_value="/private/var/mobile/Containers/Data/Application/UUID/Documents/dart_mlx_ffi_runtime_smoke/paddle_ocr_vl/coreml",
            ) as copy_to:
                remote_path, meta = ios_flutter_smoke.push_artifact_to_ios_device(
                    device_id="ios-device",
                    bundle_id="com.example.dartMlxFfiProbe",
                    model_id="paddle_ocr_vl",
                    engine="coreml",
                    local_artifact=pipeline,
                    device_artifact_dir="Documents/dart_mlx_ffi_runtime_smoke",
                    copy_retries=2,
                )
        self.assertTrue(remote_path.endswith("/pipeline.json"))
        self.assertEqual(
            meta["copy_destination"],
            "Documents/dart_mlx_ffi_runtime_smoke/paddle_ocr_vl/coreml",
        )
        copy_to.assert_called_once()

    def test_flutter_command_includes_wait_for_artifact_define(self) -> None:
        command = ios_flutter_smoke.flutter_command(
            device_id="ios-device",
            model_id="paddle_ocr_vl",
            engine="coreml",
            build_mode="debug",
            artifact="Documents/dart_mlx_ffi_runtime_smoke/paddle_ocr_vl/coreml/pipeline.json",
            wait_for_artifact_seconds=120,
        )
        self.assertIn(
            "--dart-define=DMF_RUNTIME_SMOKE_WAIT_FOR_ARTIFACT_SECONDS=120",
            command,
        )

    def test_flutter_command_includes_coreml_compute_units_define(self) -> None:
        command = ios_flutter_smoke.flutter_command(
            device_id="ios-device",
            model_id="qwen3_asr",
            engine="coreml",
            build_mode="debug",
            artifact="Documents/dart_mlx_ffi_runtime_smoke/qwen3_asr/coreml/model",
            coreml_compute_units="cpuAndGPU",
        )
        self.assertIn(
            "--dart-define=DMF_RUNTIME_SMOKE_COREML_COMPUTE_UNITS=cpuAndGPU",
            command,
        )

    def test_deferred_ios_push_triggers_on_boot_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.mlmodelc"
            source.mkdir()
            deferred = ios_flutter_smoke._DeferredIosArtifactPush(
                device_id="ios-device",
                bundle_id="com.example.dartMlxFfiProbe",
                source_path=source,
                destination="Documents/dart_mlx_ffi_runtime_smoke/silero_vad/coreml/model.mlmodelc",
                retries=2,
                runtime_artifact="Documents/dart_mlx_ffi_runtime_smoke/silero_vad/coreml/model.mlmodelc",
                direct_fallback_on_push_failure=False,
            )
            with mock.patch(
                "ios_flutter_smoke._devicectl_copy_to",
                return_value="/private/var/mobile/Containers/Data/Application/UUID/Documents/dart_mlx_ffi_runtime_smoke/silero_vad/coreml/model.mlmodelc",
            ):
                deferred.observe_line("flutter: DMF_RUNTIME_SMOKE:BOOT")
        self.assertTrue(deferred.pushed)
        self.assertTrue(deferred.attempted)

    def test_deferred_ios_push_materializes_symlink_trees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob = root / "blob.bin"
            blob.write_bytes(b"weights")
            source = root / "model.mlmodelc"
            source.mkdir()
            (source / "coremldata.bin").symlink_to(blob)

            def fake_copy_to(**kwargs: object) -> str:
                copy_source = kwargs["source_path"]
                assert isinstance(copy_source, Path)
                copied_file = copy_source / "coremldata.bin"
                self.assertFalse(copied_file.is_symlink())
                self.assertEqual(copied_file.read_bytes(), b"weights")
                return "/private/var/mobile/Containers/Data/Application/UUID/Documents/model.mlmodelc"

            deferred = ios_flutter_smoke._DeferredIosArtifactPush(
                device_id="ios-device",
                bundle_id="com.example.dartMlxFfiProbe",
                source_path=source,
                destination="Documents/model.mlmodelc",
                retries=1,
                runtime_artifact="Documents/model.mlmodelc",
                direct_fallback_on_push_failure=False,
            )
            with mock.patch(
                "ios_flutter_smoke._devicectl_copy_to",
                side_effect=fake_copy_to,
            ):
                deferred.observe_line("flutter: DMF_RUNTIME_SMOKE:BOOT")
        self.assertTrue(deferred.pushed)
        self.assertIsNotNone(deferred.materialized_copy_source)

    def test_deferred_ios_push_records_error_on_copy_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.mlmodelc"
            source.mkdir()
            deferred = ios_flutter_smoke._DeferredIosArtifactPush(
                device_id="ios-device",
                bundle_id="com.example.dartMlxFfiProbe",
                source_path=source,
                destination="Documents/dart_mlx_ffi_runtime_smoke/silero_vad/coreml/model.mlmodelc",
                retries=1,
                runtime_artifact="Documents/dart_mlx_ffi_runtime_smoke/silero_vad/coreml/model.mlmodelc",
                direct_fallback_on_push_failure=False,
            )
            with mock.patch(
                "ios_flutter_smoke._devicectl_copy_to",
                side_effect=RuntimeError("copy failed"),
            ):
                deferred.observe_line("flutter: DMF_RUNTIME_SMOKE:BOOT")
        self.assertFalse(deferred.pushed)
        self.assertTrue(deferred.attempted)
        self.assertIn("copy failed", deferred.error or "")

    def test_devicectl_copy_to_parses_destination_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.txt"
            source.write_text("hello", encoding="utf-8")

            def fake_run(*args, **kwargs):
                command = args[0]
                json_index = command.index("--json-output") + 1
                json_path = Path(command[json_index])
                payload = {
                    "result": {
                        "destination": "file:///private/var/mobile/Containers/Data/Application/UUID/Documents/sample.txt"
                    }
                }
                json_path.write_text(json.dumps(payload), encoding="utf-8")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("ios_flutter_smoke.subprocess.run", side_effect=fake_run):
                destination = ios_flutter_smoke._devicectl_copy_to(
                    device_id="ios-device",
                    bundle_id="com.example.dartMlxFfiProbe",
                    source_path=source,
                    destination="Documents/sample.txt",
                )
        self.assertEqual(
            destination,
            "/private/var/mobile/Containers/Data/Application/UUID/Documents/sample.txt",
        )

    def test_devicectl_copy_to_retries_transient_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.txt"
            source.write_text("hello", encoding="utf-8")
            calls = {"count": 0}

            def fake_run(*args, **kwargs):
                calls["count"] += 1
                command = args[0]
                json_index = command.index("--json-output") + 1
                json_path = Path(command[json_index])
                if calls["count"] == 1:
                    return mock.Mock(
                        returncode=1,
                        stdout="",
                        stderr=(
                            "The file service client failed to write data to the "
                            "network socket because the socket was closed unexpectedly."
                        ),
                    )
                payload = {
                    "result": {
                        "destination": "file:///private/var/mobile/Containers/Data/Application/UUID/Documents/sample.txt"
                    }
                }
                json_path.write_text(json.dumps(payload), encoding="utf-8")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with (
                mock.patch("ios_flutter_smoke.subprocess.run", side_effect=fake_run),
                mock.patch("ios_flutter_smoke.time.sleep"),
            ):
                destination = ios_flutter_smoke._devicectl_copy_to(
                    device_id="ios-device",
                    bundle_id="com.example.dartMlxFfiProbe",
                    source_path=source,
                    destination="Documents/sample.txt",
                    retries=2,
                )
        self.assertEqual(
            destination,
            "/private/var/mobile/Containers/Data/Application/UUID/Documents/sample.txt",
        )
        self.assertEqual(calls["count"], 2)

    def test_classify_run_result_passed(self) -> None:
        status, error = ios_flutter_smoke.classify_run_result(
            marker_payload={"passed": True},
            timed_out=False,
            flutter_exit_code=0,
            captured_lines=[],
        )
        self.assertEqual(status, "passed")
        self.assertIsNone(error)

    def test_classify_run_result_launch_failed_from_flutter_log(self) -> None:
        status, error = ios_flutter_smoke.classify_run_result(
            marker_payload=None,
            timed_out=False,
            flutter_exit_code=1,
            captured_lines=[
                "Installing and launching...",
                "Could not run build/ios/iphoneos/Runner.app on DEVICE.",
                "Error running application on iPhone.",
            ],
        )
        self.assertEqual(status, "launch_failed")
        self.assertEqual(error, "Error running application on iPhone.")

    def test_classify_run_result_runtime_failed_uses_marker_error(self) -> None:
        status, error = ios_flutter_smoke.classify_run_result(
            marker_payload={"passed": False, "error": "Core ML input mismatch"},
            timed_out=False,
            flutter_exit_code=0,
            captured_lines=[],
        )
        self.assertEqual(status, "runtime_failed")
        self.assertEqual(error, "Core ML input mismatch")

    def test_classify_run_result_timeout(self) -> None:
        status, error = ios_flutter_smoke.classify_run_result(
            marker_payload=None,
            timed_out=True,
            flutter_exit_code=None,
            captured_lines=[],
        )
        self.assertEqual(status, "timeout")
        self.assertIn("Timed out", error or "")

    def test_prepare_out_path_uses_fallback_on_parent_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocked = root / "blocked"
            blocked.write_text("not-a-directory", encoding="utf-8")
            requested = blocked / "runtime" / "device_smoke.json"
            fallback = root / "fallback" / "runtime" / "device_smoke.json"

            resolved, meta = ios_flutter_smoke.prepare_out_path(
                out_path=requested,
                fallback_out_path=fallback,
            )

        self.assertEqual(resolved, fallback)
        self.assertEqual(meta["mode"], "fallback")
        self.assertIn("requested_out_path", meta)
        self.assertIn("resolved_out_path", meta)


if __name__ == "__main__":
    unittest.main()
