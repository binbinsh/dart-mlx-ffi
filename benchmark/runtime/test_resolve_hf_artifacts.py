from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RUNTIME_DIR = Path(__file__).resolve().parent
ROOT = RUNTIME_DIR.parents[1]

import sys

sys.path.insert(0, str(RUNTIME_DIR))

from resolve_hf_artifacts import HuggingFaceArtifactResolver  # noqa: E402


class ResolveHfArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_resolve_hf_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_download_passes_endpoint_fallbacks(self) -> None:
        snapshot = self.tmp / "snapshot"
        (snapshot / "onnx").mkdir(parents=True)
        (snapshot / "onnx" / "model.onnx").write_bytes(b"onnx")
        calls: list[dict[str, object]] = []

        def fake_snapshot_download_with_fallback(**kwargs: object) -> str:
            calls.append(kwargs)
            return str(snapshot)

        resolver = HuggingFaceArtifactResolver(
            catalog={},
            catalog_path=ROOT / "benchmark/runtime/hf_artifacts.yaml",
            out_path=ROOT / "benchmark/runtime/artifacts.local.yaml",
            cache_root=self.tmp / "cache",
            model_filter=set(),
            platform_filter=set(),
            engine_filter=set(),
            local_files_only=False,
            allow_missing=False,
            endpoint="https://hf-mirror.com",
            fallback_endpoint="https://huggingface.co",
        )

        with mock.patch(
            "resolve_hf_artifacts.snapshot_download_with_fallback",
            side_effect=fake_snapshot_download_with_fallback,
        ):
            resolved = resolver._download(
                {"repo": "owner/model", "artifact": "onnx/model.onnx"}
            )

        self.assertEqual(resolved, snapshot / "onnx" / "model.onnx")
        self.assertEqual(calls[0]["endpoint"], "https://hf-mirror.com")
        self.assertEqual(calls[0]["fallback_endpoint"], "https://huggingface.co")


if __name__ == "__main__":
    unittest.main()
