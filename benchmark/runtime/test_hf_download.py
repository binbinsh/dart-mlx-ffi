from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR / "converters"))

import hf_download  # noqa: E402


class HfDownloadTest(unittest.TestCase):
    def test_endpoint_list_deduplicates_primary_and_fallbacks(self) -> None:
        self.assertEqual(
            hf_download._download_endpoints(
                "https://hf-mirror.com/",
                "https://hf-mirror.com, https://hf-mirror.com",
            ),
            ["https://hf-mirror.com"],
        )

    def test_snapshot_download_retries_fallback_endpoint(self) -> None:
        calls: list[str | None] = []

        def fake_snapshot_download(*, endpoint: str | None, **kwargs: object) -> str:
            del kwargs
            calls.append(endpoint)
            if endpoint is None:
                raise RuntimeError("429 Too Many Requests")
            return "/tmp/snapshot"

        with mock.patch.object(
            hf_download,
            "snapshot_download",
            side_effect=fake_snapshot_download,
        ), mock.patch.dict("os.environ", {"HF_ENDPOINT": ""}, clear=False):
            path = hf_download.snapshot_download_with_fallback(
                repo_id="owner/model",
                fallback_endpoint="https://hf-mirror.com",
                attempts_per_endpoint=1,
            )

        self.assertEqual(path, "/tmp/snapshot")
        self.assertEqual(calls, [None, "https://hf-mirror.com"])

    def test_hf_hub_download_reports_all_failed_endpoints(self) -> None:
        with mock.patch.object(
            hf_download,
            "hf_hub_download",
            side_effect=RuntimeError("download failed"),
        ), mock.patch.object(hf_download.time, "sleep"), mock.patch.dict(
            "os.environ",
            {"HF_ENDPOINT": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "hf-mirror.com"):
                hf_download.hf_hub_download_with_fallback(
                    repo_id="owner/model",
                    filename="model.onnx",
                    fallback_endpoint="https://hf-mirror.com",
                    attempts_per_endpoint=1,
                )

    def test_hf_hub_download_retries_endpoint_before_fallback(self) -> None:
        calls: list[str | None] = []

        def fake_hf_hub_download(*, endpoint: str | None, **kwargs: object) -> str:
            del kwargs
            calls.append(endpoint)
            if len(calls) < 2:
                raise RuntimeError("connection reset")
            return "/tmp/model.onnx"

        with mock.patch.object(
            hf_download,
            "hf_hub_download",
            side_effect=fake_hf_hub_download,
        ), mock.patch.object(hf_download.time, "sleep"), mock.patch.dict(
            "os.environ",
            {"HF_ENDPOINT": ""},
            clear=False,
        ):
            path = hf_download.hf_hub_download_with_fallback(
                repo_id="owner/model",
                filename="model.onnx",
                fallback_endpoint="https://hf-mirror.com",
                attempts_per_endpoint=2,
            )

        self.assertEqual(path, "/tmp/model.onnx")
        self.assertEqual(calls, [None, None])

    def test_retry_count_can_be_set_explicitly(self) -> None:
        with mock.patch.object(
            hf_download,
            "hf_hub_download",
            side_effect=RuntimeError("download failed"),
        ), mock.patch.dict("os.environ", {"HF_ENDPOINT": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, r"default\[1/2\]"):
                hf_download.hf_hub_download_with_fallback(
                    repo_id="owner/model",
                    filename="model.onnx",
                    fallback_endpoint="",
                    retry_backoff_seconds=0,
                    attempts_per_endpoint=2,
                )

    def test_default_cache_uses_repo_local_hf_home_when_env_is_missing(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_hf_hub_download(*, endpoint: str | None, **kwargs: object) -> str:
            del endpoint
            calls.append(dict(kwargs))
            return "/tmp/model.onnx"

        with mock.patch.object(
            hf_download,
            "hf_hub_download",
            side_effect=fake_hf_hub_download,
        ), mock.patch.dict(
            "os.environ",
            {
                "HF_ENDPOINT": "",
                "HF_HOME": "",
                "HUGGINGFACE_HUB_CACHE": "",
            },
            clear=False,
        ):
            hf_download.hf_hub_download_with_fallback(
                repo_id="owner/model",
                filename="model.onnx",
                fallback_endpoint="",
                attempts_per_endpoint=1,
            )

        cache_dir = Path(str(calls[0]["cache_dir"]))
        self.assertEqual(cache_dir.name, "hub")
        self.assertEqual(cache_dir.parent.name, ".hf_home")


if __name__ == "__main__":
    unittest.main()
