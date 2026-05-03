from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CONVERTER_DIR = Path(__file__).resolve().parent / "converters"
import sys

sys.path.insert(0, str(CONVERTER_DIR))

import ming_omni_source  # noqa: E402


class MingOmniSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_ming_source_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_prepare_patched_source_pulls_missing_auto_map_modules(self) -> None:
        fallback = self.tmp / "fallback"
        fallback.mkdir(parents=True, exist_ok=True)
        (fallback / "configuration_bailingmm.py").write_text(
            "from configuration_audio import GLMAudioConfig\n",
            encoding="utf-8",
        )
        (fallback / "configuration_audio.py").write_text(
            "class GLMAudioConfig:\n    pass\n",
            encoding="utf-8",
        )

        def fake_snapshot_download(**kwargs: object) -> str:
            source_dir = Path(str(kwargs["local_dir"]))
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "config.json").write_text(
                json.dumps(
                    {
                        "auto_map": {
                            "AutoConfig": "configuration_bailingmm.BailingMMConfig",
                        }
                    }
                ),
                encoding="utf-8",
            )
            return str(source_dir)

        def fake_download_first_available(
            *,
            repositories: list[str],
            filename: str,
            endpoint: str | None = None,
            fallback_endpoint: str | None = None,
        ) -> str | None:
            del endpoint, fallback_endpoint
            candidate = fallback / filename
            return str(candidate) if candidate.exists() else None

        with (
            mock.patch(
                "ming_omni_source.snapshot_download_with_fallback",
                side_effect=fake_snapshot_download,
            ),
            mock.patch(
                "ming_omni_source._download_first_available",
                side_effect=fake_download_first_available,
            ),
        ):
            source_dir, report = ming_omni_source.prepare_patched_source(
                source_model="inclusionAI/Ming-omni-tts-0.5B",
                work_dir=self.tmp / "work",
            )

        self.assertTrue((source_dir / "configuration_bailingmm.py").exists())
        self.assertTrue((source_dir / "configuration_audio.py").exists())
        self.assertEqual(
            report["patched_files"],
            ["configuration_audio.py", "configuration_bailingmm.py"],
        )
        self.assertEqual(report["missing_files"], [])


if __name__ == "__main__":
    unittest.main()
