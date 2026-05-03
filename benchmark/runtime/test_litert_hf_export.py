from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR / "converters"))

import litert_hf_export  # noqa: E402


class LiteRtHfExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_litert_hf_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_prepare_source_returns_model_without_patch(self) -> None:
        source = litert_hf_export._prepare_source(
            model="Qwen/Qwen3.6-27B",
            output_dir=self.tmp / "out",
            revision=None,
            model_type_from=None,
            model_type_to=None,
            endpoint=None,
            fallback_endpoint=None,
            allow_patterns=[],
        )

        self.assertEqual(source, "Qwen/Qwen3.6-27B")

    def test_prepare_source_patches_nested_text_config_for_litert(self) -> None:
        def fake_snapshot_download_with_fallback(**kwargs: object) -> str:
            local_dir = Path(str(kwargs["local_dir"]))
            local_dir.mkdir(parents=True)
            (local_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "text_config": {
                            "hidden_size": 5120,
                            "eos_token_id": 248044,
                            "pad_token_id": None,
                            "rope_parameters": {
                                "mrope_section": [11, 11, 10],
                                "partial_rotary_factor": 0.25,
                                "rope_theta": 10000000,
                                "rope_type": "default",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(local_dir)

        with mock.patch(
            "hf_download.snapshot_download",
            side_effect=fake_snapshot_download_with_fallback,
        ):
            source = litert_hf_export._prepare_source(
                model="Qwen/Qwen3.6-27B",
                output_dir=self.tmp / "out",
                revision=None,
                model_type_from="qwen3_5",
                model_type_to="qwen3",
                endpoint="https://hf-mirror.com",
                fallback_endpoint=None,
                allow_patterns=["config.json", " "],
            )

        config = json.loads((Path(source) / "config.json").read_text(encoding="utf-8"))
        report = json.loads(
            (self.tmp / "out" / "source_patch_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["model_type"], "qwen3")
        self.assertEqual(config["hidden_size"], 5120)
        self.assertEqual(config["pad_token_id"], 248044)
        self.assertNotIn("pad_token_id", config["text_config"])
        self.assertEqual(config["rope_scaling"]["mrope_section"], [11, 11, 10])
        self.assertNotIn("type", config["rope_scaling"])
        self.assertEqual(report["dropped_null_text_config_keys"], ["pad_token_id"])
        self.assertEqual(report["patched_model_type"], "qwen3")

    def test_patch_minicpm_utils_rewrites_optional_import(self) -> None:
        utils_path = self.tmp / "utils.py"
        utils_path.write_text(
            "def load_video():\n"
            "    from minicpmo.utils import get_video_frame_audio_segments\n"
            "    return get_video_frame_audio_segments\n",
            encoding="utf-8",
        )

        patched = litert_hf_export._patch_minicpm_utils_file(utils_path)

        text = utils_path.read_text(encoding="utf-8")
        self.assertTrue(patched)
        self.assertNotIn("from minicpmo.utils import", text)
        self.assertIn("importlib.import_module", text)

    def test_patch_minicpm_optional_deps_records_local_patch(self) -> None:
        source = self.tmp / "source"
        source.mkdir()
        (source / "utils.py").write_text(
            "def load_video():\n"
            "    from minicpmo.utils import get_video_frame_audio_segments\n"
            "    return get_video_frame_audio_segments\n",
            encoding="utf-8",
        )

        report = litert_hf_export._patch_minicpm_optional_deps(
            model=source,
            repo_id="openbmb/MiniCPM-o-4_5",
            output_dir=self.tmp / "out",
            revision=None,
            endpoint=None,
            fallback_endpoint=None,
        )

        self.assertEqual(
            report["patched_minicpm_optional_dep_files"],
            [str(source / "utils.py")],
        )
        report_path = self.tmp / "out" / "minicpm_optional_deps_patch_report.json"
        self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
