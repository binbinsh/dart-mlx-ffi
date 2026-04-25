from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

from audit import audit


class AuditBlockerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_audit_blocker_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_prefers_artifact_blocker_reason_over_model_default(self) -> None:
        config_path = self.tmp / "models.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "support_policy": {
                        "production_requires": {
                            "platforms": ["windows"],
                        }
                    },
                    "first_wave": [
                        {
                            "id": "gemma_sea_lion_v4_4b_vl",
                            "family": "Gemma SEA-LION v4 4B VL",
                            "blocked_platforms": {
                                "windows": "No directly loadable ONNX Runtime artifact found on Hugging Face.",
                            },
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        artifacts_path = self.tmp / "artifacts.yaml"
        artifacts_path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "gemma_sea_lion_v4_4b_vl": {
                            "blocked_platforms": {
                                "windows": "Converter exited with code 1. See conversion.log.",
                            }
                        }
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        payload = audit(
            config_path=config_path,
            out_root=self.tmp / "out",
            artifacts_path=artifacts_path,
            model_id="gemma_sea_lion_v4_4b_vl",
        )

        platform = payload["models"][0]["platforms"][0]
        self.assertEqual(platform["state"], "blocked")
        self.assertEqual(
            platform["blocked_reason"],
            "Converter exited with code 1. See conversion.log.",
        )


if __name__ == "__main__":
    unittest.main()
