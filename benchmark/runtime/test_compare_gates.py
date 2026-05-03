from __future__ import annotations

import sys
import unittest
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR))

from compare import compare_report


class ComparePromotionGateTest(unittest.TestCase):
    def test_requires_matching_input_signature_when_enabled(self) -> None:
        report = _report(task="tensor")
        report["baseline"]["input_signature"] = {"digest": "a"}
        report["candidate"]["input_signature"] = {"digest": "b"}

        verdict = compare_report(report, {**_thresholds(), "require_input_signature": True})

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["identity"])
        self.assertIn("input_signature_match", failed)

    def test_requires_input_signature_presence_when_enabled(self) -> None:
        report = _report(task="tensor")

        verdict = compare_report(report, {**_thresholds(), "require_input_signature": True})

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["identity"])
        self.assertIn("input_signature_present", failed)

    def test_requires_matching_run_config_when_enabled(self) -> None:
        report = _report(task="tensor")
        report["baseline"]["run_config"] = _run_config(task="tensor", iters=5)
        report["candidate"]["run_config"] = _run_config(task="tensor", iters=3)

        verdict = compare_report(report, {**_thresholds(), "require_run_config": True})

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["identity"])
        self.assertIn("run_config_match", failed)

    def test_requires_run_config_presence_when_enabled(self) -> None:
        report = _report(task="tensor")

        verdict = compare_report(report, {**_thresholds(), "require_run_config": True})

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["identity"])
        self.assertIn("run_config_present", failed)

    def test_requires_iteration_count_presence_when_run_config_is_required(self) -> None:
        report = _report(task="tensor")
        report["baseline"]["run_config"] = _run_config(task="tensor", iters=5)
        report["candidate"]["run_config"] = _run_config(task="tensor", iters=5)

        verdict = compare_report(report, {**_thresholds(), "require_run_config": True})

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["speed"])
        self.assertIn("metrics_iteration_count_present", failed)

    def test_rejects_iteration_count_mismatch(self) -> None:
        report = _report(task="tensor")
        report["baseline"]["run_config"] = _run_config(task="tensor", iters=5)
        report["candidate"]["run_config"] = _run_config(task="tensor", iters=5)
        report["baseline"]["metrics"]["iteration_count"] = 5
        report["candidate"]["metrics"]["iteration_count"] = 4

        verdict = compare_report(report, {**_thresholds(), "require_run_config": True})

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["speed"])
        self.assertIn("candidate_iteration_count", failed)

    def test_accepts_matching_iteration_counts(self) -> None:
        report = _report(task="tensor")
        report["baseline"]["run_config"] = _run_config(task="tensor", iters=5)
        report["candidate"]["run_config"] = _run_config(task="tensor", iters=5)
        report["baseline"]["metrics"]["iteration_count"] = 5
        report["candidate"]["metrics"]["iteration_count"] = 5

        verdict = compare_report(report, {**_thresholds(), "require_run_config": True})

        self.assertTrue(verdict["passed"])
        checks = {item["name"] for item in verdict["speed"]["checks"]}
        self.assertIn("baseline_iteration_count", checks)
        self.assertIn("candidate_iteration_count", checks)

    def test_rejects_latency_sample_count_mismatch(self) -> None:
        report = _report(task="tensor")
        report["candidate"]["metrics"]["iteration_count"] = 3
        report["candidate"]["metrics"]["latency_ms"] = {
            "values": [9.0, 11.0],
            "mean": 10.0,
        }

        verdict = compare_report(report, _thresholds())

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["speed"])
        self.assertIn("candidate_latency_sample_count", failed)

    def test_rejects_latency_mean_mismatch(self) -> None:
        report = _report(task="tensor")
        report["candidate"]["metrics"]["iteration_count"] = 2
        report["candidate"]["metrics"]["latency_ms"] = {
            "values": [9.0, 11.0],
            "mean": 9.0,
        }

        verdict = compare_report(report, _thresholds())

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["speed"])
        self.assertIn("candidate_latency_sample_mean", failed)

    def test_text_requires_task_specific_correctness_evidence(self) -> None:
        report = _report(task="text")
        report["baseline"]["correctness"] = {"output_text": "hello"}
        report["candidate"]["correctness"] = {"output_text": "hello"}

        verdict = compare_report(report, _thresholds())

        self.assertTrue(verdict["passed"])
        checks = {item["name"] for item in verdict["correctness"]["checks"]}
        self.assertIn("task_text_evidence", checks)

    def test_embedding_requires_cosine_and_l2_evidence(self) -> None:
        report = _report(task="embedding")
        report["baseline"]["correctness"] = {"max_abs_diff": 0.0}
        report["candidate"]["correctness"] = {"max_abs_diff": 0.0}

        verdict = compare_report(report, _thresholds())

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["correctness"])
        self.assertIn("task_embedding_evidence", failed)

    def test_report_identity_mismatch_blocks_promotion(self) -> None:
        report = _report(task="tensor")
        report["mismatches"] = [
            {"field": "platform", "baseline": "ios", "candidate": "android"}
        ]

        verdict = compare_report(report, _thresholds())

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["identity"])
        self.assertIn("report_identity_mismatches", failed)

    def test_text_accepts_top_k_output_summary_evidence(self) -> None:
        report = _report(task="text")
        report["baseline"]["correctness"] = {
            "output_summaries": {
                "logits": _summary(top_k=[7, 4, 2]),
            }
        }
        report["candidate"]["correctness"] = {
            "output_summaries": {
                "logits": _summary(top_k=[7, 8, 9]),
            }
        }

        verdict = compare_report(report, _thresholds())

        self.assertTrue(verdict["passed"])
        checks = {item["name"] for item in verdict["correctness"]["checks"]}
        self.assertIn("output_summary_top1_match_ratio", checks)
        self.assertIn("output_summary_top3_coverage_ratio", checks)

    def test_text_fails_when_baseline_top1_is_not_in_candidate_top3(self) -> None:
        report = _report(task="text")
        report["baseline"]["correctness"] = {
            "output_summaries": {
                "logits": _summary(top_k=[7, 4, 2]),
            }
        }
        report["candidate"]["correctness"] = {
            "output_summaries": {
                "logits": _summary(top_k=[1, 8, 9]),
            }
        }

        verdict = compare_report(report, _thresholds())

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["correctness"])
        self.assertIn("output_summary_top3_coverage_ratio", failed)

    def test_tensor_output_values_reject_nan(self) -> None:
        report = _report(task="tensor")
        report["candidate"]["correctness"]["output_values"]["output"]["values"] = [
            float("nan")
        ]

        verdict = compare_report(report, _thresholds())

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["correctness"])
        self.assertIn("output_values_finite", failed)

    def test_embedding_output_values_reject_infinity(self) -> None:
        report = _report(task="embedding")
        report["baseline"]["correctness"] = {
            "output_values": {
                "embedding": {
                    "dtype": "float32",
                    "shape": [2],
                    "values": [1.0, 0.0],
                }
            }
        }
        report["candidate"]["correctness"] = {
            "output_values": {
                "embedding": {
                    "dtype": "float32",
                    "shape": [2],
                    "values": [float("inf"), 0.0],
                }
            }
        }

        verdict = compare_report(report, _thresholds())

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["correctness"])
        self.assertIn("embedding_values_finite", failed)

    def test_end_to_end_ratio_uses_own_threshold(self) -> None:
        report = _report(task="tensor")
        report["baseline"]["metrics"]["end_to_end_ms"] = 100.0
        report["candidate"]["metrics"]["end_to_end_ms"] = 115.0

        verdict = compare_report(
            report,
            {**_thresholds(), "max_ttft_ratio": 2.0, "max_end_to_end_ratio": 1.10},
        )

        self.assertFalse(verdict["passed"])
        failed = _failed(verdict["speed"])
        self.assertIn("end_to_end_ratio", failed)


def _report(*, task: str) -> dict:
    values = {
        "output": {
            "dtype": "float32",
            "shape": [1],
            "values": [1.0],
        }
    }
    return {
        "model_id": "demo",
        "platform": "linux",
        "baseline": {
            "model_id": "demo",
            "platform": "linux",
            "task": task,
            "engine": "onnx",
            "correctness": {"output_values": values},
            "metrics": {
                "end_to_end_ms": 10.0,
                "peak_memory_bytes": 100,
            },
        },
        "candidate": {
            "model_id": "demo",
            "platform": "linux",
            "task": task,
            "engine": "onnx",
            "correctness": {"output_values": values},
            "metrics": {
                "end_to_end_ms": 10.0,
                "peak_memory_bytes": 100,
            },
        },
        "mismatches": [],
    }


def _thresholds() -> dict:
    return {
        "min_speed_ratio": 0.90,
        "max_ttft_ratio": 1.10,
        "max_end_to_end_ratio": 1.10,
        "max_peak_memory_ratio": 1.10,
        "min_embedding_cosine": 0.999,
        "max_embedding_l2": 0.001,
        "max_abs_diff": 0.0001,
        "require_device_profile": False,
        "required_coreml_preferred_device": None,
        "required_provider": None,
        "required_delegate": None,
        "require_input_signature": False,
        "require_run_config": False,
    }


def _run_config(*, task: str, iters: int) -> dict:
    return {
        "format": "dart_mlx_ffi.run_config.v1",
        "task": task,
        "warmup": 1,
        "iters": iters,
        "max_tokens": 64,
        "sampling_strategy": "greedy",
    }


def _summary(*, top_k: list[int]) -> dict:
    return {
        "dtype": "float32",
        "shape": [1, 32000],
        "top_k": [
            {"index": index, "value": float(len(top_k) - rank)}
            for rank, index in enumerate(top_k)
        ],
    }


def _failed(section: dict) -> set[str]:
    return {
        str(item["name"])
        for item in section["checks"]
        if isinstance(item, dict) and not item["passed"]
    }


if __name__ == "__main__":
    unittest.main()
