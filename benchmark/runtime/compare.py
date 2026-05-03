from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "min_speed_ratio": 0.90,
    "max_ttft_ratio": 1.10,
    "max_end_to_end_ratio": 1.10,
    "max_peak_memory_ratio": 1.10,
    "min_embedding_cosine": 0.999,
    "max_embedding_l2": 0.001,
    "max_abs_diff": 1e-4,
    "require_device_profile": False,
    "required_coreml_preferred_device": None,
    "required_provider": None,
    "required_delegate": None,
    "require_input_signature": False,
    "require_run_config": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare runtime candidate and baseline reports."
    )
    parser.add_argument("report", type=Path, help="JSON report to compare")
    parser.add_argument("--out", type=Path, help="Optional output JSON path")
    parser.add_argument("--min-speed-ratio", type=float)
    parser.add_argument("--max-ttft-ratio", type=float)
    parser.add_argument("--max-end-to-end-ratio", type=float)
    parser.add_argument("--max-peak-memory-ratio", type=float)
    parser.add_argument("--min-embedding-cosine", type=float)
    parser.add_argument("--max-embedding-l2", type=float)
    parser.add_argument("--max-abs-diff", type=float)
    parser.add_argument("--require-device-profile", action="store_true")
    parser.add_argument("--required-coreml-preferred-device")
    parser.add_argument("--required-provider")
    parser.add_argument("--required-delegate")
    parser.add_argument("--require-input-signature", action="store_true")
    parser.add_argument("--require-run-config", action="store_true")
    args = parser.parse_args()

    thresholds = dict(DEFAULT_THRESHOLDS)
    for name in thresholds:
        value = getattr(args, name.replace("-", "_"), None)
        if value is not None:
            thresholds[name] = value

    report = json.loads(args.report.read_text(encoding="utf-8"))
    verdict = compare_report(report, thresholds)
    payload = {**report, "thresholds": thresholds, "verdict": verdict}

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if not verdict["passed"]:
        raise SystemExit(1)


def compare_report(
    report: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    baseline = report.get("baseline") or {}
    candidate = report.get("candidate") or {}
    identity = compare_identity(report, thresholds)
    task = _report_task(report, baseline, candidate)
    correctness = compare_correctness(
        baseline.get("correctness") or {},
        candidate.get("correctness") or {},
        thresholds,
        task=task,
    )
    speed = compare_speed(
        baseline.get("metrics") or {},
        candidate.get("metrics") or {},
        thresholds,
        baseline_run_config=_run_config(baseline),
        candidate_run_config=_run_config(candidate),
    )
    peak_memory = compare_peak_memory(
        baseline.get("metrics") or {},
        candidate.get("metrics") or {},
        thresholds,
    )
    device_profile = compare_device_profile(candidate, thresholds)
    return {
        "passed": correctness["passed"]
        and identity["passed"]
        and speed["passed"]
        and peak_memory["passed"]
        and device_profile["passed"],
        "identity": identity,
        "correctness": correctness,
        "speed": speed,
        "peak_memory": peak_memory,
        "device_profile": device_profile,
    }


def compare_identity(
    report: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    baseline = report.get("baseline") or {}
    candidate = report.get("candidate") or {}
    checks: list[dict[str, Any]] = []
    mismatches = report.get("mismatches") or []
    checks.append(
        _check(
            "report_identity_mismatches",
            not mismatches,
            value=mismatches,
        )
    )
    for field in ("model_id", "platform", "task"):
        left = baseline.get(field)
        right = candidate.get(field)
        if left is None and right is None:
            continue
        checks.append(
            _check(
                f"{field}_match",
                left == right,
                expected=left,
                actual=right,
            )
        )

    baseline_signature = _input_signature(baseline)
    candidate_signature = _input_signature(candidate)
    require_input = bool(thresholds.get("require_input_signature"))
    if baseline_signature is not None or candidate_signature is not None or require_input:
        checks.append(
            _check(
                "input_signature_present",
                baseline_signature is not None and candidate_signature is not None,
                expected=baseline_signature,
                actual=candidate_signature,
            )
        )
        checks.append(
            _check(
                "input_signature_match",
                baseline_signature is not None
                and baseline_signature == candidate_signature,
                expected=baseline_signature,
                actual=candidate_signature,
            )
        )
    baseline_run_config = _run_config(baseline)
    candidate_run_config = _run_config(candidate)
    require_run_config = bool(thresholds.get("require_run_config"))
    if baseline_run_config is not None or candidate_run_config is not None or require_run_config:
        checks.append(
            _check(
                "run_config_present",
                baseline_run_config is not None and candidate_run_config is not None,
                expected=baseline_run_config,
                actual=candidate_run_config,
            )
        )
        checks.append(
            _check(
                "run_config_match",
                baseline_run_config is not None
                and baseline_run_config == candidate_run_config,
                expected=baseline_run_config,
                actual=candidate_run_config,
            )
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def compare_correctness(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: dict[str, float],
    *,
    task: str | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    if "token_ids" in baseline or "token_ids" in candidate:
        checks.append(
            _check(
                "token_ids_exact",
                baseline.get("token_ids") == candidate.get("token_ids"),
                expected="same greedy token ids",
            )
        )

    if "output_text" in baseline or "output_text" in candidate:
        checks.append(
            _check(
                "output_text_exact",
                baseline.get("output_text") == candidate.get("output_text"),
                expected="same greedy decoded text",
            )
        )

    if "top1_match_ratio" in candidate:
        checks.append(
            _check(
                "top1_match_ratio",
                float(candidate["top1_match_ratio"]) >= 0.95,
                value=candidate["top1_match_ratio"],
            )
        )

    if "top3_coverage_ratio" in candidate:
        checks.append(
            _check(
                "top3_coverage_ratio",
                float(candidate["top3_coverage_ratio"]) >= 1.0,
                value=candidate["top3_coverage_ratio"],
            )
        )

    if "embedding_cosine" in candidate:
        checks.append(
            _check(
                "embedding_cosine",
                float(candidate["embedding_cosine"])
                >= thresholds["min_embedding_cosine"],
                value=candidate["embedding_cosine"],
            )
        )

    if "embedding_l2" in candidate:
        checks.append(
            _check(
                "embedding_l2",
                float(candidate["embedding_l2"]) <= thresholds["max_embedding_l2"],
                value=candidate["embedding_l2"],
            )
        )

    if "function_call" in baseline or "function_call" in candidate:
        checks.append(
            _check(
                "function_call_exact",
                baseline.get("function_call") == candidate.get("function_call"),
                expected="same function name and required arguments",
            )
        )

    if "max_abs_diff" in candidate:
        checks.append(
            _check(
                "max_abs_diff",
                float(candidate["max_abs_diff"]) <= thresholds["max_abs_diff"],
                value=candidate["max_abs_diff"],
            )
        )

    if "output_values" in baseline or "output_values" in candidate:
        baseline_values = baseline.get("output_values") or {}
        candidate_values = candidate.get("output_values") or {}
        if _is_embedding_output(baseline, candidate):
            checks.extend(
                compare_embedding_output_values(
                    baseline_values,
                    candidate_values,
                    thresholds,
                )
            )
        else:
            checks.append(
                compare_output_values(
                    baseline_values,
                    candidate_values,
                    thresholds,
                )
            )

    if "output_summaries" in baseline or "output_summaries" in candidate:
        checks.extend(
            compare_output_summaries(
                baseline.get("output_summaries") or {},
                candidate.get("output_summaries") or {},
            )
        )

    checks.extend(_task_evidence_checks(task, checks))

    if not checks:
        checks.append(_check("correctness_evidence_present", False))

    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _task_evidence_checks(
    task: str | None,
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not task:
        return []
    names = {str(check.get("name")) for check in checks}
    if task == "embedding":
        return [
            _check(
                "task_embedding_evidence",
                {"embedding_cosine", "embedding_l2"}.issubset(names),
                expected=["embedding_cosine", "embedding_l2"],
                actual=sorted(names),
            )
        ]
    if task == "function":
        return [
            _check(
                "task_function_evidence",
                "function_call_exact" in names,
                expected="function_call_exact",
                actual=sorted(names),
            )
        ]
    if task in {"text", "vlm"}:
        exact = bool({"token_ids_exact", "output_text_exact"}.intersection(names))
        ranked = bool(
            {"top1_match_ratio", "top3_coverage_ratio"}.issubset(names)
            or {
                "output_summary_top1_match_ratio",
                "output_summary_top3_coverage_ratio",
            }.issubset(names)
        )
        tensor = bool({"output_values", "max_abs_diff"}.intersection(names))
        return [
            _check(
                f"task_{task}_evidence",
                exact or ranked or tensor,
                expected=(
                    "token_ids/output_text exact, top1+top3 coverage, "
                    "or tensor output comparison"
                ),
                actual=sorted(names),
            )
        ]
    if task in {"audio", "tts", "vad", "tensor"}:
        return [
            _check(
                f"task_{task}_evidence",
                bool({"output_values", "max_abs_diff"}.intersection(names)),
                expected="tensor output comparison",
                actual=sorted(names),
            )
        ]
    return []


def compare_output_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "output_summary_names",
            set(baseline) == set(candidate) and bool(baseline),
            expected=sorted(baseline),
            actual=sorted(candidate),
        )
    )
    if set(baseline) != set(candidate) or not baseline:
        return checks

    top1_total = 0
    top1_matches = 0
    top3_total = 0
    top3_matches = 0
    for name, expected in baseline.items():
        actual = candidate.get(name) or {}
        checks.append(
            _check(
                "output_summary_dtype",
                expected.get("dtype") == actual.get("dtype"),
                output=name,
                expected=expected.get("dtype"),
                actual=actual.get("dtype"),
            )
        )
        checks.append(
            _check(
                "output_summary_shape",
                expected.get("shape") == actual.get("shape"),
                output=name,
                expected=expected.get("shape"),
                actual=actual.get("shape"),
            )
        )
        expected_top = _top_k_indices(expected, limit=3)
        actual_top = _top_k_indices(actual, limit=3)
        checks.append(
            _check(
                "output_summary_top_k_present",
                bool(expected_top) and bool(actual_top),
                output=name,
                expected=expected_top,
                actual=actual_top,
            )
        )
        if not expected_top or not actual_top:
            continue
        top1_total += 1
        top3_total += 1
        if expected_top[0] == actual_top[0]:
            top1_matches += 1
        if expected_top[0] in set(actual_top[:3]):
            top3_matches += 1

    if top1_total:
        top1_ratio = top1_matches / top1_total
        checks.append(
            _check(
                "output_summary_top1_match_ratio",
                top1_ratio >= 0.95,
                value=top1_ratio,
            )
        )
    if top3_total:
        top3_ratio = top3_matches / top3_total
        checks.append(
            _check(
                "output_summary_top3_coverage_ratio",
                top3_ratio >= 1.0,
                value=top3_ratio,
            )
        )
    return checks


def compare_output_values(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        return _check(
            "output_values",
            False,
            expected=sorted(baseline),
            actual=sorted(candidate),
        )
    max_abs_diff = 0.0
    for name, expected in baseline.items():
        actual = candidate[name]
        if expected.get("dtype") != actual.get("dtype"):
            return _check(
                "output_values",
                False,
                output=name,
                expected=expected.get("dtype"),
                actual=actual.get("dtype"),
            )
        if expected.get("shape") != actual.get("shape"):
            return _check(
                "output_values",
                False,
                output=name,
                expected=expected.get("shape"),
                actual=actual.get("shape"),
            )
        expected_values = expected.get("values") or []
        actual_values = actual.get("values") or []
        if len(expected_values) != len(actual_values):
            return _check(
                "output_values",
                False,
                output=name,
                expected=len(expected_values),
                actual=len(actual_values),
            )
        for left, right in zip(expected_values, actual_values):
            left_float = float(left)
            right_float = float(right)
            if not math.isfinite(left_float) or not math.isfinite(right_float):
                return _check(
                    "output_values_finite",
                    False,
                    output=name,
                    expected=left,
                    actual=right,
                )
            max_abs_diff = max(max_abs_diff, abs(left_float - right_float))
    return _check(
        "output_values",
        max_abs_diff <= thresholds["max_abs_diff"],
        value=max_abs_diff,
    )


def compare_embedding_output_values(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    expected = _embedding_tensor(baseline)
    actual = _embedding_tensor(candidate)
    checks = [
        _check(
            "embedding_output_present",
            expected is not None and actual is not None,
            expected=sorted(baseline),
            actual=sorted(candidate),
        )
    ]
    if expected is None or actual is None:
        return checks

    expected_name, expected_tensor = expected
    actual_name, actual_tensor = actual
    checks.append(
        _check(
            "embedding_dtype_numeric",
            _is_float_dtype(expected_tensor.get("dtype"))
            and _is_float_dtype(actual_tensor.get("dtype")),
            expected=expected_tensor.get("dtype"),
            actual=actual_tensor.get("dtype"),
        )
    )
    checks.append(
        _check(
            "embedding_shape",
            expected_tensor.get("shape") == actual_tensor.get("shape"),
            expected=expected_tensor.get("shape"),
            actual=actual_tensor.get("shape"),
        )
    )

    expected_values = _flatten_numbers(expected_tensor.get("values") or [])
    actual_values = _flatten_numbers(actual_tensor.get("values") or [])
    checks.append(
        _check(
            "embedding_value_count",
            len(expected_values) == len(actual_values) and len(expected_values) > 0,
            expected=len(expected_values),
            actual=len(actual_values),
        )
    )
    if not all(check["passed"] for check in checks):
        return checks
    checks.append(
        _check(
            "embedding_values_finite",
            all(math.isfinite(value) for value in expected_values)
            and all(math.isfinite(value) for value in actual_values),
        )
    )
    if not checks[-1]["passed"]:
        return checks

    expected_norm = math.sqrt(sum(value * value for value in expected_values))
    actual_norm = math.sqrt(sum(value * value for value in actual_values))
    dot = sum(left * right for left, right in zip(expected_values, actual_values))
    cosine = dot / (expected_norm * actual_norm) if expected_norm and actual_norm else 0.0
    l2 = math.sqrt(
        sum(
            (left - right) * (left - right)
            for left, right in zip(expected_values, actual_values)
        )
    )
    checks.append(
        _check(
            "embedding_cosine",
            cosine >= thresholds["min_embedding_cosine"],
            value=cosine,
            expected_output=expected_name,
            actual_output=actual_name,
        )
    )
    checks.append(
        _check(
            "embedding_l2",
            l2 <= thresholds["max_embedding_l2"],
            value=l2,
            expected_output=expected_name,
            actual_output=actual_name,
        )
    )
    return checks


def _is_embedding_output(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    baseline_values = baseline.get("output_values") or {}
    candidate_values = candidate.get("output_values") or {}
    return (
        "embedding_dim" in baseline
        or "embedding_dim" in candidate
        or "embedding" in baseline_values
        or "embedding" in candidate_values
    )


def _report_task(
    report: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> str | None:
    for source in (report, candidate, baseline):
        task = source.get("task")
        if isinstance(task, str) and task:
            return task
    return None


def _input_signature(report: dict[str, Any]) -> Any:
    signature = report.get("input_signature")
    if signature is not None:
        return signature
    digest = report.get("input_digest")
    if digest is not None:
        return {"digest": digest}
    return None


def _run_config(report: dict[str, Any]) -> Any:
    config = report.get("run_config")
    return config if config is not None else None


def _embedding_tensor(values: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if not values:
        return None
    if isinstance(values.get("embedding"), dict):
        return "embedding", values["embedding"]
    if len(values) == 1:
        name, tensor = next(iter(values.items()))
        if isinstance(tensor, dict):
            return name, tensor
    return None


def _top_k_indices(summary: dict[str, Any], *, limit: int) -> list[int]:
    raw = summary.get("top_k")
    if not isinstance(raw, list):
        return []
    indices: list[int] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if index is None:
            continue
        indices.append(int(index))
    return indices


def _is_float_dtype(value: Any) -> bool:
    return str(value).lower() in {"float16", "float32", "float64"}


def _flatten_numbers(values: Any) -> list[float]:
    if isinstance(values, list):
        flattened: list[float] = []
        for value in values:
            flattened.extend(_flatten_numbers(value))
        return flattened
    return [float(values)]


def compare_speed(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: dict[str, float],
    *,
    baseline_run_config: dict[str, Any] | None = None,
    candidate_run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.extend(
        _iteration_count_checks(
            baseline=baseline,
            candidate=candidate,
            baseline_run_config=baseline_run_config,
            candidate_run_config=candidate_run_config,
            require=bool(thresholds.get("require_run_config")),
        )
    )
    checks.extend(_latency_sample_checks("baseline", baseline))
    checks.extend(_latency_sample_checks("candidate", candidate))
    for name in ("prefill_tokens_per_second", "decode_tokens_per_second"):
        ratio = _throughput_ratio(candidate.get(name), baseline.get(name))
        if ratio is not None:
            checks.append(
                _check(
                    f"{name}_ratio",
                    ratio >= thresholds["min_speed_ratio"],
                    value=ratio,
                )
            )

    ttft_ratio = _latency_ratio(candidate.get("ttft_ms"), baseline.get("ttft_ms"))
    if ttft_ratio is not None:
        checks.append(
            _check(
                "ttft_ratio",
                ttft_ratio <= thresholds["max_ttft_ratio"],
                value=ttft_ratio,
            )
        )

    e2e_ratio = _latency_ratio(
        candidate.get("end_to_end_ms"),
        baseline.get("end_to_end_ms"),
    )
    if e2e_ratio is not None:
        checks.append(
            _check(
                "end_to_end_ratio",
                e2e_ratio <= thresholds["max_end_to_end_ratio"],
                value=e2e_ratio,
            )
        )

    if not checks:
        checks.append(_check("speed_evidence_present", False))
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _iteration_count_checks(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_run_config: dict[str, Any] | None,
    candidate_run_config: dict[str, Any] | None,
    require: bool,
) -> list[dict[str, Any]]:
    expected = _run_config_iters(baseline_run_config)
    actual_expected = _run_config_iters(candidate_run_config)
    if expected is not None and actual_expected is not None and expected != actual_expected:
        return [
            _check(
                "metrics_iteration_count_config_match",
                False,
                expected=expected,
                actual=actual_expected,
            )
        ]
    expected_iters = expected if expected is not None else actual_expected
    baseline_count = _int_value(baseline.get("iteration_count"))
    candidate_count = _int_value(candidate.get("iteration_count"))
    if expected_iters is None:
        return []
    checks = []
    if require:
        checks.append(
            _check(
                "metrics_iteration_count_present",
                baseline_count is not None and candidate_count is not None,
                expected=expected_iters,
                actual={"baseline": baseline_count, "candidate": candidate_count},
            )
        )
    if baseline_count is not None:
        checks.append(
            _check(
                "baseline_iteration_count",
                baseline_count == expected_iters,
                expected=expected_iters,
                actual=baseline_count,
            )
        )
    if candidate_count is not None:
        checks.append(
            _check(
                "candidate_iteration_count",
                candidate_count == expected_iters,
                expected=expected_iters,
                actual=candidate_count,
            )
        )
    return checks


def _latency_sample_checks(role: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    sample = metrics.get("latency_ms")
    if sample is None:
        return []
    if not isinstance(sample, dict):
        return [_check(f"{role}_latency_samples_shape", False, actual=sample)]

    raw_values = sample.get("values")
    checks = [
        _check(
            f"{role}_latency_samples_present",
            isinstance(raw_values, list) and bool(raw_values),
            actual=raw_values,
        )
    ]
    if not isinstance(raw_values, list) or not raw_values:
        return checks

    values = [_float_value(value) for value in raw_values]
    checks.append(
        _check(
            f"{role}_latency_samples_finite",
            all(value is not None and math.isfinite(value) for value in values),
            actual=raw_values,
        )
    )
    if not checks[-1]["passed"]:
        return checks

    numeric_values = [value for value in values if value is not None]
    count = _int_value(metrics.get("iteration_count"))
    if count is not None:
        checks.append(
            _check(
                f"{role}_latency_sample_count",
                len(numeric_values) == count,
                expected=count,
                actual=len(numeric_values),
            )
        )

    measured_mean = sum(numeric_values) / len(numeric_values)
    sample_mean = _float_value(sample.get("mean"))
    if sample_mean is not None:
        checks.append(
            _check(
                f"{role}_latency_sample_mean",
                _close_enough(measured_mean, sample_mean),
                expected=measured_mean,
                actual=sample_mean,
            )
        )
    end_to_end_ms = _float_value(metrics.get("end_to_end_ms"))
    if end_to_end_ms is not None:
        checks.append(
            _check(
                f"{role}_latency_end_to_end_mean",
                _close_enough(measured_mean, end_to_end_ms),
                expected=measured_mean,
                actual=end_to_end_ms,
            )
        )
    return checks


def _float_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _close_enough(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-6, abs(left) * 1e-6, abs(right) * 1e-6)


def _run_config_iters(config: dict[str, Any] | None) -> int | None:
    if not isinstance(config, dict):
        return None
    value = _int_value(config.get("iters"))
    if value is None or value <= 0:
        return None
    return value


def _int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def compare_peak_memory(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    ratio = _latency_ratio(
        candidate.get("peak_memory_bytes"),
        baseline.get("peak_memory_bytes"),
    )
    if ratio is None:
        return {
            "passed": False,
            "checks": [_check("peak_memory_evidence_present", False)],
        }
    check = _check(
        "peak_memory_ratio",
        ratio <= thresholds["max_peak_memory_ratio"],
        value=ratio,
    )
    return {"passed": check["passed"], "checks": [check]}


def compare_device_profile(
    candidate: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    require_profile = bool(thresholds.get("require_device_profile"))
    required_provider = thresholds.get("required_provider")
    required_delegate = thresholds.get("required_delegate")
    required_coreml_device = thresholds.get("required_coreml_preferred_device")
    needs_check = (
        require_profile
        or required_provider is not None
        or required_delegate is not None
        or required_coreml_device is not None
    )
    if not needs_check:
        return {"passed": True, "checks": [_check("device_profile_optional", True)]}

    profile = candidate.get("device_profile")
    diagnostics = (
        profile.get("runtime_diagnostics") if isinstance(profile, dict) else None
    )
    engine = candidate.get("engine")
    checks = [_check("device_profile_present", isinstance(profile, dict))]
    if engine in {"coreml", "onnx", "litert"}:
        checks.append(
            _check("runtime_diagnostics_present", isinstance(diagnostics, dict))
        )
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}

    if engine == "coreml":
        compute_plan = diagnostics.get("compute_plan")
        checks.append(_check("coreml_compute_plan_present", isinstance(compute_plan, dict)))
        compute_plan = compute_plan if isinstance(compute_plan, dict) else {}
        checks.append(_check("coreml_compute_plan_available", compute_plan.get("available") is True))
        checks.append(
            _check(
                "coreml_compute_plan_errors",
                int(compute_plan.get("error_count") or 0) == 0,
                value=compute_plan.get("error_count"),
            )
        )
        if required_coreml_device is not None:
            preferred = compute_plan.get("preferred_device_counts") or {}
            count = int(preferred.get(str(required_coreml_device)) or 0)
            checks.append(
                _check(
                    "coreml_required_preferred_device",
                    count > 0,
                    expected=required_coreml_device,
                    value=count,
                )
            )

    if required_provider is not None:
        providers = _diagnostic_providers(diagnostics)
        checks.append(
            _check(
                "required_provider",
                bool(providers)
                and all(
                    str(required_provider).lower() in provider
                    for provider in providers
                ),
                expected=required_provider,
                actual=providers,
            )
        )

    if required_delegate is not None:
        delegates = diagnostics.get("delegates") or []
        checks.append(
            _check(
                "required_delegate",
                str(required_delegate).lower()
                in {str(delegate).lower() for delegate in delegates},
                expected=required_delegate,
                actual=delegates,
            )
        )

    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _diagnostic_providers(diagnostics: dict[str, Any]) -> list[str]:
    if diagnostics.get("pipeline") is True:
        providers = []
        for stage in diagnostics.get("stages") or []:
            if not isinstance(stage, dict):
                continue
            stage_diagnostics = stage.get("diagnostics")
            if isinstance(stage_diagnostics, dict):
                provider = stage_diagnostics.get("provider")
                if provider is not None:
                    providers.append(str(provider).lower())
        return providers
    provider = diagnostics.get("provider")
    return [str(provider).lower()] if provider is not None else []


def _throughput_ratio(candidate: Any, baseline: Any) -> float | None:
    candidate_f = _positive_float(candidate)
    baseline_f = _positive_float(baseline)
    if candidate_f is None or baseline_f is None:
        return None
    return candidate_f / baseline_f


def _latency_ratio(candidate: Any, baseline: Any) -> float | None:
    candidate_f = _positive_float(candidate)
    baseline_f = _positive_float(baseline)
    if candidate_f is None or baseline_f is None:
        return None
    return candidate_f / baseline_f


def _positive_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _check(name: str, passed: bool, **extra: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, **extra}


if __name__ == "__main__":
    main()
