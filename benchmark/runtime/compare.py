from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "min_speed_ratio": 0.90,
    "max_ttft_ratio": 1.10,
    "max_peak_memory_ratio": 1.10,
    "min_embedding_cosine": 0.999,
    "max_embedding_l2": 0.001,
    "max_abs_diff": 1e-4,
    "require_device_profile": False,
    "required_coreml_preferred_device": None,
    "required_provider": None,
    "required_delegate": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare runtime candidate and baseline reports."
    )
    parser.add_argument("report", type=Path, help="JSON report to compare")
    parser.add_argument("--out", type=Path, help="Optional output JSON path")
    parser.add_argument("--min-speed-ratio", type=float)
    parser.add_argument("--max-ttft-ratio", type=float)
    parser.add_argument("--max-peak-memory-ratio", type=float)
    parser.add_argument("--min-embedding-cosine", type=float)
    parser.add_argument("--max-embedding-l2", type=float)
    parser.add_argument("--max-abs-diff", type=float)
    parser.add_argument("--require-device-profile", action="store_true")
    parser.add_argument("--required-coreml-preferred-device")
    parser.add_argument("--required-provider")
    parser.add_argument("--required-delegate")
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
    correctness = compare_correctness(
        baseline.get("correctness") or {},
        candidate.get("correctness") or {},
        thresholds,
    )
    speed = compare_speed(
        baseline.get("metrics") or {},
        candidate.get("metrics") or {},
        thresholds,
    )
    peak_memory = compare_peak_memory(
        baseline.get("metrics") or {},
        candidate.get("metrics") or {},
        thresholds,
    )
    device_profile = compare_device_profile(candidate, thresholds)
    return {
        "passed": correctness["passed"]
        and speed["passed"]
        and peak_memory["passed"]
        and device_profile["passed"],
        "correctness": correctness,
        "speed": speed,
        "peak_memory": peak_memory,
        "device_profile": device_profile,
    }


def compare_correctness(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: dict[str, float],
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

    if not checks:
        checks.append(_check("correctness_evidence_present", False))

    return {"passed": all(check["passed"] for check in checks), "checks": checks}


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
            max_abs_diff = max(max_abs_diff, abs(float(left) - float(right)))
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
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
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
                e2e_ratio <= thresholds["max_ttft_ratio"],
                value=e2e_ratio,
            )
        )

    if not checks:
        checks.append(_check("speed_evidence_present", False))
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


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
