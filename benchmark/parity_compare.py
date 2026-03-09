from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-json", default="benchmark/out/parity_python.json")
    parser.add_argument("--dart-json", default="benchmark/out/parity_dart.json")
    parser.add_argument("--output", default="benchmark/out/parity_report.json")
    args = parser.parse_args()

    py = json.loads((ROOT / args.python_json).read_text(encoding="utf-8"))
    da = json.loads((ROOT / args.dart_json).read_text(encoding="utf-8"))

    report = compare(py["cases"], da["cases"])
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"parity_report: {out}")
    print(f"groups: {len(report['groups'])}")
    print(f"outputs: {report['total_outputs']}")
    print(f"failures: {len(report['failures'])}")
    if report["failures"]:
        for failure in report["failures"][:20]:
            print(
                failure["group"],
                failure["name"],
                failure["reason"],
                failure.get("max_abs_diff"),
            )


def compare(py_cases: dict[str, Any], da_cases: dict[str, Any]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_outputs = 0
    for group, py_outputs in py_cases.items():
        da_outputs = da_cases[group]
        group_ok = True
        for name, py_value in py_outputs.items():
            total_outputs += 1
            da_value = da_outputs[name]
            ok, detail = compare_entry(py_value, da_value)
            if not ok:
                group_ok = False
                failures.append({"group": group, "name": name, **detail})
        groups.append({"group": group, "ok": group_ok, "count": len(py_outputs)})
    return {
        "groups": groups,
        "total_outputs": total_outputs,
        "failures": failures,
    }


def compare_entry(py: dict[str, Any], da: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if py["shape"] != da["shape"]:
        return False, {"reason": "shape_mismatch", "python_shape": py["shape"], "dart_shape": da["shape"]}
    diffs = []
    for a, b in zip(py["values"], da["values"]):
        if isinstance(a, bool) or isinstance(b, bool):
            if a != b:
                return False, {"reason": "bool_mismatch"}
            continue
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                return False, {"reason": "int_mismatch"}
            continue
        diffs.append(abs(float(a) - float(b)))
    if not diffs:
        return True, {}
    max_abs = max(diffs)
    mean_abs = sum(diffs) / len(diffs)
    if max_abs > 1e-4:
        return False, {"reason": "float_mismatch", "max_abs_diff": max_abs, "mean_abs_diff": mean_abs}
    return True, {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs}


if __name__ == "__main__":
    main()
