from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from onnx_to_litert import (
    Source,
    _apply_onnx2tf_patches,
    _build_onnx2tf_runners,
    _convert_source,
)


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a catalog ONNX component pipeline into a LiteRT pipeline."
        )
    )
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-engine", default="onnx")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--prefer-local", action="store_true")
    parser.add_argument("--onnx2tf-extra-arg", action="append", default=[])
    parser.add_argument("--retry-auto-prf", action="store_true")
    parser.add_argument("--no-retry-auto-prf", action="store_false", dest="retry_auto_prf")
    parser.add_argument("--patch-onnx2tf", action="store_true")
    parser.add_argument("--no-patch-onnx2tf", action="store_false", dest="patch_onnx2tf")
    parser.add_argument("--fallback-isolated-onnx2tf2", action="store_true")
    parser.add_argument(
        "--no-fallback-isolated-onnx2tf2",
        action="store_false",
        dest="fallback_isolated_onnx2tf2",
    )
    parser.add_argument("--isolated-onnx2tf2-version", default="2.4.0")
    parser.add_argument("--isolated-tensorflow-version", default="2.19.0")
    parser.add_argument("--isolated-tf-keras-version", default="2.19.0")
    parser.add_argument("--isolated-workdir", type=Path)
    parser.add_argument("--attempt-timeout-seconds", type=int, default=900)
    parser.set_defaults(
        retry_auto_prf=True,
        patch_onnx2tf=True,
        fallback_isolated_onnx2tf2=True,
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = _catalog_artifact(
        catalog_path=args.catalog,
        model_id=args.model_id,
        source_engine=args.source_engine,
    )
    component_artifacts = artifact.get("component_artifacts") or {}
    pipeline = artifact.get("pipeline") or {}
    if not isinstance(component_artifacts, dict) or not component_artifacts:
        raise SystemExit(f"{args.model_id}/{args.source_engine} has no component_artifacts")
    if not isinstance(pipeline, dict) or not pipeline.get("stages"):
        raise SystemExit(f"{args.model_id}/{args.source_engine} has no pipeline spec")

    patch_results = (
        _apply_onnx2tf_patches() if args.patch_onnx2tf else [{"status": "disabled"}]
    )
    runners = _build_onnx2tf_runners(
        fallback_isolated_onnx2tf2=args.fallback_isolated_onnx2tf2,
        isolated_onnx2tf2_version=args.isolated_onnx2tf2_version.strip(),
        isolated_tensorflow_version=_empty_to_none(args.isolated_tensorflow_version),
        isolated_tf_keras_version=_empty_to_none(args.isolated_tf_keras_version),
        isolated_workdir=args.isolated_workdir.resolve()
        if isinstance(args.isolated_workdir, Path)
        else None,
    )

    repo = str(artifact.get("repo") or "")
    if not repo:
        raise SystemExit(f"{args.model_id}/{args.source_engine} artifact has no repo")

    converted: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    for index, (name, onnx_path) in enumerate(sorted(component_artifacts.items()), start=1):
        component_dir = output_dir / "components" / _safe_name(str(name))
        component_dir.mkdir(parents=True, exist_ok=True)
        result = _convert_source(
            source=Source(repo=repo, artifact=str(onnx_path)),
            source_index=index,
            output_dir=component_dir,
            revision=args.revision,
            prefer_local=args.prefer_local,
            onnx2tf_extra_args=args.onnx2tf_extra_arg,
            retry_auto_prf=args.retry_auto_prf,
            runners=runners,
            attempt_timeout_seconds=args.attempt_timeout_seconds,
        )
        attempts.append({"component": name, **result})
        if not result.get("success"):
            _write_report(
                output_dir,
                {
                    "format": "dart_mlx_ffi.onnx_pipeline_to_litert.v1",
                    "status": "failed",
                    "model_id": args.model_id,
                    "component": name,
                    "attempts": attempts,
                    "patches": patch_results,
                    "error": result.get("error"),
                },
            )
            raise SystemExit(str(result.get("error") or f"{name} conversion failed"))
        final_tflite = component_dir / "model.tflite"
        selected = Path(str(result["selected_tflite"]))
        if final_tflite.exists():
            final_tflite.unlink()
        shutil.copy2(selected, final_tflite)
        converted[str(name)] = {
            "onnx_artifact": str(onnx_path),
            "tflite": final_tflite,
            "inputs": _tflite_tensor_names(final_tflite, inputs=True),
            "outputs": _tflite_tensor_names(final_tflite, inputs=False),
            "report": result,
        }

    litert_pipeline = _litert_pipeline_spec(
        pipeline=pipeline,
        converted=converted,
        output_dir=output_dir,
    )
    pipeline_path = output_dir / "pipeline.json"
    pipeline_path.write_text(
        json.dumps(litert_pipeline, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "format": "dart_mlx_ffi.onnx_pipeline_to_litert.v1",
        "status": "converted",
        "model_id": args.model_id,
        "source_engine": args.source_engine,
        "artifact": str(pipeline_path),
        "components": {
            name: {
                "onnx_artifact": data["onnx_artifact"],
                "tflite": str(data["tflite"]),
                "inputs": data["inputs"],
                "outputs": data["outputs"],
            }
            for name, data in converted.items()
        },
        "patches": patch_results,
        "runners": [runner.as_dict() for runner in runners],
        "attempts": attempts,
    }
    _write_report(output_dir, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _catalog_artifact(
    *,
    catalog_path: Path,
    model_id: str,
    source_engine: str,
) -> dict[str, Any]:
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    model = ((catalog.get("models") or {}).get(model_id) or {})
    artifact = ((model.get("artifacts") or {}).get(source_engine) or {})
    if not isinstance(artifact, dict):
        return {}
    return artifact


def _litert_pipeline_spec(
    *,
    pipeline: dict[str, Any],
    converted: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    spec = _replace_component_refs(copy.deepcopy(pipeline), converted, output_dir)
    spec["format"] = "dart_mlx_ffi.litert_pipeline.v1"
    stages = spec.get("stages") or []
    final_outputs = spec.get("outputs") if isinstance(spec.get("outputs"), list) else []
    model_stage_indexes = [
        index for index, stage in enumerate(stages) if isinstance(stage, dict) and stage.get("model")
    ]
    last_model_stage = model_stage_indexes[-1] if model_stage_indexes else None
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or not stage.get("model"):
            continue
        component_name = _component_name_from_model(str(stage["model"]), converted)
        if component_name is None:
            continue
        signature = converted[component_name]
        inputs = stage.get("inputs") if isinstance(stage.get("inputs"), dict) else {}
        stage["inputs"] = {
            **{name: name for name in signature["inputs"]},
            **inputs,
        }
        outputs = stage.get("outputs") if isinstance(stage.get("outputs"), dict) else {}
        stage["outputs"] = _remap_outputs(
            tflite_outputs=list(signature["outputs"]),
            requested_outputs=final_outputs if index == last_model_stage else [],
            existing=outputs,
        )
    return spec


def _replace_component_refs(
    value: Any,
    converted: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(r"\{component:([^}]+)\}", value)
        if match:
            name = match.group(1)
            if name not in converted:
                raise SystemExit(f"Pipeline references unknown component: {name}")
            return str(Path(converted[name]["tflite"]).relative_to(output_dir))
        return value
    if isinstance(value, list):
        return [_replace_component_refs(item, converted, output_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_component_refs(item, converted, output_dir)
            for key, item in value.items()
        }
    return value


def _component_name_from_model(
    model: str,
    converted: dict[str, dict[str, Any]],
) -> str | None:
    normalized = model.replace("\\", "/")
    for name, data in converted.items():
        rel = str(Path(data["tflite"]).name)
        if normalized.endswith(f"/{rel}") or normalized == rel:
            return name
        parent = f"components/{_safe_name(name)}/model.tflite"
        if normalized.endswith(parent):
            return name
    return None


def _remap_outputs(
    *,
    tflite_outputs: list[str],
    requested_outputs: list[Any],
    existing: dict[str, str],
) -> dict[str, str]:
    if existing:
        values = list(existing.values())
        if len(values) == len(tflite_outputs):
            return dict(zip(tflite_outputs, values, strict=True))
        return existing
    requested = [str(item) for item in requested_outputs if isinstance(item, str)]
    if len(requested) == len(tflite_outputs):
        return dict(zip(tflite_outputs, requested, strict=True))
    return {}


def _tflite_tensor_names(path: Path, *, inputs: bool) -> list[str]:
    try:
        import tensorflow as tf
    except Exception as error:  # pragma: no cover - depends on conversion env
        raise SystemExit(f"TensorFlow is required to inspect {path}: {error}") from error
    interpreter = tf.lite.Interpreter(model_path=str(path))
    details = (
        interpreter.get_input_details()
        if inputs
        else interpreter.get_output_details()
    )
    names = []
    for index, detail in enumerate(details):
        name = str(detail.get("name") or f"{'input' if inputs else 'output'}_{index}")
        names.append(name)
    return names


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "onnx_pipeline_to_litert_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())


def _empty_to_none(value: str) -> str | None:
    value = value.strip()
    return value or None


if __name__ == "__main__":
    main()
