from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent


def _expand_command(command: list[Any], context: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for item in command:
        text = str(item)
        if (
            text.startswith("{")
            and text.endswith("}")
            and text.count("{") == 1
            and text.count("}") == 1
        ):
            key = text[1:-1]
            value = context.get(key)
            if isinstance(value, (list, tuple)):
                result.extend(str(arg) for arg in value)
                continue
        if text == "{extra_args}":
            result.extend(str(arg) for arg in context.get("extra_args") or [])
            continue
        result.append(text.format(**context))
    return result


def _seed_models_from_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    raw_models = catalog.get("models")
    if not isinstance(raw_models, dict):
        return {}
    platform_defaults = (catalog.get("defaults") or {}).get("platforms")
    default_by_platform = (
        platform_defaults if isinstance(platform_defaults, dict) else {}
    )
    seeded: dict[str, Any] = {}
    for model_id, model in raw_models.items():
        if not isinstance(model, dict):
            continue
        seeded_model: dict[str, Any] = {
            key: copy.deepcopy(value)
            for key, value in model.items()
            if key
            in {
                "family",
                "source_model",
                "task",
                "required_fixtures",
                "tools_file",
                "embedding_query_file",
                "embedding_dim",
                "artifact_coverage",
                "migrated_platforms",
                "platforms",
                "blocked_platforms",
                "blocked_platform_reports",
                "blocked_platform_failure_classes",
                "blocked_platform_failure_reasons",
                "blocked_platform_logs",
            }
        }
        artifacts = model.get("artifacts")
        if not isinstance(artifacts, dict):
            seeded[model_id] = seeded_model
            continue
        platforms = _seed_platform_cells(
            catalog=catalog,
            model=model,
            artifacts=artifacts,
            platform_defaults=default_by_platform,
        )
        if platforms:
            seeded_model["platforms"] = platforms
        seeded[model_id] = seeded_model
    return seeded


def _seed_platform_cells(
    *,
    catalog: dict[str, Any],
    model: dict[str, Any],
    artifacts: dict[str, Any],
    platform_defaults: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    engine_platforms = catalog.get("engine_platforms")
    engine_platform_map = engine_platforms if isinstance(engine_platforms, dict) else {}
    engine_order_map = catalog.get("engine_order")
    order_by_platform = engine_order_map if isinstance(engine_order_map, dict) else {}
    for platform in _seed_platform_list(catalog, model):
        selection = _select_seed_artifact(
            artifacts=artifacts,
            platform=platform,
            order_by_platform=order_by_platform,
            engine_platform_map=engine_platform_map,
        )
        if selection is None:
            continue
        selected_engine, selected_artifact, fallback_from = selection
        source_uri = _seed_source_uri(selected_artifact)
        cell: dict[str, Any] = {
            "engine": selected_engine,
            "artifact": source_uri,
            "source_uri": source_uri,
            "artifact_source": "huggingface",
        }
        if fallback_from:
            cell["fallback_from"] = fallback_from
        platform_defaults_cell = platform_defaults.get(platform)
        if isinstance(platform_defaults_cell, dict):
            cell.update(copy.deepcopy(platform_defaults_cell))
        baseline_engine = _seed_baseline_engine(model, platform, selected_engine, cell)
        if baseline_engine:
            cell["baseline_engine"] = baseline_engine
        result[platform] = cell
    return result


def _seed_platform_list(catalog: dict[str, Any], model: dict[str, Any]) -> list[str]:
    policy = (catalog.get("support_policy") or {}).get("production_requires") or {}
    policy_platforms = policy.get("platforms")
    ordered = (
        [str(item) for item in policy_platforms]
        if isinstance(policy_platforms, list) and policy_platforms
        else ["ios", "macos", "windows", "linux", "android"]
    )
    model_platforms = model.get("platforms")
    if isinstance(model_platforms, list) and model_platforms:
        selected = {str(item) for item in model_platforms}
        filtered = [platform for platform in ordered if platform in selected]
        extras = sorted(selected.difference(filtered))
        return filtered + extras
    return ordered


def _select_seed_artifact(
    *,
    artifacts: dict[str, Any],
    platform: str,
    order_by_platform: dict[str, Any],
    engine_platform_map: dict[str, Any],
) -> tuple[str, dict[str, Any], list[str]] | None:
    raw_order = order_by_platform.get(platform)
    order = (
        [str(item) for item in raw_order]
        if isinstance(raw_order, list) and raw_order
        else ["coreml", "mlx", "onnx", "litert"]
    )
    fallback_from: list[str] = []
    for engine in order:
        artifact = artifacts.get(engine)
        if not isinstance(artifact, dict):
            if _seed_engine_supports(
                engine=engine,
                platform=platform,
                engine_platform_map=engine_platform_map,
            ):
                fallback_from.append(engine)
            continue
        if not _seed_artifact_supports(
            artifact=artifact,
            engine=engine,
            platform=platform,
            engine_platform_map=engine_platform_map,
        ):
            continue
        return engine, artifact, fallback_from
    return None


def _seed_artifact_supports(
    *,
    artifact: dict[str, Any],
    engine: str,
    platform: str,
    engine_platform_map: dict[str, Any],
) -> bool:
    platforms = artifact.get("platforms")
    if isinstance(platforms, list):
        return platform in {str(item) for item in platforms}
    return _seed_engine_supports(
        engine=engine,
        platform=platform,
        engine_platform_map=engine_platform_map,
    )


def _seed_engine_supports(
    *,
    engine: str,
    platform: str,
    engine_platform_map: dict[str, Any],
) -> bool:
    raw = engine_platform_map.get(engine)
    if isinstance(raw, list) and raw:
        return platform in {str(item) for item in raw}
    return True


def _seed_source_uri(artifact: dict[str, Any]) -> str:
    repo = str(artifact.get("repo") or "")
    path = str(artifact.get("artifact") or ".")
    return f"hf://{repo}/{path}"


def _seed_baseline_engine(
    model: dict[str, Any],
    platform: str,
    selected_engine: str,
    cell: dict[str, Any],
) -> str | None:
    default = cell.get("baseline_engine")
    if platform in {"ios", "macos"}:
        if model.get("coreml_llm_baseline") is False:
            return "mlx"
        return "coreml-llm" if selected_engine == "coreml" else "mlx"
    if isinstance(default, str) and default and default != "external":
        return default
    return selected_engine


def _expand_env(raw_env: Any, context: dict[str, Any]) -> dict[str, str]:
    if not isinstance(raw_env, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in raw_env.items():
        key_text = str(key).strip()
        if not key_text or value is None:
            continue
        result[key_text] = str(value).format(**context)
    return result


def _classify_conversion_failure(log_path: Path, returncode: int) -> dict[str, Any]:
    text = ""
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    lowered = text.lower()
    result: dict[str, Any] = {}
    failure_reason: str | None = None
    if returncode == 124:
        result["failure_class"] = "conversion_timeout"
        failure_reason = "Converter exceeded timeout."
    elif returncode in {130, 143}:
        result["failure_class"] = "conversion_interrupted"
        failure_reason = "Converter process was interrupted before completion."
    elif "you are trying to access a gated repo" in lowered:
        result["failure_class"] = "gated_repo_access_denied"
        failure_reason = "Source model is gated on Hugging Face and requires an authenticated token."
    elif "onnx2tf attempt timed out" in lowered:
        result["failure_class"] = "onnx2tf_attempt_timeout"
        failure_reason = "onnx2tf attempt exceeded per-attempt timeout."
    elif "mutex lock failed" in lowered:
        result["failure_class"] = "exporter_runtime_crash"
        failure_reason = "Exporter crashed while acquiring a runtime mutex."
    elif "llvm error: inconsistency in registered commandline options" in lowered:
        result["failure_class"] = "exporter_runtime_crash"
        failure_reason = "Exporter crashed due to LLVM command-line option collision."
    elif "no module named 'onnx2tf.ops.loop'" in lowered:
        result["failure_class"] = "onnx2tf_unsupported_operator_loop"
        failure_reason = "onnx2tf does not implement Loop op conversion."
    elif "loop op is not yet implemented." in lowered:
        result["failure_class"] = "onnx2tf_unsupported_operator_loop"
        failure_reason = "onnx2tf does not implement Loop op conversion."
    elif "onnx2tf/ops/sequenceempty.py" in lowered and "dict' object is not callable" in lowered:
        result["failure_class"] = "onnx2tf_sequenceempty_bug"
        failure_reason = "onnx2tf SequenceEmpty dtype lookup bug (dict called like function)."
    elif "onnx2tf/ops/unsqueeze.py" in lowered and "input_tensor_shape" in lowered and "unboundlocalerror" in lowered:
        result["failure_class"] = "onnx2tf_unsqueeze_shape_bug"
        failure_reason = (
            "onnx2tf Unsqueeze shape fallback bug "
            "(input_tensor_shape is unbound for unknown-rank tensors)."
        )
    elif (
        "nodes in a graph must be topologically sorted" in lowered
        and "constant_" in lowered
        and "optimum____if" in lowered
    ):
        result["failure_class"] = "onnx_invalid_subgraph_constant_binding"
        failure_reason = (
            "The ONNX artifact has If-subgraph Constant tensor bindings that "
            "are not valid for direct onnx2tf/LiteRT conversion; use the ONNX "
            "Runtime artifact as-is or re-export a LiteRT-specific graph."
        )
    elif "keyerror:" in lowered and "onnx2tf/ops/if.py" in lowered:
        result["failure_class"] = "onnx2tf_if_subgraph_binding_bug"
        failure_reason = (
            "onnx2tf failed resolving If subgraph tensors (KeyError in onnx2tf/ops/If.py)."
        )
    elif "keyerror:" in lowered and "onnx2tf/ops/sequenceat.py" in lowered:
        result["failure_class"] = "onnx2tf_sequence_binding_bug"
        failure_reason = (
            "onnx2tf failed resolving SequenceAt tensors "
            "(KeyError in onnx2tf/ops/SequenceAt.py)."
        )
    elif "keyerror:" in lowered and "onnx2tf/ops/gather.py" in lowered:
        result["failure_class"] = "onnx2tf_graph_binding_bug"
        failure_reason = (
            "onnx2tf failed resolving Gather input tensors (KeyError in onnx2tf/ops/Gather.py)."
        )
    elif (
        "nodes in a graph must be topologically sorted" in lowered
        and "/splittosequence_output_0" in lowered
    ):
        result["failure_class"] = "onnx_invalid_graph_topology"
        failure_reason = (
            "ONNX graph is not topologically sorted "
            "(missing producer for /SplitToSequence_output_0)."
        )
    elif "read this and deal with it. https://github.com/pinto0309/onnx2tf#parameter-replacement" in lowered:
        result["failure_class"] = "onnx2tf_parameter_replacement_required"
        failure_reason = "onnx2tf requested parameter-replacement JSON."
    elif "concat input dtypes must be compatible in flatbuffer_direct" in lowered:
        result["failure_class"] = "onnx2tf_concat_dtype_mismatch"
        failure_reason = (
            "onnx2tf flatbuffer_direct failed due to mixed input dtypes in Concat."
        )
    elif "onnx2tf/ops/slice.py" in lowered and "attributeerror: 'tuple' object has no attribute 'rank'" in lowered:
        result["failure_class"] = "onnx2tf_slice_shape_rank_bug"
        failure_reason = (
            "onnx2tf Slice bug: tuple shape object does not expose .rank."
        )
    elif "onnx2tf/ops/slice.py" in lowered and "input 'y' of 'sub' op has type int64 that does not match type int32" in lowered:
        result["failure_class"] = "onnx2tf_slice_dtype_mismatch"
        failure_reason = "onnx2tf Slice lowering produced int32/int64 subtraction mismatch."
    elif "no such file or directory: 'onnxsim'" in lowered:
        result["failure_class"] = "converter_dependency_missing"
        failure_reason = "onnxsim executable is missing."
    elif "onnx_op_name:" in lowered and "onnx2tf" in lowered:
        result["failure_class"] = "onnx2tf_conversion_failed"
        failure_reason = "onnx2tf failed on a specific ONNX operator."
    elif "cannot import name 'check_model_inputs'" in lowered:
        result["failure_class"] = "transformers_api_mismatch"
        failure_reason = "Exporter expects a Transformers API removed in current version."
    elif "object has no attribute 'text_config'" in lowered:
        result["failure_class"] = "transformers_config_mismatch"
        failure_reason = "Model config is incompatible with current Transformers exporter."
    elif (
        "asked to export a gemma3 model for the task image-to-text" in lowered
        and "only supports the tasks" in lowered
    ):
        result["failure_class"] = "vlm_onnx_exporter_missing_for_architecture"
        failure_reason = (
            "Optimum ONNX supports this Gemma3 architecture for text tasks, "
            "but not full image-to-text VLM export."
        )
    elif (
        "unrecognized configuration class" in lowered
        and "minicpmoconfig" in lowered
        and (
            "automodelforvision2seq" in lowered
            or "automodelforimagetexttotext" in lowered
        )
    ):
        result["failure_class"] = "vlm_automodel_mapping_missing"
        failure_reason = (
            "Transformers/Optimum image-text AutoModel mapping has no MiniCPM-o "
            "configuration support for full VLM export."
        )
    elif (
        "unrecognized configuration class" in lowered
        and "for this kind of automodel" in lowered
    ):
        result["failure_class"] = "model_architecture_unsupported"
        failure_reason = (
            "Model configuration is not supported by the selected AutoModel "
            "export path."
        )
    elif "transformers does not recognize this architecture" in lowered:
        result["failure_class"] = "model_architecture_unsupported"
        failure_reason = "Model architecture is not recognized by the current Transformers build."
    elif (
        "does not appear to have a file named configuration_" in lowered
        and "transformers/utils/hub.py" in lowered
    ):
        result["failure_class"] = "dynamic_module_file_missing"
        failure_reason = (
            "Model repository is missing dynamic-module configuration files "
            "declared by auto_map."
        )
    elif (
        "this modeling file requires the following packages that were not found in your environment"
        in lowered
        and "configuration_" in lowered
    ):
        result["failure_class"] = "dynamic_module_dependency_missing"
        failure_reason = (
            "Dynamic-module local dependencies are missing from the model "
            "snapshot used for export."
        )
    elif "requires the following packages that were not found in your environment" in lowered:
        result["failure_class"] = "converter_dependency_missing"
        failure_reason = "Exporter is missing required Python dependencies."
    elif "ming omni tts litert component export is incomplete" in lowered:
        result["failure_class"] = "tts_component_runtime_incomplete"
        failure_reason = (
            "Ming Omni TTS LLM LiteRT export succeeded, but native flow/audio "
            "decoder components are still required for a complete TTS runtime."
        )
    elif "but cannot be imported" in lowered and " is required by " in lowered:
        result["failure_class"] = "converter_dependency_missing"
        failure_reason = "Exporter runtime dependency is missing."
    elif "modulenotfounderror: no module named" in lowered:
        result["failure_class"] = "converter_dependency_missing"
        failure_reason = "Exporter is missing a Python module dependency."
    elif "nvcc was not found" in lowered and "mamba_ssm was requested" in lowered:
        result["failure_class"] = "converter_dependency_requires_cuda_toolchain"
        failure_reason = (
            "A required exporter dependency needs the CUDA/NVCC build toolchain."
        )
    elif (
        "problem with the package or the build" in lowered
        and "environment" in lowered
    ):
        result["failure_class"] = "converter_dependency_build_failed"
        failure_reason = "A required exporter dependency failed to build from source."
    elif "please use a supported task" in lowered and "only supports the tasks" in lowered:
        result["failure_class"] = "export_task_unsupported"
        failure_reason = "Exporter does not support this task for the selected model architecture."
    elif "valueerror: unsupported task:" in lowered:
        result["failure_class"] = "export_task_unsupported"
        failure_reason = "Exporter does not support the requested conversion task."
    elif "unrecognized model in" in lowered and "`model_type` key" in lowered:
        result["failure_class"] = "model_architecture_unsupported"
        failure_reason = "Model architecture is not recognized by the exporter."
    elif "out of memory" in lowered or "cuda out of memory" in lowered:
        result["failure_class"] = "conversion_oom"
        failure_reason = "Conversion process ran out of memory."
    elif returncode == 134:
        result["failure_class"] = "exporter_runtime_crash"
        failure_reason = "Exporter aborted with SIGABRT (return code 134)."
    elif returncode != 0:
        result["failure_class"] = "conversion_failed"

    log_tail = _log_tail(text)
    if failure_reason:
        result["failure_reason"] = failure_reason
    elif log_tail:
        result["failure_reason"] = log_tail
    return result


def _log_tail(text: str, *, max_lines: int = 12, max_chars: int = 1200) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


def _normalized_extra_args(
    exporter: str,
    extra_args: list[Any],
) -> tuple[list[str], list[str]]:
    args = [str(arg) for arg in extra_args]
    if exporter != "coreml-llm":
        return args, []
    unsupported_flags = {"--trust-remote-code"}
    kept: list[str] = []
    ignored: list[str] = []
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in unsupported_flags:
            ignored.append(arg)
            continue
        if any(arg.startswith(f"{flag}=") for flag in unsupported_flags):
            ignored.append(arg)
            continue
        kept.append(arg)
        if arg in unsupported_flags and index + 1 < len(args):
            skip_next = True
    return kept, ignored


def _normalized_with_packages(extra_with: Any) -> list[str]:
    if extra_with is None:
        return []
    if isinstance(extra_with, str):
        values = [extra_with]
    elif isinstance(extra_with, (list, tuple)):
        values = [str(item) for item in extra_with]
    else:
        values = [str(extra_with)]
    result: list[str] = []
    for value in values:
        package = value.strip()
        if not package or package in result:
            continue
        result.append(package)
    return result


def _artifact_health_command(
    plan: dict[str, Any],
    artifact: Path,
    platform: str,
    report: Path,
) -> list[str]:
    engine = plan["engine"]
    if engine == "onnx":
        cmd = ["uv", "run", "--group", "onnx-convert", "python"]
    else:
        cmd = [sys.executable]
    cmd.extend(
        [
            str(RUNTIME_DIR / "artifact_health.py"),
            "--engine",
            engine,
            "--platform",
            platform,
            "--artifact",
            str(artifact),
            "--out",
            str(report),
        ]
    )
    provider = (plan.get("provider_by_platform") or {}).get(platform)
    delegate = (plan.get("delegate_by_platform") or {}).get(platform)
    if engine == "onnx" and provider:
        cmd.extend(["--provider", str(provider)])
    if engine == "litert" and delegate:
        cmd.extend(["--delegate", str(delegate)])
    return cmd


def _converter_cache_env() -> dict[str, str]:
    defaults = {
        "UV_CACHE_DIR": ROOT / "benchmark" / ".uv_cache",
        "HF_HOME": ROOT / "benchmark" / ".hf_home",
        "XDG_CACHE_HOME": ROOT / "benchmark" / ".cache",
    }
    env: dict[str, str] = {}
    for name, path in defaults.items():
        if os.environ.get(name):
            continue
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    if not os.environ.get("TOKENIZERS_PARALLELISM"):
        env["TOKENIZERS_PARALLELISM"] = "false"
    if not os.environ.get("HF_HUB_DISABLE_XET"):
        env["HF_HUB_DISABLE_XET"] = "1"
    return env


def _find_artifact(output_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns or ["*"]:
        matches = sorted(
            path for path in output_dir.glob(pattern) if _is_runtime_artifact(path)
        )
        if matches:
            return matches[0]
    return None


def _is_runtime_artifact(path: Path) -> bool:
    if "_patched_source" in path.parts:
        return False
    if path.is_file():
        if path.suffix.lower() == ".json":
            return _is_pipeline_artifact(path)
        return path.suffix.lower() in {
            ".onnx",
            ".tflite",
            ".task",
            ".litertlm",
        }
    if path.is_dir():
        return path.suffix.lower() in {".mlmodelc", ".mlpackage"}
    return False


def _is_pipeline_artifact(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    return data.get("format") in {
        "dart_mlx_ffi.coreml_pipeline.v1",
        "dart_mlx_ffi.onnx_pipeline.v1",
        "dart_mlx_ffi.litert_pipeline.v1",
        "dart_mlx_ffi.qwen3_asr_litert_bundle.v1",
    }


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return ROOT / expanded


def _rel(path: Path) -> str:
    raw = path.expanduser()
    try:
        return str(raw.absolute().relative_to(ROOT))
    except ValueError:
        pass
    try:
        return str(raw.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(raw)


def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


def _existing_log_path(record: dict[str, Any], record_path: Path) -> Path | None:
    log_path = record.get("log_path")
    if isinstance(log_path, str) and log_path.strip():
        path = Path(log_path)
        return path if path.is_absolute() else ROOT / path
    default = record_path.parent / "conversion.log"
    return default if default.exists() else None


def _safe_tool_name(repo: str) -> str:
    name = repo.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name.lower()


def _run_tool_command(cmd: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(cmd, cwd=cwd or ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"Tool setup command failed with exit code {completed.returncode}: "
            + " ".join(cmd)
        )
