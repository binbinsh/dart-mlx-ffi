from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Hugging Face ONNX artifact to LiteRT/TFLite with onnx2tf."
        )
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefer-local", action="store_true")
    parser.add_argument(
        "--source-candidate",
        action="append",
        default=[],
        metavar="REPO::ARTIFACT",
        type=_parse_source_candidate,
        help=(
            "Extra ONNX source candidate, formatted as "
            "<huggingface-repo>::<path/to/model.onnx>."
        ),
    )
    parser.add_argument(
        "--onnx2tf-extra-arg",
        action="append",
        default=[],
        help="Extra argument passed through to onnx2tf.",
    )
    parser.add_argument(
        "--retry-auto-prf",
        action="store_true",
        help=(
            "Retry a failed onnx2tf run with the auto-generated -prf JSON when "
            "onnx2tf emits one."
        ),
    )
    parser.add_argument(
        "--no-retry-auto-prf",
        action="store_false",
        dest="retry_auto_prf",
        help="Disable auto -prf retry.",
    )
    parser.add_argument(
        "--patch-onnx2tf",
        action="store_true",
        help="Apply known onnx2tf hotfixes before conversion.",
    )
    parser.add_argument(
        "--no-patch-onnx2tf",
        action="store_false",
        dest="patch_onnx2tf",
        help="Disable onnx2tf hotfixes.",
    )
    parser.add_argument(
        "--fallback-isolated-onnx2tf2",
        action="store_true",
        help=(
            "When the local onnx2tf command fails, retry conversion with an "
            "isolated uv --no-project onnx2tf 2.x toolchain."
        ),
    )
    parser.add_argument(
        "--no-fallback-isolated-onnx2tf2",
        action="store_false",
        dest="fallback_isolated_onnx2tf2",
        help="Disable isolated onnx2tf 2.x fallback.",
    )
    parser.add_argument(
        "--isolated-onnx2tf2-version",
        default="2.4.0",
        help="Pinned onnx2tf version used by isolated fallback.",
    )
    parser.add_argument(
        "--isolated-tensorflow-version",
        default="2.19.0",
        help=(
            "Pinned tensorflow version for isolated fallback. "
            "Set to an empty string to omit tensorflow pinning."
        ),
    )
    parser.add_argument(
        "--isolated-tf-keras-version",
        default="2.19.0",
        help=(
            "Pinned tf-keras version for isolated fallback. "
            "Set to an empty string to omit tf-keras pinning."
        ),
    )
    parser.add_argument(
        "--isolated-workdir",
        type=Path,
        help="Working directory used by isolated fallback commands.",
    )
    parser.add_argument(
        "--attempt-timeout-seconds",
        type=int,
        default=900,
        help=(
            "Timeout for each onnx2tf invocation attempt. "
            "Use 0 to disable per-attempt timeout."
        ),
    )
    parser.add_argument(
        "--tflite-preference",
        choices=["dynamic_range", "float16", "float32"],
        default="dynamic_range",
        help="Preferred .tflite variant selected from onnx2tf outputs.",
    )
    parser.set_defaults(
        retry_auto_prf=True,
        patch_onnx2tf=True,
        fallback_isolated_onnx2tf2=True,
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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

    sources = _build_sources(
        primary=Source(repo=args.repo, artifact=args.artifact),
        extra=args.source_candidate,
    )
    attempts = []
    selected: dict[str, str] | None = None
    last_error: str | None = None
    for index, source in enumerate(sources, start=1):
        result = _convert_source(
            source=source,
            source_index=index,
            output_dir=output_dir,
            revision=args.revision,
            prefer_local=args.prefer_local,
            onnx2tf_extra_args=args.onnx2tf_extra_arg,
            retry_auto_prf=args.retry_auto_prf,
            runners=runners,
            attempt_timeout_seconds=args.attempt_timeout_seconds,
            tflite_preference=args.tflite_preference,
        )
        attempts.append(result)
        if result["success"]:
            selected = result
            break
        last_error = str(result.get("error") or last_error or "")

    final_artifact = output_dir / "model.tflite"
    if selected is None:
        report = {
            "format": "dart_inference.onnx_to_litert.v2",
            "status": "failed",
            "sources": [source.as_dict() for source in sources],
            "attempts": attempts,
            "patches": patch_results,
            "runners": [runner.as_dict() for runner in runners],
            "error": last_error or "onnx2tf conversion failed for all candidates.",
        }
        _write_report(output_dir, report)
        print(json.dumps(report, indent=2))
        raise SystemExit(report["error"])

    selected_tflite = Path(selected["selected_tflite"])
    if final_artifact.exists():
        final_artifact.unlink()
    shutil.copy2(selected_tflite, final_artifact)

    report = {
        "format": "dart_inference.onnx_to_litert.v2",
        "status": "converted",
        "repo": selected["repo"],
        "artifact": selected["artifact"],
        "revision": args.revision,
        "onnx_path": selected["onnx_path"],
        "onnx2tf_output_dir": selected["onnx2tf_output_dir"],
        "selected_tflite": str(selected_tflite),
        "final_artifact": str(final_artifact),
        "tflite_preference": args.tflite_preference,
        "runner": selected["runner"],
        "patches": patch_results,
        "runners": [runner.as_dict() for runner in runners],
        "attempts": attempts,
    }
    _write_report(output_dir, report)
    print(json.dumps(report, indent=2))


class Source(NamedTuple):
    repo: str
    artifact: str

    def as_dict(self) -> dict[str, str]:
        return {"repo": self.repo, "artifact": self.artifact}


class Onnx2TfRunner(NamedTuple):
    id: str
    command_prefix: list[str]
    cwd: Path | None
    extra_args: list[str]

    def as_dict(self) -> dict[str, str | list[str] | None]:
        return {
            "id": self.id,
            "command_prefix": self.command_prefix,
            "cwd": str(self.cwd) if self.cwd else None,
            "extra_args": self.extra_args,
        }


def _build_sources(*, primary: Source, extra: list[Source]) -> list[Source]:
    seen: set[tuple[str, str]] = set()
    result: list[Source] = []
    for source in [primary, *extra]:
        key = (source.repo, source.artifact)
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


def _build_onnx2tf_runners(
    *,
    fallback_isolated_onnx2tf2: bool,
    isolated_onnx2tf2_version: str,
    isolated_tensorflow_version: str | None,
    isolated_tf_keras_version: str | None,
    isolated_workdir: Path | None,
) -> list[Onnx2TfRunner]:
    runners = [
        Onnx2TfRunner(
            id="local",
            command_prefix=["onnx2tf"],
            cwd=None,
            extra_args=[],
        )
    ]
    if not fallback_isolated_onnx2tf2:
        return runners

    if not isolated_onnx2tf2_version:
        raise SystemExit("--isolated-onnx2tf2-version must not be empty.")

    command_prefix = [
        "env",
        "-u",
        "VIRTUAL_ENV",
        "-u",
        "PYTHONPATH",
        "PYTHONNOUSERSITE=1",
        "uv",
        "run",
        "--no-project",
        "--with",
        f"onnx2tf=={isolated_onnx2tf2_version}",
    ]
    if isolated_tensorflow_version:
        command_prefix.extend(["--with", f"tensorflow=={isolated_tensorflow_version}"])
    if isolated_tf_keras_version:
        command_prefix.extend(["--with", f"tf-keras=={isolated_tf_keras_version}"])
    command_prefix.append("onnx2tf")

    cwd = (isolated_workdir or Path(tempfile.gettempdir())).resolve()
    runners.append(
        Onnx2TfRunner(
            id=f"isolated_onnx2tf_{isolated_onnx2tf2_version}",
            command_prefix=command_prefix,
            cwd=cwd,
            extra_args=[],
        )
    )
    runners.append(
        Onnx2TfRunner(
            id=f"isolated_onnx2tf_{isolated_onnx2tf2_version}_tf_converter",
            command_prefix=command_prefix,
            cwd=cwd,
            extra_args=["-tb", "tf_converter", "-agje"],
        )
    )
    return runners


def _convert_source(
    *,
    source: Source,
    source_index: int,
    output_dir: Path,
    revision: str | None,
    prefer_local: bool,
    onnx2tf_extra_args: list[str],
    retry_auto_prf: bool,
    runners: list[Onnx2TfRunner],
    attempt_timeout_seconds: int,
    tflite_preference: str,
) -> dict[str, object]:
    onnx_path = _resolve_onnx_artifact(
        repo=source.repo,
        artifact=source.artifact,
        revision=revision,
        prefer_local=prefer_local,
    )

    attempt_runs: list[dict[str, object]] = []
    runner_errors: list[str] = []
    model_dir = output_dir / f"onnx2tf_out_source{source_index}"
    for runner in runners:
        run_result = _run_onnx2tf_with_runner(
            runner=runner,
            source_index=source_index,
            output_dir=output_dir,
            model_dir=model_dir,
            onnx_path=onnx_path,
            onnx2tf_extra_args=onnx2tf_extra_args,
            retry_auto_prf=retry_auto_prf,
            attempt_timeout_seconds=attempt_timeout_seconds,
        )
        attempt_runs.extend(run_result["attempt_runs"])
        if run_result["success"]:
            tflite_path = _pick_tflite(
                model_dir,
                preference=tflite_preference,
            )
            return {
                "repo": source.repo,
                "artifact": source.artifact,
                "onnx_path": str(onnx_path),
                "onnx2tf_output_dir": str(model_dir),
                "selected_tflite": str(tflite_path),
                "runner": runner.id,
                "attempt_runs": attempt_runs,
                "success": True,
            }
        runner_errors.append(
            _runner_error_text(runner.id, run_result)
        )

    return {
        "repo": source.repo,
        "artifact": source.artifact,
        "onnx_path": str(onnx_path),
        "onnx2tf_output_dir": str(model_dir),
        "attempt_runs": attempt_runs,
        "success": False,
        "error": (
            "onnx2tf failed for source "
            f"{source.repo}::{source.artifact} "
            f"across runners: {', '.join(runner_errors)}"
        ),
    }


def _run_onnx2tf_with_runner(
    *,
    runner: Onnx2TfRunner,
    source_index: int,
    output_dir: Path,
    model_dir: Path,
    onnx_path: Path,
    onnx2tf_extra_args: list[str],
    retry_auto_prf: bool,
    attempt_timeout_seconds: int,
) -> dict[str, object]:
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    base_command = [
        *runner.command_prefix,
        "-i",
        str(onnx_path),
        "-o",
        str(model_dir),
        "-odrqt",
    ]
    base_command.extend(runner.extra_args)
    base_command.extend(onnx2tf_extra_args)

    runner_tag = _sanitize_filename_part(runner.id)
    attempt_runs: list[dict[str, object]] = []
    attempt1_log = output_dir / f"onnx2tf_source{source_index}_{runner_tag}_attempt1.log"
    attempt1 = _run_onnx2tf_attempt(
        command=base_command,
        output_dir=model_dir,
        log_path=attempt1_log,
        cwd=runner.cwd,
        attempt_timeout_seconds=attempt_timeout_seconds,
    )
    attempt_runs.append(
        {
            "runner": runner.id,
            "attempt": 1,
            "command": base_command,
            "cwd": str(runner.cwd) if runner.cwd else None,
            "returncode": attempt1["returncode"],
            "log_path": str(attempt1_log),
            "auto_prf_path": attempt1.get("auto_prf_path"),
            "timed_out": bool(attempt1.get("timed_out")),
            "timeout_seconds": attempt_timeout_seconds,
        }
    )
    final_result = attempt1
    if attempt1["returncode"] != 0 and retry_auto_prf:
        auto_prf = attempt1.get("auto_prf_path")
        if isinstance(auto_prf, str) and auto_prf:
            attempt2_command = [*base_command, "-prf", auto_prf]
            attempt2_log = output_dir / f"onnx2tf_source{source_index}_{runner_tag}_attempt2.log"
            attempt2 = _run_onnx2tf_attempt(
                command=attempt2_command,
                output_dir=model_dir,
                log_path=attempt2_log,
                cwd=runner.cwd,
                attempt_timeout_seconds=attempt_timeout_seconds,
            )
            attempt_runs.append(
                {
                    "runner": runner.id,
                    "attempt": 2,
                    "command": attempt2_command,
                    "cwd": str(runner.cwd) if runner.cwd else None,
                    "returncode": attempt2["returncode"],
                    "log_path": str(attempt2_log),
                    "used_prf": auto_prf,
                    "auto_prf_path": attempt2.get("auto_prf_path"),
                    "timed_out": bool(attempt2.get("timed_out")),
                    "timeout_seconds": attempt_timeout_seconds,
                }
            )
            final_result = attempt2
    return {
        "success": final_result["returncode"] == 0,
        "attempt_runs": attempt_runs,
        "final_returncode": final_result["returncode"],
        "timed_out": bool(final_result.get("timed_out")),
    }


def _run_onnx2tf_attempt(
    *,
    command: list[str],
    output_dir: Path,
    log_path: Path,
    cwd: Path | None,
    attempt_timeout_seconds: int,
) -> dict[str, object]:
    if cwd:
        cwd.mkdir(parents=True, exist_ok=True)
    timeout_seconds = attempt_timeout_seconds if attempt_timeout_seconds > 0 else None
    try:
        completed = subprocess.run(
            command,
            check=False,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        timed_out = False
        output = completed.stdout or ""
    except subprocess.TimeoutExpired as error:
        output = _timeout_output(error, timeout_seconds=timeout_seconds)
        returncode = 124
        timed_out = True
    log_path.write_text(output, encoding="utf-8")
    print(output, end="")
    return {
        "returncode": returncode,
        "auto_prf_path": _find_auto_prf_path(output, output_dir=output_dir),
        "timed_out": timed_out,
    }


def _write_report(output_dir: Path, report: dict[str, object]) -> None:
    (output_dir / "onnx_to_litert_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_source_candidate(value: str) -> Source:
    if "::" not in value:
        raise argparse.ArgumentTypeError(
            "--source-candidate must use the format REPO::ARTIFACT"
        )
    repo, artifact = value.split("::", 1)
    repo = repo.strip()
    artifact = artifact.strip()
    if not repo or not artifact:
        raise argparse.ArgumentTypeError(
            "--source-candidate must use the format REPO::ARTIFACT"
        )
    return Source(repo=repo, artifact=artifact)


def _find_auto_prf_path(output: str, *, output_dir: Path) -> str | None:
    patterns = (
        r"Auto-generated replacement JSON saved to:\s*(.+)",
        r"Please try running the conversion again with:\s*-prf\s+(.+)",
    )
    for pattern in patterns:
        for raw in re.findall(pattern, output):
            candidate = raw.strip().strip('"').strip("'")
            if not candidate:
                continue
            path = Path(candidate)
            if not path.is_absolute():
                path = (output_dir / path).resolve()
            if path.exists():
                return str(path)
    return None


def _apply_onnx2tf_patches() -> list[dict[str, str]]:
    return [
        _patch_onnx2tf_sequenceempty_bug(),
        _patch_onnx2tf_unsqueeze_shape_bug(),
        _patch_onnx2tf_sequence_lookup_bug(),
        _patch_onnx2tf_slice_dtype_bug(),
        _patch_onnx2tf_trilu_dynamic_shape_bug(),
        _patch_onnx2tf_external_data_resave_bug(),
    ]


def _patch_onnx2tf_sequenceempty_bug() -> dict[str, str]:
    result = {"name": "onnx2tf_sequenceempty_dtype_lookup", "status": "skipped"}
    candidates: list[Path] = []
    try:
        import onnx2tf.ops.SequenceEmpty as sequence_empty  # type: ignore[import-not-found]

        candidates.append(Path(sequence_empty.__file__).resolve())
    except Exception as error:  # noqa: BLE001
        result["import_error"] = str(error)
    candidates.extend(_uv_archive_onnx2tf_module_paths("SequenceEmpty.py"))
    return _patch_candidate_paths(
        result=result,
        candidates=candidates,
        patch_fn=_patch_sequenceempty_source,
    )


def _patch_sequenceempty_source(source: str) -> tuple[str, bool]:
    buggy = "ONNX_DTYPES_TO_TF_DTYPES(graph_node.attrs.get('dtype', 1))"
    fixed = "ONNX_DTYPES_TO_TF_DTYPES[graph_node.attrs.get('dtype', 1)]"
    if fixed in source:
        return source, False
    if buggy not in source:
        return source, False
    return source.replace(buggy, fixed), True


def _patch_onnx2tf_unsqueeze_shape_bug() -> dict[str, str]:
    result = {"name": "onnx2tf_unsqueeze_shape_fallback", "status": "skipped"}
    candidates: list[Path] = []
    try:
        import onnx2tf.ops.Unsqueeze as unsqueeze  # type: ignore[import-not-found]

        candidates.append(Path(unsqueeze.__file__).resolve())
    except Exception as error:  # noqa: BLE001
        result["import_error"] = str(error)
    candidates.extend(_uv_archive_onnx2tf_module_paths("Unsqueeze.py"))
    return _patch_candidate_paths(
        result=result,
        candidates=candidates,
        patch_fn=_patch_unsqueeze_source,
    )


def _patch_unsqueeze_source(source: str) -> tuple[str, bool]:
    updated = source
    changed = False
    if "_lookup_unsqueeze_axes(" not in updated:
        axes_old = """    axes = tf_layers_dict[graph_node_input_2.name]['tf_node'] \\
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
"""
        axes_new = """    axes = _lookup_unsqueeze_axes(tf_layers_dict, graph_node_input_2) \\
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
"""
        if axes_old in updated:
            updated = updated.replace(axes_old, axes_new)
            marker = "\n\n@print_node_info"
            helper_block = """

def _lookup_unsqueeze_axes(tf_layers_dict: dict, graph_input: gs.Variable):
    key = getattr(graph_input, 'name', None)
    normalized = key.lstrip('/') if isinstance(key, str) else key
    for candidate in (key, normalized, f'/{normalized}' if normalized else None):
        if not candidate:
            continue
        entry = tf_layers_dict.get(candidate)
        if isinstance(entry, dict) and 'tf_node' in entry:
            return entry['tf_node']
    for producer in getattr(graph_input, 'inputs', []) or []:
        attrs = getattr(producer, 'attrs', {}) or {}
        if 'value' not in attrs:
            continue
        value = attrs['value']
        return getattr(value, 'values', value)
    raise KeyError(key)
"""
            if marker in updated:
                updated = updated.replace(marker, helper_block + marker, 1)
            changed = True

    if "else:\n        input_tensor_shape = [None]\n        tensor_rank = len(input_tensor_shape)" in updated:
        return updated, changed
    original = """    if input_tensor.shape != tf.TensorShape(None):
        input_tensor_shape = list(input_tensor.shape)
        tensor_rank = len(input_tensor_shape)
    elif graph_node_output.shape is not None:
        input_tensor_shape = [
            dim for idx, dim in enumerate(graph_node_output.shape) if idx not in axes
        ]
        input_tensor_shape = [
            dim if not isinstance(dim, str) else None for dim in input_tensor_shape
        ]
        tensor_rank = len(input_tensor_shape)
"""
    fixed = """    if input_tensor.shape != tf.TensorShape(None):
        input_tensor_shape = list(input_tensor.shape)
        tensor_rank = len(input_tensor_shape)
    elif graph_node_output.shape is not None:
        input_tensor_shape = [
            dim for idx, dim in enumerate(graph_node_output.shape) if idx not in axes
        ]
        input_tensor_shape = [
            dim if not isinstance(dim, str) else None for dim in input_tensor_shape
        ]
        tensor_rank = len(input_tensor_shape)
    else:
        input_tensor_shape = [None]
        tensor_rank = len(input_tensor_shape)
"""
    if original not in updated:
        return updated, changed
    return updated.replace(original, fixed), True


def _patch_onnx2tf_slice_dtype_bug() -> dict[str, str]:
    result = {"name": "onnx2tf_slice_dtype_alignment", "status": "skipped"}
    candidates: list[Path] = []
    try:
        import onnx2tf.ops.Slice as slice_op  # type: ignore[import-not-found]

        candidates.append(Path(slice_op.__file__).resolve())
    except Exception as error:  # noqa: BLE001
        result["import_error"] = str(error)
    candidates.extend(_uv_archive_onnx2tf_module_paths("Slice.py"))
    return _patch_candidate_paths(
        result=result,
        candidates=candidates,
        patch_fn=_patch_slice_source,
    )


def _patch_onnx2tf_sequence_lookup_bug() -> dict[str, str]:
    result = {"name": "onnx2tf_sequence_lookup_fallback", "status": "skipped"}
    candidates: list[Path] = []
    module_names = ("SequenceAt.py", "SequenceErase.py", "SequenceInsert.py", "SequenceLength.py")
    try:
        import onnx2tf.ops as onnx2tf_ops  # type: ignore[import-not-found]

        ops_dir = Path(onnx2tf_ops.__file__).resolve().parent
        candidates.extend((ops_dir / module_name for module_name in module_names))
    except Exception as error:  # noqa: BLE001
        result["import_error"] = str(error)
    for module_name in module_names:
        candidates.extend(_uv_archive_onnx2tf_module_paths(module_name))
    return _patch_candidate_paths(
        result=result,
        candidates=candidates,
        patch_fn=_patch_sequence_lookup_source,
    )


def _patch_onnx2tf_trilu_dynamic_shape_bug() -> dict[str, str]:
    result = {"name": "onnx2tf_trilu_dynamic_shape", "status": "skipped"}
    candidates: list[Path] = []
    try:
        import onnx2tf.ops.Trilu as trilu  # type: ignore[import-not-found]

        candidates.append(Path(trilu.__file__).resolve())
    except Exception as error:  # noqa: BLE001
        result["import_error"] = str(error)
    candidates.extend(_uv_archive_onnx2tf_module_paths("Trilu.py"))
    return _patch_candidate_paths(
        result=result,
        candidates=candidates,
        patch_fn=_patch_trilu_source,
    )


def _patch_onnx2tf_external_data_resave_bug() -> dict[str, str]:
    result = {"name": "onnx2tf_external_data_resave", "status": "skipped"}
    candidates: list[Path] = []
    try:
        import onnx2tf.onnx2tf as onnx2tf_main  # type: ignore[import-not-found]

        candidates.append(Path(onnx2tf_main.__file__).resolve())
    except Exception as error:  # noqa: BLE001
        result["import_error"] = str(error)
    candidates.extend(_uv_archive_onnx2tf_main_paths())
    return _patch_candidate_paths(
        result=result,
        candidates=candidates,
        patch_fn=_patch_external_data_resave_source,
    )


def _uv_archive_onnx2tf_module_paths(module_name: str) -> list[Path]:
    bases: list[Path] = []
    uv_cache = (os.environ.get("UV_CACHE_DIR") or "").strip()
    if uv_cache:
        bases.append(Path(uv_cache).expanduser())
    bases.append(Path.home() / ".cache" / "uv")

    matches: list[Path] = []
    for base in bases:
        archive = base / "archive-v0"
        if archive.exists():
            matches.extend(
                archive.glob(f"*/lib/python*/site-packages/onnx2tf/ops/{module_name}")
            )
    return sorted(matches)


def _uv_archive_onnx2tf_main_paths() -> list[Path]:
    bases: list[Path] = []
    uv_cache = (os.environ.get("UV_CACHE_DIR") or "").strip()
    if uv_cache:
        bases.append(Path(uv_cache).expanduser())
    bases.append(Path.home() / ".cache" / "uv")

    matches: list[Path] = []
    for base in bases:
        archive = base / "archive-v0"
        if archive.exists():
            matches.extend(
                archive.glob("*/lib/python*/site-packages/onnx2tf/onnx2tf.py")
            )
    return sorted(matches)


def _patch_candidate_paths(
    *,
    result: dict[str, str],
    candidates: list[Path],
    patch_fn: Callable[[str], tuple[str, bool]],
) -> dict[str, str]:
    unique = _unique_existing_paths(candidates)
    if not unique:
        result["status"] = "not_found"
        return result
    patched_count = 0
    already_count = 0
    write_errors = 0
    read_errors = 0
    for path in unique:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            read_errors += 1
            continue
        patched, changed = patch_fn(source)
        if not changed:
            already_count += 1
            continue
        try:
            path.write_text(patched, encoding="utf-8")
        except OSError:
            write_errors += 1
            continue
        patched_count += 1
    result["candidates"] = str(len(unique))
    result["patched_count"] = str(patched_count)
    result["already_count"] = str(already_count)
    if read_errors:
        result["read_errors"] = str(read_errors)
    if patched_count > 0:
        result["status"] = "patched"
    elif write_errors > 0:
        result["status"] = "write_error"
    else:
        result["status"] = "already_patched"
    return result


def _unique_existing_paths(candidates: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _patch_slice_source(source: str) -> tuple[str, bool]:
    already_begin_end = "begin_clipped = tf.clip_by_value" in source
    already_rank = "elif input_tensor.shape != tf.TensorShape(None):" in source
    begin_old = """            cliped_values = tf.cast(
                tf.subtract(
                    1,
                    tf.clip_by_value(t=begin_, clip_value_min=0, clip_value_max=1),
                ),
                dtype=tf.int32,
            )
"""
    begin_new = """            begin_clipped = tf.clip_by_value(t=begin_, clip_value_min=0, clip_value_max=1)
            begin_scalar = tf.cast(1, begin_clipped.dtype) if hasattr(begin_clipped, 'dtype') else 1
            cliped_values = tf.cast(
                tf.subtract(
                    begin_scalar,
                    begin_clipped,
                ),
                dtype=tf.int32,
            )
"""
    end_old = """            cliped_values = tf.cast(
                tf.subtract(
                    1,
                    tf.clip_by_value(t=end_, clip_value_min=0, clip_value_max=1),
                ),
                dtype=tf.int32,
            )
"""
    rank_old = "            elif input_tensor.shape.rank is not None:\n"
    rank_new = "            elif input_tensor.shape != tf.TensorShape(None):\n"
    end_new = """            end_clipped = tf.clip_by_value(t=end_, clip_value_min=0, clip_value_max=1)
            end_scalar = tf.cast(1, end_clipped.dtype) if hasattr(end_clipped, 'dtype') else 1
            cliped_values = tf.cast(
                tf.subtract(
                    end_scalar,
                    end_clipped,
                ),
                dtype=tf.int32,
            )
"""
    if already_begin_end and already_rank:
        return source, False
    updated = source
    changed = False
    if begin_old in updated and not already_begin_end:
        updated = updated.replace(begin_old, begin_new)
        changed = True
    if end_old in updated and not already_begin_end:
        updated = updated.replace(end_old, end_new)
        changed = True
    if rank_old in updated and not already_rank:
        updated = updated.replace(rank_old, rank_new)
        changed = True
    return updated, changed


def _patch_sequence_lookup_source(source: str) -> tuple[str, bool]:
    helper_name = "_lookup_tf_node"
    if helper_name in source:
        return source, False
    lookup_pattern = re.compile(
        r"tf_layers_dict\[(graph_node_input_[0-9]+)\.name\]\['tf_node'\]"
    )
    rewritten, replaced_count = lookup_pattern.subn(
        r"_lookup_tf_node(tf_layers_dict, \1)",
        source,
    )
    if replaced_count == 0:
        return source, False
    helper_block = """
def _lookup_tf_node(tf_layers_dict: dict, graph_input: gs.Variable):
    key = getattr(graph_input, 'name', None)
    if not isinstance(key, str):
        raise KeyError(key)
    normalized = key.lstrip('/')
    candidates = (key, normalized, f'/{normalized}')
    for candidate in candidates:
        entry = tf_layers_dict.get(candidate)
        if isinstance(entry, dict) and 'tf_node' in entry:
            return entry['tf_node']
    raise KeyError(key)

"""
    marker = "@print_node_info"
    index = rewritten.find(marker)
    if index == -1:
        return rewritten, True
    updated = rewritten[:index] + helper_block + rewritten[index:]
    return updated, True


def _patch_trilu_source(source: str) -> tuple[str, bool]:
    fixed = "tensor_shape[-1] is not None and k > tensor_shape[-1]"
    if fixed in source:
        return source, False
    original = """    if k is not None:
        if k > tensor_shape[-1]:
            k = tensor_shape[-1]
        elif k < 0 - tensor_shape[-2]:
            k = 0 - tensor_shape[-2]
"""
    replacement = """    if k is not None:
        if tensor_shape[-1] is not None and k > tensor_shape[-1]:
            k = tensor_shape[-1]
        elif tensor_shape[-2] is not None and k < 0 - tensor_shape[-2]:
            k = 0 - tensor_shape[-2]
"""
    if original not in source:
        return source, False
    return source.replace(original, replacement), True


def _patch_external_data_resave_source(source: str) -> tuple[str, bool]:
    fixed = "save_as_external_data=True"
    originals = (
        "                        onnx.save(estimated_graph, f=input_onnx_file_path)",
        "                        onnx.save(estimated_graph, input_onnx_file_path)",
    )
    replacement = """                        onnx.save_model(
                            estimated_graph,
                            f=input_onnx_file_path,
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            location=f'{os.path.basename(input_onnx_file_path)}.data',
                            size_threshold=1024,
                            convert_attribute=False,
                        )"""
    if fixed in source and not any(original in source for original in originals):
        return source, False
    updated = source
    changed = False
    for original in originals:
        if original in updated:
            updated = updated.replace(original, replacement)
            changed = True
    return updated, changed


def _resolve_onnx_artifact(
    *,
    repo: str,
    artifact: str,
    revision: str | None,
    prefer_local: bool,
) -> Path:
    candidate = Path(artifact)
    if candidate.exists():
        return candidate.resolve()
    from hf_download import hf_hub_download_with_fallback

    path = hf_hub_download_with_fallback(
        repo_id=repo,
        filename=artifact,
        revision=revision,
        local_files_only=prefer_local,
    )
    return Path(path).resolve()


def _pick_tflite(model_dir: Path, *, preference: str = "dynamic_range") -> Path:
    matches = sorted(model_dir.rglob("*.tflite"))
    if not matches:
        raise RuntimeError(f"onnx2tf did not generate a .tflite artifact in {model_dir}")

    preferred = _preferred_tflite_matches(matches, preference)
    if preferred:
        return preferred[0]
    return matches[0]


def _preferred_tflite_matches(matches: list[Path], preference: str) -> list[Path]:
    if preference == "float16":
        return [path for path in matches if "float16" in path.name.lower() or "fp16" in path.name.lower()]
    if preference == "float32":
        return [
            path
            for path in matches
            if "float32" in path.name.lower()
            and "float16" not in path.name.lower()
            and "dynamic" not in path.name.lower()
            and "drq" not in path.name.lower()
        ]
    return [
        path
        for path in matches
        if "dynamic_range" in path.name.lower() or "drq" in path.name.lower()
    ]


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "runner"


def _runner_error_text(runner_id: str, run_result: dict[str, object]) -> str:
    returncode = run_result.get("final_returncode")
    if run_result.get("timed_out"):
        return f"{runner_id} timeout rc={returncode}"
    return f"{runner_id} rc={returncode}"


def _timeout_output(
    error: subprocess.TimeoutExpired,
    *,
    timeout_seconds: int | None,
) -> str:
    timeout_text = (
        f" after {timeout_seconds}s" if isinstance(timeout_seconds, int) else ""
    )
    header = (
        "ERROR: onnx2tf attempt timed out"
        f"{timeout_text}. command={_command_string(error.cmd)}\n"
    )
    body = _coerce_subprocess_text(error.stdout)
    if not body:
        return header
    return header + body


def _command_string(cmd: object) -> str:
    if isinstance(cmd, (list, tuple)):
        return " ".join(str(part) for part in cmd)
    return str(cmd)


def _coerce_subprocess_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value)


if __name__ == "__main__":
    main()
