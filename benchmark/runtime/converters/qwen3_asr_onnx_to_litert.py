from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, NamedTuple

from hf_download import (
    DEFAULT_FALLBACK_ENDPOINT,
    hf_hub_download_with_fallback,
    snapshot_download_with_fallback,
)
from onnx_to_litert import (
    Source,
    _apply_onnx2tf_patches,
    _build_onnx2tf_runners,
    _convert_source,
)
from qwen3_asr_onnx_rewrite import (
    _dequantize_matmulnbits_weight,
    _rewrite_decoder_custom_ops,
    _write_decoder_compatible_onnx,
)


MODEL_ID = "qwen3_asr"
SOURCE_MODEL = "Qwen/Qwen3-ASR-1.7B"
BUNDLE_FORMAT = "dart_mlx_ffi.qwen3_asr_litert_bundle.v1"
RUNNER = "Qwen3AsrNativeRunner.loadLiteRtBundle"
DEFAULT_ENCODER_MEL_FRAMES = 3000
DEFAULT_DECODER_INIT_SEQ_LEN = 128
DEFAULT_DECODER_AUDIO_LEN = 104


class ComponentSet(NamedTuple):
    name: str
    components: dict[str, str]
    external_data: list[str]


class PreparedComponent(NamedTuple):
    source: Path
    extra_args: list[str]


INT4_COMPONENTS = ComponentSet(
    name="int4",
    components={
        "encoder": "encoder.int4.onnx",
        "decoder_init": "decoder_init.int4.onnx",
        "decoder_step": "decoder_step.int4.onnx",
    },
    external_data=["decoder_weights.int4.data"],
)

FP32_COMPONENTS = ComponentSet(
    name="fp32",
    components={
        "encoder": "encoder.onnx",
        "decoder_init": "decoder_init.onnx",
        "decoder_step": "decoder_step.onnx",
    },
    external_data=["decoder_weights.data"],
)

SIDECAR_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "embed_tokens.bin",
)
REQUIRED_SIDECAR_FILES = ("config.json", "embed_tokens.bin")
TOKENIZER_SIDECAR_FILES = ("tokenizer.json", "vocab.json", "merges.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Qwen3-ASR 1.7B ONNX component bundle into a "
            "same-model LiteRT component bundle."
        )
    )
    parser.add_argument("--repo", default="andrewleech/qwen3-asr-1.7b-onnx")
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--prefer-local", action="store_true")
    parser.add_argument("--prefer-fp32", action="store_true")
    parser.add_argument("--endpoint")
    parser.add_argument(
        "--fallback-endpoint",
        default=os.environ.get("HF_FALLBACK_ENDPOINT") or DEFAULT_FALLBACK_ENDPOINT,
    )
    parser.add_argument("--onnx2tf-extra-arg", action="append", default=[])
    parser.add_argument("--retry-auto-prf", action="store_true")
    parser.add_argument(
        "--no-retry-auto-prf",
        action="store_false",
        dest="retry_auto_prf",
    )
    parser.add_argument("--patch-onnx2tf", action="store_true")
    parser.add_argument(
        "--no-patch-onnx2tf",
        action="store_false",
        dest="patch_onnx2tf",
    )
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
    parser.add_argument("--attempt-timeout-seconds", type=int, default=1800)
    parser.set_defaults(
        retry_auto_prf=True,
        patch_onnx2tf=True,
        fallback_isolated_onnx2tf2=True,
    )
    args = parser.parse_args()

    report = convert_qwen3_asr_onnx_to_litert(
        repo=args.repo,
        revision=args.revision,
        output_dir=args.output_dir,
        snapshot_dir=args.snapshot_dir,
        prefer_local=args.prefer_local,
        prefer_fp32=args.prefer_fp32,
        endpoint=args.endpoint,
        fallback_endpoint=args.fallback_endpoint,
        onnx2tf_extra_args=args.onnx2tf_extra_arg,
        retry_auto_prf=args.retry_auto_prf,
        patch_onnx2tf=args.patch_onnx2tf,
        fallback_isolated_onnx2tf2=args.fallback_isolated_onnx2tf2,
        isolated_onnx2tf2_version=args.isolated_onnx2tf2_version,
        isolated_tensorflow_version=_empty_to_none(args.isolated_tensorflow_version),
        isolated_tf_keras_version=_empty_to_none(args.isolated_tf_keras_version),
        isolated_workdir=args.isolated_workdir,
        attempt_timeout_seconds=args.attempt_timeout_seconds,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def convert_qwen3_asr_onnx_to_litert(
    *,
    repo: str,
    revision: str | None,
    output_dir: Path,
    snapshot_dir: Path | None = None,
    prefer_local: bool = False,
    prefer_fp32: bool = False,
    endpoint: str | None = None,
    fallback_endpoint: str | None = None,
    onnx2tf_extra_args: list[str] | None = None,
    retry_auto_prf: bool = True,
    patch_onnx2tf: bool = True,
    fallback_isolated_onnx2tf2: bool = True,
    isolated_onnx2tf2_version: str = "2.4.0",
    isolated_tensorflow_version: str | None = "2.19.0",
    isolated_tf_keras_version: str | None = "2.19.0",
    isolated_workdir: Path | None = None,
    attempt_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _resolve_snapshot(
        repo=repo,
        revision=revision,
        snapshot_dir=snapshot_dir,
        prefer_local=prefer_local,
        prefer_fp32=prefer_fp32,
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
    )
    component_set = _select_component_set(snapshot, prefer_fp32=prefer_fp32)
    _validate_sidecar_sources(snapshot)
    patch_results = (
        _apply_onnx2tf_patches() if patch_onnx2tf else [{"status": "disabled"}]
    )
    runners = _build_onnx2tf_runners(
        fallback_isolated_onnx2tf2=fallback_isolated_onnx2tf2,
        isolated_onnx2tf2_version=isolated_onnx2tf2_version.strip(),
        isolated_tensorflow_version=isolated_tensorflow_version,
        isolated_tf_keras_version=isolated_tf_keras_version,
        isolated_workdir=isolated_workdir.resolve()
        if isinstance(isolated_workdir, Path)
        else None,
    )

    attempts: list[dict[str, Any]] = []
    converted: dict[str, dict[str, Any]] = {}
    for index, (name, relative_path) in enumerate(
        component_set.components.items(),
        start=1,
    ):
        component_output = output_dir / "components" / name
        component_output.mkdir(parents=True, exist_ok=True)
        source_path = snapshot / relative_path
        prepared = _prepare_component_source(
            name,
            source_path,
            component_output,
            onnx2tf_extra_args or [],
        )
        conversion_source = prepared.source
        result = _convert_source(
            source=Source(repo=repo, artifact=str(conversion_source)),
            source_index=index,
            output_dir=component_output,
            revision=revision,
            prefer_local=True,
            onnx2tf_extra_args=_component_onnx2tf_extra_args(
                name,
                onnx2tf_extra_args or [],
                prepared.extra_args,
            ),
            retry_auto_prf=retry_auto_prf,
            runners=runners,
            attempt_timeout_seconds=attempt_timeout_seconds,
        )
        attempts.append({"component": name, **result})
        if not result.get("success"):
            report = _report(
                status="failed",
                repo=repo,
                revision=revision,
                snapshot=snapshot,
                quantization=component_set.name,
                output_dir=output_dir,
                components=converted,
                sidecars={},
                patch_results=patch_results,
                runners=[runner.as_dict() for runner in runners],
                attempts=attempts,
                error=_component_failure_error(name, result),
            )
            _write_report(output_dir, report)
            raise SystemExit(report["error"])
        destination = output_dir / f"{name}.tflite"
        if destination.exists():
            destination.unlink()
        shutil.copy2(Path(str(result["selected_tflite"])), destination)
        converted[name] = {
            "source": relative_path,
            "conversion_source": str(conversion_source),
            "artifact": destination.name,
            "report": result,
        }

    sidecars = _materialize_sidecars(snapshot, output_dir)
    bundle = _bundle_spec(
        repo=repo,
        revision=revision,
        quantization=component_set.name,
        components=converted,
        sidecars=sidecars,
    )
    bundle_path = output_dir / "qwen3_asr_litert_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = _report(
        status="converted",
        repo=repo,
        revision=revision,
        snapshot=snapshot,
        quantization=component_set.name,
        output_dir=output_dir,
        components=converted,
        sidecars=sidecars,
        patch_results=patch_results,
        runners=[runner.as_dict() for runner in runners],
        attempts=attempts,
        artifact=bundle_path,
    )
    _write_report(output_dir, report)
    return report


def _resolve_snapshot(
    *,
    repo: str,
    revision: str | None,
    snapshot_dir: Path | None,
    prefer_local: bool,
    prefer_fp32: bool,
    endpoint: str | None,
    fallback_endpoint: str | None,
) -> Path:
    if snapshot_dir is not None:
        return snapshot_dir.expanduser().resolve()
    path = snapshot_download_with_fallback(
        repo_id=repo,
        revision=revision,
        local_files_only=prefer_local,
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
        allow_patterns=_allow_patterns(prefer_fp32=prefer_fp32),
    )
    snapshot = Path(path).resolve()
    _ensure_required_snapshot_files(
        snapshot=snapshot,
        repo=repo,
        revision=revision,
        prefer_fp32=prefer_fp32,
        prefer_local=prefer_local,
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
    )
    return snapshot


def _ensure_required_snapshot_files(
    *,
    snapshot: Path,
    repo: str,
    revision: str | None,
    prefer_fp32: bool,
    prefer_local: bool,
    endpoint: str | None,
    fallback_endpoint: str | None,
) -> None:
    component_set = FP32_COMPONENTS if prefer_fp32 else INT4_COMPONENTS
    required = [
        *component_set.components.values(),
        *component_set.external_data,
        *REQUIRED_SIDECAR_FILES,
    ]
    for filename in required:
        target = snapshot / filename
        if target.is_file() and target.stat().st_size > 0:
            continue
        hf_hub_download_with_fallback(
            repo_id=repo,
            filename=filename,
            revision=revision,
            local_files_only=prefer_local,
            endpoint=endpoint,
            fallback_endpoint=fallback_endpoint,
            cache_dir=str(snapshot.parents[2]),
        )
    _ensure_tokenizer_snapshot_files(
        snapshot=snapshot,
        repo=repo,
        revision=revision,
        prefer_local=prefer_local,
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
    )


def _ensure_tokenizer_snapshot_files(
    *,
    snapshot: Path,
    repo: str,
    revision: str | None,
    prefer_local: bool,
    endpoint: str | None,
    fallback_endpoint: str | None,
) -> None:
    if (snapshot / "tokenizer.json").is_file():
        return
    try:
        hf_hub_download_with_fallback(
            repo_id=repo,
            filename="tokenizer.json",
            revision=revision,
            local_files_only=prefer_local,
            endpoint=endpoint,
            fallback_endpoint=fallback_endpoint,
            cache_dir=str(snapshot.parents[2]),
        )
        return
    except RuntimeError:
        pass
    missing_bpe = []
    for filename in ("vocab.json", "merges.txt"):
        target = snapshot / filename
        if target.is_file() and target.stat().st_size > 0:
            continue
        try:
            hf_hub_download_with_fallback(
                repo_id=repo,
                filename=filename,
                revision=revision,
                local_files_only=prefer_local,
                endpoint=endpoint,
                fallback_endpoint=fallback_endpoint,
                cache_dir=str(snapshot.parents[2]),
            )
        except RuntimeError:
            missing_bpe.append(filename)
    if missing_bpe:
        return


def _allow_patterns(*, prefer_fp32: bool) -> list[str]:
    patterns = set(SIDECAR_FILES)
    component_set = FP32_COMPONENTS if prefer_fp32 else INT4_COMPONENTS
    patterns.update(component_set.components.values())
    patterns.update(component_set.external_data)
    return sorted(patterns)


def _select_component_set(snapshot: Path, *, prefer_fp32: bool) -> ComponentSet:
    ordered = [FP32_COMPONENTS, INT4_COMPONENTS] if prefer_fp32 else [
        INT4_COMPONENTS,
        FP32_COMPONENTS,
    ]
    missing_by_set: dict[str, list[str]] = {}
    for component_set in ordered:
        missing = [
            relative
            for relative in [
                *component_set.components.values(),
                *component_set.external_data,
            ]
            if not (snapshot / relative).is_file()
        ]
        if not missing:
            return component_set
        missing_by_set[component_set.name] = missing
    details = "; ".join(
        f"{name}: {', '.join(missing)}" for name, missing in missing_by_set.items()
    )
    raise SystemExit(f"Missing Qwen3-ASR ONNX component files in {snapshot}: {details}")


def _component_onnx2tf_extra_args(
    component: str,
    user_args: list[str],
    generated_args: list[str] | None = None,
) -> list[str]:
    args = list(user_args)
    if component == "decoder_init":
        _append_decoder_safety_flags(args)
        if generated_args:
            if not _contains_flag(args, "-ois", "--overwrite_input_shape"):
                args.extend(_flag_group(generated_args, "-ois"))
            if not _contains_flag(args, "-prf", "--param_replacement_file"):
                args.extend(_flag_group(generated_args, "-prf"))
        if not _contains_flag(args, "-kat", "--keep_shape_absolutely_input_names"):
            args.extend(
                [
                    "-kat",
                    "input_ids",
                    "position_ids",
                    "audio_features",
                    "audio_offset",
                ]
            )
        return args
    if component == "decoder_step":
        _append_decoder_safety_flags(args)
        return args
    if component != "encoder":
        return args
    if not _contains_flag(
        args,
        "-k",
        "--keep_ncw_or_nchw_or_ncdhw_input_names",
        "-kt",
        "--keep_nwc_or_nhwc_or_ndhwc_input_names",
        "-kat",
        "--keep_shape_absolutely_input_names",
    ):
        args.extend(["-kt", "mel"])
    if generated_args and not _contains_flag(
        args,
        "-prf",
        "--param_replacement_file",
    ):
        args.extend(generated_args)
    return args


def _contains_flag(args: list[str], *flags: str) -> bool:
    return any(arg in flags for arg in args)


def _component_failure_error(component: str, result: dict[str, Any]) -> str:
    base = str(result.get("error") or f"{component} conversion failed")
    evidence = _attempt_log_excerpt(result)
    if component.startswith("decoder") and (
        "tensorflow.GraphDef" in evidence or "Invalid GraphDef" in evidence
    ):
        return (
            base
            + " Qwen3-ASR decoder conversion expanded ORT int4 MatMulNBits "
            "weights beyond TensorFlow/TFLite single-graph protobuf limits. "
            "Keep this artifact out of production LiteRT until the decoder is "
            "split into chunked pipeline stages or an Android runtime with native "
            "MatMulNBits support is selected."
        )
    if component.startswith("decoder") and "MatMulNBits OP is not yet implemented" in evidence:
        return (
            base
            + " Qwen3-ASR decoder still contains ORT MatMulNBits nodes; "
            "the decoder compatibility rewrite did not run or did not cover "
            "all nodes."
        )
    return base


def _attempt_log_excerpt(result: dict[str, Any]) -> str:
    chunks: list[str] = []
    for run in result.get("attempt_runs") or []:
        if not isinstance(run, dict):
            continue
        log_path = run.get("log_path")
        if not isinstance(log_path, str):
            continue
        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunks.append(text[-20000:])
    return "\n".join(chunks)


def _append_decoder_safety_flags(args: list[str]) -> None:
    if not _contains_flag(args, "-nuo", "--not_use_onnxsim"):
        args.append("-nuo")
    if not _contains_flag(args, "-nuonag", "--not_use_opname_auto_generate"):
        args.append("-nuonag")


def _flag_group(args: list[str], flag: str) -> list[str]:
    if flag not in args:
        return []
    start = args.index(flag)
    end = start + 1
    while end < len(args) and not args[end].startswith("-"):
        end += 1
    return args[start:end]


def _prepare_component_source(
    component: str,
    source_path: Path,
    output_dir: Path,
    user_args: list[str],
) -> PreparedComponent:
    if component == "decoder_init":
        return _prepare_decoder_init_source(source_path, output_dir, user_args)
    if component == "decoder_step":
        return _prepare_decoder_step_source(source_path, output_dir)
    if component != "encoder":
        return _prepare_external_data_source(source_path, output_dir)
    mel_frames = _encoder_mel_frames_from_args(user_args)
    output = output_dir / (
        f"{source_path.stem}.mel_{mel_frames}.static{source_path.suffix}"
    )
    _write_static_encoder_input_onnx(source_path, output, mel_frames=mel_frames)
    prf_path = output_dir / f"{source_path.stem}.mel_{mel_frames}.layout_prf.json"
    _write_encoder_layout_prf(output, prf_path, mel_frames=mel_frames)
    return PreparedComponent(source=output, extra_args=["-prf", str(prf_path)])


def _prepare_decoder_init_source(
    source_path: Path,
    output_dir: Path,
    user_args: list[str],
) -> PreparedComponent:
    prepared = _prepare_external_data_source(source_path, output_dir)
    seq_len, audio_len = _decoder_init_static_shape_from_args(user_args)
    output = output_dir / (
        f"{source_path.stem}.seq_{seq_len}.audio_{audio_len}.compatible"
        f"{source_path.suffix}"
    )
    _write_decoder_compatible_onnx(prepared.source, output)
    prf_path = output_dir / (
        f"{source_path.stem}.seq_{seq_len}.audio_{audio_len}.layout_prf.json"
    )
    _write_decoder_layout_prf(output, prf_path)
    return PreparedComponent(
        source=output,
        extra_args=[
            "-ois",
            f"input_ids:1,{seq_len}",
            f"position_ids:1,{seq_len}",
            f"audio_features:1,{audio_len},2048",
            "audio_offset:1",
            "-prf",
            str(prf_path),
        ],
    )


def _prepare_decoder_step_source(
    source_path: Path,
    output_dir: Path,
) -> PreparedComponent:
    prepared = _prepare_external_data_source(source_path, output_dir)
    output = output_dir / f"{source_path.stem}.compatible{source_path.suffix}"
    _write_decoder_compatible_onnx(prepared.source, output)
    return PreparedComponent(source=output, extra_args=[])


def _prepare_external_data_source(
    source_path: Path,
    output_dir: Path,
) -> PreparedComponent:
    import onnx

    try:
        model = onnx.load(str(source_path), load_external_data=False)
    except Exception:  # noqa: BLE001
        return PreparedComponent(source=source_path, extra_args=[])
    locations = _external_data_locations(onnx, model)
    if not locations:
        return PreparedComponent(source=source_path, extra_args=[])

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source_path.name
    _copy_regular_file(source_path.resolve(), destination)
    for location in locations:
        external_source = (source_path.parent / location).resolve()
        external_destination = output_dir / location
        external_destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_regular_file(external_source, external_destination)
    return PreparedComponent(source=destination, extra_args=[])


def _external_data_locations(onnx: Any, model: Any) -> list[Path]:
    locations: list[Path] = []
    seen: set[Path] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        location = None
        for entry in tensor.external_data:
            if entry.key == "location":
                location = Path(entry.value)
                break
        if location is None or location.is_absolute() or location in seen:
            continue
        seen.add(location)
        locations.append(location)
    return locations


def _encoder_mel_frames_from_args(args: list[str]) -> int:
    shape_flags = {"-ois", "--overwrite_input_shape"}
    for index, arg in enumerate(args):
        if arg not in shape_flags:
            continue
        for value in args[index + 1 :]:
            if value.startswith("-"):
                break
            if not value.startswith("mel:"):
                continue
            dims = value.split(":", 1)[1].split(",")
            if len(dims) != 3:
                break
            try:
                return int(dims[2])
            except ValueError:
                break
    return DEFAULT_ENCODER_MEL_FRAMES


def _decoder_init_static_shape_from_args(args: list[str]) -> tuple[int, int]:
    seq_len = DEFAULT_DECODER_INIT_SEQ_LEN
    audio_len = DEFAULT_DECODER_AUDIO_LEN
    shape_flags = {"-ois", "--overwrite_input_shape"}
    for index, arg in enumerate(args):
        if arg not in shape_flags:
            continue
        for value in args[index + 1 :]:
            if value.startswith("-"):
                break
            if value.startswith("input_ids:"):
                dims = value.split(":", 1)[1].split(",")
                if len(dims) == 2:
                    try:
                        seq_len = int(dims[1])
                    except ValueError:
                        pass
            elif value.startswith("audio_features:"):
                dims = value.split(":", 1)[1].split(",")
                if len(dims) == 3:
                    try:
                        audio_len = int(dims[1])
                    except ValueError:
                        pass
    return seq_len, audio_len


def _write_static_encoder_input_onnx(
    source: Path,
    destination: Path,
    *,
    mel_frames: int,
) -> None:
    import onnx

    model = onnx.load(str(source), load_external_data=True)
    for input_value in model.graph.input:
        if input_value.name != "mel":
            continue
        dims = input_value.type.tensor_type.shape.dim
        for dim, value in zip(dims, (1, 128, mel_frames), strict=True):
            dim.ClearField("dim_param")
            dim.dim_value = value
        break
    else:
        raise SystemExit(f"Qwen3-ASR encoder input 'mel' not found in {source}")
    _fold_encoder_mel_shape(model, mel_frames=mel_frames)
    _rewrite_encoder_custom_ops(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination))


def _fold_encoder_mel_shape(model: Any, *, mel_frames: int) -> None:
    import onnx

    for node in model.graph.node:
        if node.op_type != "Shape" or list(node.input) != ["mel"]:
            continue
        attrs = {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in node.attribute
        }
        if attrs.get("start") != 2 or attrs.get("end") != 3:
            continue
        del node.input[:]
        del node.attribute[:]
        node.op_type = "Constant"
        node.attribute.extend(
            [
                onnx.helper.make_attribute(
                    "value",
                    onnx.helper.make_tensor(
                        name=f"{node.name}_value",
                        data_type=onnx.TensorProto.INT64,
                        dims=[1],
                        vals=[mel_frames],
                    ),
                )
            ]
        )
        return


def _rewrite_encoder_custom_ops(model: Any) -> None:
    import onnx

    rewritten = []
    for node in model.graph.node:
        if node.op_type == "BiasGelu":
            rewritten.extend(_expand_bias_gelu(onnx, node))
        elif node.op_type == "SkipLayerNormalization":
            rewritten.extend(_expand_skip_layer_normalization(onnx, node))
        else:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)


def _expand_bias_gelu(onnx: Any, node: Any) -> list[Any]:
    if not node.output:
        return []
    if len(node.input) < 2:
        return [onnx.helper.make_node("Gelu", list(node.input), list(node.output), name=node.name)]
    bias_add = f"{node.name}_bias_add"
    return [
        onnx.helper.make_node(
            "Add",
            [node.input[0], node.input[1]],
            [bias_add],
            name=f"{node.name}_Add",
        ),
        onnx.helper.make_node(
            "Gelu",
            [bias_add],
            [node.output[0]],
            name=f"{node.name}_Gelu",
        ),
    ]


def _expand_skip_layer_normalization(onnx: Any, node: Any) -> list[Any]:
    if len(node.input) < 4 or not node.output:
        return [node]
    input_name, skip_name, gamma_name, beta_name = node.input[:4]
    bias_name = node.input[4] if len(node.input) > 4 and node.input[4] else None
    residual_name = (
        node.output[3]
        if len(node.output) > 3 and node.output[3]
        else f"{node.name}_input_skip_bias_sum"
    )
    first_sum = residual_name if bias_name is None else f"{node.name}_input_skip_sum"
    attrs = {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in node.attribute
    }
    nodes = [
        onnx.helper.make_node(
            "Add",
            [input_name, skip_name],
            [first_sum],
            name=f"{node.name}_SkipAdd",
        )
    ]
    if bias_name is not None:
        nodes.append(
            onnx.helper.make_node(
                "Add",
                [first_sum, bias_name],
                [residual_name],
                name=f"{node.name}_BiasAdd",
            )
        )
    nodes.append(
        onnx.helper.make_node(
            "LayerNormalization",
            [residual_name, gamma_name, beta_name],
            [node.output[0]],
            name=f"{node.name}_LayerNormalization",
            axis=int(attrs.get("axis", -1)),
            epsilon=float(attrs.get("epsilon", 1e-5)),
            stash_type=int(attrs.get("stash_type", 1)),
        )
    )
    return nodes


def _write_encoder_layout_prf(
    model_path: Path,
    destination: Path,
    *,
    mel_frames: int,
) -> None:
    import onnx

    if mel_frames % 100 != 0:
        raise SystemExit(
            "Qwen3-ASR encoder LiteRT conversion requires mel frames to be a "
            f"multiple of 100; got {mel_frames}."
        )
    model = onnx.load(str(model_path), load_external_data=False)
    operations: list[dict[str, Any]] = [
        {
            "op_name": "node_unsqueeze",
            "param_target": "op",
            "new_shape": [mel_frames // 100, 128, 100, 1],
        },
        {
            "op_name": "node_add_71",
            "param_target": "inputs",
            "param_name": "unsqueeze_1",
            "pre_process_transpose_perm": [0, 2, 1],
        },
    ]
    for node in model.graph.node:
        if node.op_type == "LayerNormalization" and node.input and node.output:
            operations.extend(
                [
                    {
                        "op_name": node.name,
                        "param_target": "inputs",
                        "param_name": node.input[0],
                        "pre_process_transpose_perm": [0, 2, 1],
                    },
                    {
                        "op_name": node.name,
                        "param_target": "outputs",
                        "param_name": node.output[0],
                        "post_process_transpose_perm": [0, 2, 1],
                    },
                ]
            )
        elif node.op_type == "Split":
            operations.append(
                {
                    "op_name": node.name,
                    "param_target": "attributes",
                    "param_name": "axis",
                    "values": 2,
                }
            )
        elif node.op_type == "Softmax":
            operations.append(
                {
                    "op_name": node.name,
                    "param_target": "attributes",
                    "param_name": "axis",
                    "values": 3,
                }
            )
        elif node.op_type == "Transpose":
            attrs = {
                attr.name: onnx.helper.get_attribute_value(attr)
                for attr in node.attribute
            }
            perm = attrs.get("perm")
            if perm:
                operations.append(
                    {
                        "op_name": node.name,
                        "param_target": "attributes",
                        "param_name": "perm",
                        "values": list(perm),
                    }
                )
    destination.write_text(
        json.dumps(
            {
                "format_version": 1,
                "operations": operations,
                "_comment": "Qwen3-ASR encoder layout overrides for onnx2tf.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_decoder_layout_prf(model_path: Path, destination: Path) -> None:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=False)
    except Exception:  # noqa: BLE001
        inferred = model
    shapes = _value_shapes(inferred)
    by_output = {output: node for node in model.graph.node for output in node.output}
    operations: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type == "MatMul" and node.input:
            producer = by_output.get(node.input[0])
            if producer is None or producer.op_type != "Softmax":
                continue
            if len(shapes.get(node.input[0], [])) != 4:
                continue
            operations.append(
                {
                    "op_name": node.name,
                    "param_target": "inputs",
                    "param_name": node.input[0],
                    "pre_process_transpose_perm": [0, 3, 1, 2],
                }
            )
    destination.write_text(
        json.dumps(
            {
                "format_version": 1,
                "operations": operations,
                "_comment": "Qwen3-ASR decoder layout overrides for onnx2tf.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _value_shapes(model: Any) -> dict[str, list[Any]]:
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    shapes: dict[str, list[Any]] = {}
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shapes[value.name] = [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            for dim in tensor_type.shape.dim
        ]
    return shapes


def _materialize_sidecars(snapshot: Path, output_dir: Path) -> dict[str, str]:
    sidecars: dict[str, str] = {}
    for name in SIDECAR_FILES:
        source = snapshot / name
        if not source.is_file():
            continue
        destination = output_dir / name
        _link_or_copy(source, destination)
        sidecars[name] = name
    _validate_sidecars(output_dir, sidecars)
    return sidecars


def _validate_sidecar_sources(snapshot: Path) -> None:
    sidecars = {name: name for name in SIDECAR_FILES if (snapshot / name).is_file()}
    _validate_sidecars(snapshot, sidecars)


def _validate_sidecars(output_dir: Path, sidecars: dict[str, str]) -> None:
    missing = []
    for required in ("config.json", "embed_tokens.bin"):
        if required not in sidecars:
            missing.append(required)
    has_fast_tokenizer = "tokenizer.json" in sidecars
    has_bpe_tokenizer = {"vocab.json", "merges.txt"}.issubset(sidecars)
    if not has_fast_tokenizer and not has_bpe_tokenizer:
        missing.append("tokenizer.json or vocab.json+merges.txt")
    if missing:
        raise SystemExit(
            "Missing Qwen3-ASR LiteRT bundle sidecars in "
            f"{output_dir}: {', '.join(missing)}"
        )


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_regular_file(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    shutil.copy2(source, destination)


def _bundle_spec(
    *,
    repo: str,
    revision: str | None,
    quantization: str,
    components: dict[str, dict[str, Any]],
    sidecars: dict[str, str],
) -> dict[str, Any]:
    return {
        "format": BUNDLE_FORMAT,
        "model_id": MODEL_ID,
        "source_model": SOURCE_MODEL,
        "engine": "litert",
        "task": "audio",
        "repo": repo,
        "revision": revision,
        "quantization": quantization,
        "runner": RUNNER,
        "components": {
            name: data["artifact"] for name, data in sorted(components.items())
        },
        "sidecars": dict(sorted(sidecars.items())),
    }


def _report(
    *,
    status: str,
    repo: str,
    revision: str | None,
    snapshot: Path,
    quantization: str,
    output_dir: Path,
    components: dict[str, dict[str, Any]],
    sidecars: dict[str, str],
    patch_results: list[dict[str, str]],
    runners: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    artifact: Path | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "format": "dart_mlx_ffi.qwen3_asr_onnx_to_litert.v1",
        "status": status,
        "model_id": MODEL_ID,
        "source_model": SOURCE_MODEL,
        "repo": repo,
        "revision": revision,
        "snapshot": str(snapshot),
        "quantization": quantization,
        "artifact": str(artifact) if artifact else None,
        "output_dir": str(output_dir),
        "components": {
            name: {
                "source": data.get("source"),
                "artifact": data.get("artifact"),
            }
            for name, data in sorted(components.items())
        },
        "sidecars": dict(sorted(sidecars.items())),
        "patches": patch_results,
        "runners": runners,
        "attempts": attempts,
        **({"error": error} if error else {}),
    }


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "qwen3_asr_litert_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


if __name__ == "__main__":
    main()
