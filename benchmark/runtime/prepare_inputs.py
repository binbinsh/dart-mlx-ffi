from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare native runtime input JSON from HF processors."
    )
    parser.add_argument("--model-id")
    parser.add_argument("--source-model")
    parser.add_argument("--task")
    parser.add_argument("--engine", choices=["mlx", "coreml", "onnx", "litert"])
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUNTIME_DIR / "hf_artifacts.yaml",
        help="Artifact catalog used to resolve source_model and fixtures.",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--embedding-query")
    parser.add_argument("--embedding-query-file", type=Path)
    parser.add_argument("--image-file", type=Path)
    parser.add_argument("--audio-file", type=Path)
    parser.add_argument("--onnx-artifact", type=Path)
    parser.add_argument("--coreml-artifact", type=Path)
    parser.add_argument("--litert-artifact", type=Path)
    parser.add_argument("--hf-cache-root", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--past-length", type=int, default=0)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--sidecar-threshold", type=int, default=4096)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = _resolve_config(args)
    source_model = args.source_model or config.get("source_model")
    if not source_model:
        raise SystemExit("Missing --source-model and no source_model in catalog.")
    task = args.task or str(config.get("task") or "text")
    out = args.out or (
        ROOT / "benchmark" / "out" / "runtime_inputs" / str(args.model_id or "model")
    ).with_suffix(".json")

    prompt = _text_fixture(
        explicit=args.prompt,
        file=args.prompt_file or _path(config.get("prompt_file")),
        fallback="Hello",
    )
    embedding_query = _text_fixture(
        explicit=args.embedding_query,
        file=args.embedding_query_file or _path(config.get("embedding_query_file")),
        fallback=prompt,
    )
    image_file = args.image_file or _path(config.get("image_file"))
    audio_file = args.audio_file or _path(config.get("audio_file"))
    _configure_hf_cache(args.hf_cache_root)

    tensors, prepare_fallback = _prepare_tensors_with_audio_fallback(
        source_model=source_model,
        task=task,
        prompt=prompt,
        embedding_query=embedding_query,
        image_file=image_file,
        audio_file=audio_file,
        trust_remote_code=args.trust_remote_code
        or bool(config.get("trust_remote_code")),
        max_length=args.max_length,
    )
    if args.onnx_artifact is not None:
        tensors = _align_onnx_inputs(
            tensors,
            args.onnx_artifact,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            past_length=args.past_length,
        )
    if args.engine == "coreml":
        if args.coreml_artifact is not None:
            tensors = _align_coreml_inputs(tensors, args.coreml_artifact)
        tensors = _coreml_tensors(tensors)
    if args.engine == "litert" and args.litert_artifact is not None:
        tensors = _align_litert_inputs(tensors, args.litert_artifact)

    resolved_out, out_path_meta = _prepare_out_path(out)
    payload = _payload(
        tensors,
        out=resolved_out,
        metadata={
            "model_id": args.model_id,
            "source_model": source_model,
            "task": task,
            "prompt": prompt,
            "image_file": str(image_file) if image_file else None,
            "audio_file": str(audio_file) if audio_file else None,
            "input_signature": _source_input_signature(
                task=task,
                prompt=prompt,
                embedding_query=embedding_query,
                image_file=image_file,
                audio_file=audio_file,
            ),
            "onnx_artifact": str(args.onnx_artifact) if args.onnx_artifact else None,
            "coreml_artifact": str(args.coreml_artifact)
            if args.coreml_artifact
            else None,
            "litert_artifact": str(args.litert_artifact)
            if args.litert_artifact
            else None,
            "prepare_fallback": prepare_fallback,
            "out_path_requested": str(out),
        },
        sidecar_threshold=args.sidecar_threshold,
    )
    if out_path_meta["mode"] == "fallback":
        payload["metadata"]["out_path_fallback"] = out_path_meta
    payload["metadata"]["out_path"] = str(resolved_out)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"out": str(resolved_out), "input_count": len(payload["inputs"])},
            indent=2,
        )
    )


def _configure_hf_cache(hf_cache_root: Path | None) -> None:
    cache_root = hf_cache_root or (ROOT / "benchmark" / ".hf_home")
    cache_root = Path(cache_root).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_cache))


def _source_input_signature(
    *,
    task: str,
    prompt: str,
    embedding_query: str,
    image_file: Path | None,
    audio_file: Path | None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = [{"name": "task", "digest": _sha256_text(task)}]
    if task == "embedding":
        items.append(
            {"name": "embedding_query", "digest": _sha256_text(embedding_query)}
        )
    else:
        items.append({"name": "prompt", "digest": _sha256_text(prompt)})
    if image_file is not None:
        items.append({"name": "image", "digest": _sha256_file(image_file)})
    if audio_file is not None:
        items.append({"name": "audio", "digest": _sha256_file(audio_file)})
    digest = hashlib.sha256(
        "\n".join(f"{item['name']}={item['digest']}" for item in items).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "format": "dart_mlx_ffi.source_input_signature.v1",
        "digest": digest,
        "items": items,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_out_path(out: Path) -> tuple[Path, dict[str, str]]:
    requested = out if out.is_absolute() else (ROOT / out)
    try:
        requested.parent.mkdir(parents=True, exist_ok=True)
        return requested, {"mode": "requested"}
    except OSError as exc:
        fallback = _fallback_out_path(requested)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback, {
            "mode": "fallback",
            "reason": str(exc),
            "requested_out_path": str(requested),
            "resolved_out_path": str(fallback),
        }


def _fallback_out_path(out: Path) -> Path:
    benchmark_out = ROOT / "benchmark" / "out"
    try:
        relative = out.relative_to(benchmark_out)
    except ValueError:
        return ROOT / "benchmark" / "out_local" / "runtime_inputs" / out.name
    return ROOT / "benchmark" / "out_local" / relative


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_id or not args.artifacts.exists():
        return {}
    data = yaml.safe_load(args.artifacts.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults") or {}
    model = (data.get("models") or {}).get(args.model_id) or {}
    return {
        **{key: value for key, value in defaults.items() if key != "platforms"},
        **{key: value for key, value in model.items() if key != "platforms"},
    }


def _prepare_tensors(
    *,
    source_model: str,
    task: str,
    prompt: str,
    embedding_query: str,
    image_file: Path | None,
    audio_file: Path | None,
    trust_remote_code: bool,
    max_length: int | None,
) -> dict[str, np.ndarray]:
    if task == "vlm" and image_file is not None:
        return _processor_tensors(
            source_model,
            text=prompt,
            image_file=image_file,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
        )
    if task in {"audio", "asr", "vad"} and audio_file is not None:
        if _uses_model_level_audio_runner(source_model):
            return _fallback_audio_waveform_tensors(audio_file, task=task)
        return _audio_tensors(
            source_model,
            audio_file=audio_file,
            trust_remote_code=trust_remote_code,
        )
    text = embedding_query if task == "embedding" else prompt
    return _tokenizer_tensors(
        source_model,
        text=text,
        trust_remote_code=trust_remote_code,
        max_length=max_length,
    )


def _prepare_tensors_with_audio_fallback(
    *,
    source_model: str,
    task: str,
    prompt: str,
    embedding_query: str,
    image_file: Path | None,
    audio_file: Path | None,
    trust_remote_code: bool,
    max_length: int | None,
) -> tuple[dict[str, np.ndarray], str | None]:
    try:
        return (
            _prepare_tensors(
                source_model=source_model,
                task=task,
                prompt=prompt,
                embedding_query=embedding_query,
                image_file=image_file,
                audio_file=audio_file,
                trust_remote_code=trust_remote_code,
                max_length=max_length,
            ),
            None,
        )
    except Exception as error:
        if task not in {"audio", "asr", "vad"} or audio_file is None:
            raise
        fallback = _fallback_audio_waveform_tensors(audio_file, task=task)
        return fallback, f"{type(error).__name__}: {error}"


def _fallback_audio_waveform_tensors(
    audio_file: Path,
    *,
    task: str,
) -> dict[str, np.ndarray]:
    import soundfile as sf

    audio, sampling_rate = sf.read(audio_file, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    if task == "vad":
        chunk_size = 512 if int(sampling_rate) >= 16000 else 256
        audio = _fixed_audio_chunk(audio, chunk_size)
    waveform = np.asarray(audio, dtype=np.float32).reshape(1, -1)
    sr = np.asarray([int(sampling_rate)], dtype=np.int64)
    # `state` is commonly required by Silero-style VAD graphs.
    state = np.zeros((2, 1, 128), dtype=np.float32)
    return {
        "input_values": waveform,
        "input": waveform,
        "audio": waveform,
        "waveform": waveform,
        "state": state,
        "sr": sr,
        "sampling_rate": sr,
    }


def _fixed_audio_chunk(audio: np.ndarray, chunk_size: int) -> np.ndarray:
    chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
    if chunk.size >= chunk_size:
        return chunk[:chunk_size]
    padded = np.zeros((chunk_size,), dtype=np.float32)
    padded[: chunk.size] = chunk
    return padded


def _uses_model_level_audio_runner(source_model: str) -> bool:
    normalized = source_model.lower().replace("_", "-")
    return normalized.endswith("qwen3-asr-1.7b") or "qwen3-asr" in normalized


def _tokenizer_tensors(
    source_model: str,
    *,
    text: str,
    trust_remote_code: bool,
    max_length: int | None,
) -> dict[str, np.ndarray]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source_model,
        trust_remote_code=trust_remote_code,
    )
    kwargs: dict[str, Any] = {"return_tensors": "np"}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    encoded = tokenizer(text, **kwargs)
    return _numpy_tensors(encoded)


def _processor_tensors(
    source_model: str,
    *,
    text: str,
    image_file: Path,
    trust_remote_code: bool,
    max_length: int | None,
) -> dict[str, np.ndarray]:
    from PIL import Image
    from transformers import AutoProcessor

    processor = _load_processor(
        source_model,
        trust_remote_code=trust_remote_code,
    )
    image = Image.open(image_file).convert("RGB")
    text = _ensure_image_placeholder(processor, text)
    kwargs: dict[str, Any] = {"return_tensors": "np"}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    try:
        encoded = processor(text=text, images=image, **kwargs)
    except ValueError as error:
        if "Only returning PyTorch tensors is currently supported" not in str(error):
            raise
        kwargs["return_tensors"] = "pt"
        encoded = processor(text=text, images=image, **kwargs)
    tensors = _numpy_tensors(encoded)
    _augment_vlm_tensors(
        tensors,
        processor=processor,
        source_model=source_model,
        trust_remote_code=trust_remote_code,
    )
    return tensors


def _audio_tensors(
    source_model: str,
    *,
    audio_file: Path,
    trust_remote_code: bool,
) -> dict[str, np.ndarray]:
    import soundfile as sf
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        source_model,
        trust_remote_code=trust_remote_code,
    )
    audio, sampling_rate = sf.read(audio_file, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    encoded = processor(audio, sampling_rate=sampling_rate, return_tensors="np")
    return _numpy_tensors(encoded)


def _ensure_image_placeholder(processor: Any, text: str) -> str:
    image_token = getattr(processor, "image_token", None) or getattr(
        getattr(processor, "tokenizer", None),
        "image_token",
        None,
    )
    if not isinstance(image_token, str) or image_token in text:
        return text
    tokenizer = getattr(processor, "tokenizer", None)
    start = _known_token(tokenizer, "<|vision_start|>") or _known_token(
        tokenizer,
        "<|IMAGE_START|>",
    )
    end = _known_token(tokenizer, "<|vision_end|>") or _known_token(
        tokenizer,
        "<|IMAGE_END|>",
    )
    if start and end:
        return f"{start}{image_token}{end}\n{text}"
    return f"{image_token}\n{text}"


def _known_token(tokenizer: Any, token: str) -> str | None:
    if tokenizer is None:
        return None
    token_id = _token_id(tokenizer, token)
    return token if token_id is not None else None


def _augment_vlm_tensors(
    tensors: dict[str, np.ndarray],
    *,
    processor: Any,
    source_model: str,
    trust_remote_code: bool,
) -> None:
    config = _load_config(source_model, trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", None)
    image_token_id = _config_int(config, "image_token_id")
    if image_token_id is None:
        image_token = getattr(processor, "image_token", None)
        if isinstance(image_token, str):
            image_token_id = _token_id(tokenizer, image_token)
    if image_token_id is not None and "input_ids" in tensors:
        input_ids = np.asarray(tensors["input_ids"])
        mask = input_ids == image_token_id
        if mask.any():
            tensors.setdefault(
                "image_token_indices",
                np.nonzero(mask[0])[0].astype(np.int64),
            )
            tensors.setdefault("mm_token_type_ids", mask.astype(np.int32))
    if "pixel_values" in tensors and np.asarray(tensors["pixel_values"]).ndim == 4:
        model_type = str(getattr(config, "model_type", ""))
        if model_type == "paddleocr_vl":
            tensors["pixel_values"] = np.asarray(tensors["pixel_values"])[None, ...]
    if (
        "position_ids" not in tensors
        and "input_ids" in tensors
        and "image_grid_thw" in tensors
    ):
        position_ids = _paddleocr_position_ids(tensors, config)
        if position_ids is not None:
            tensors["position_ids"] = position_ids


def _load_config(source_model: str, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            source_model,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        return None


def _config_int(config: Any, name: str) -> int | None:
    value = getattr(config, name, None)
    if value is None and isinstance(config, dict):
        value = config.get(name)
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _vision_config_int(config: Any, name: str, default: int) -> int:
    vision = getattr(config, "vision_config", None)
    value = getattr(vision, name, None)
    if value is None and isinstance(vision, dict):
        value = vision.get(name)
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _token_id(tokenizer: Any, token: str) -> int | None:
    if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk = getattr(tokenizer, "unk_token_id", None)
    if isinstance(token_id, int) and token_id >= 0 and token_id != unk:
        return token_id
    return None


def _load_processor(source_model: str, *, trust_remote_code: bool) -> Any:
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(
            source_model,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
    except TypeError:
        return AutoProcessor.from_pretrained(
            source_model,
            trust_remote_code=trust_remote_code,
        )


def _image_token_indices(tensors: dict[str, np.ndarray]) -> np.ndarray | None:
    if "image_token_indices" in tensors:
        return np.asarray(tensors["image_token_indices"], dtype=np.int64)
    token_types = tensors.get("mm_token_type_ids")
    if token_types is not None:
        mask = np.asarray(token_types)[0] == 1
        if mask.any():
            return np.nonzero(mask)[0].astype(np.int64)
    return None


def _paddleocr_position_ids(
    tensors: dict[str, np.ndarray],
    config: Any,
) -> np.ndarray | None:
    image_token_id = _config_int(config, "image_token_id")
    vision_start_token_id = _config_int(config, "vision_start_token_id")
    if image_token_id is None or vision_start_token_id is None:
        return None
    input_ids = np.asarray(tensors["input_ids"], dtype=np.int64)
    image_grid_thw = np.asarray(tensors["image_grid_thw"], dtype=np.int64)
    attention_mask = np.asarray(
        tensors.get("attention_mask", np.ones_like(input_ids)),
        dtype=np.int64,
    )
    spatial_merge_size = _vision_config_int(config, "spatial_merge_size", 1)
    batch, seq_len = input_ids.shape
    position_ids = np.ones((3, batch, seq_len), dtype=np.int64)
    image_index = 0
    for batch_index in range(batch):
        active = input_ids[batch_index][attention_mask[batch_index] == 1]
        tokens = active.tolist()
        pos_parts: list[np.ndarray] = []
        start = 0
        image_starts = [
            i
            for i, token in enumerate(tokens[:-1])
            if token == vision_start_token_id and tokens[i + 1] == image_token_id
        ]
        for _ in image_starts:
            try:
                image_pos = tokens.index(image_token_id, start)
            except ValueError:
                break
            if image_index >= len(image_grid_thw):
                break
            t, h, w = [int(value) for value in image_grid_thw[image_index]]
            image_index += 1
            grid_h = h // spatial_merge_size
            grid_w = w // spatial_merge_size
            text_len = image_pos - start
            offset = _next_position_offset(pos_parts)
            if text_len > 0:
                pos_parts.append(_text_positions(text_len, offset))
            pos_parts.append(_vision_positions(t, grid_h, grid_w, text_len + offset))
            start = image_pos + t * grid_h * grid_w
        if start < len(tokens):
            pos_parts.append(_text_positions(len(tokens) - start, _next_position_offset(pos_parts)))
        if not pos_parts:
            pos_parts.append(_text_positions(len(tokens), 0))
        merged = np.concatenate(pos_parts, axis=1)
        active_positions = np.nonzero(attention_mask[batch_index] == 1)[0]
        position_ids[:, batch_index, active_positions] = merged[:, : len(active_positions)]
    return position_ids


def _text_positions(length: int, offset: int) -> np.ndarray:
    values = np.arange(length, dtype=np.int64) + offset
    return np.broadcast_to(values.reshape(1, -1), (3, length)).copy()


def _vision_positions(t: int, h: int, w: int, offset: int) -> np.ndarray:
    t_index = np.zeros(t * h * w, dtype=np.int64)
    h_index = np.tile(np.repeat(np.arange(h, dtype=np.int64), w), t)
    w_index = np.tile(np.arange(w, dtype=np.int64), t * h)
    return np.stack([t_index, h_index, w_index], axis=0) + offset


def _next_position_offset(parts: list[np.ndarray]) -> int:
    if not parts:
        return 0
    return int(max(part.max() for part in parts) + 1)


def _align_onnx_inputs(
    tensors: dict[str, np.ndarray],
    onnx_artifact: Path,
    *,
    batch_size: int,
    seq_length: int,
    past_length: int,
) -> dict[str, np.ndarray]:
    if onnx_artifact.suffix.lower() == ".json":
        return _align_onnx_pipeline_inputs(
            tensors,
            onnx_artifact,
            batch_size=batch_size,
            seq_length=seq_length,
            past_length=past_length,
        )
    import onnx

    model = onnx.load(str(onnx_artifact), load_external_data=False)
    aligned: dict[str, np.ndarray] = {}
    for input_value in model.graph.input:
        name = input_value.name
        tensor_type = input_value.type.tensor_type
        dtype = _onnx_dtype(tensor_type.elem_type)
        source = _find_litert_source_tensor(tensors, name)
        if source is None:
            shape = [
                _dim_value(dim, batch_size, seq_length, past_length)
                for dim in tensor_type.shape.dim
            ]
            if not shape:
                shape = [1]
            array = np.zeros(shape, dtype=dtype)
        else:
            array = np.asarray(source)
            target_shape = _onnx_target_shape(
                tensor_type,
                source_shape=array.shape,
                batch_size=batch_size,
                seq_length=seq_length,
                past_length=past_length,
            )
            array = _align_array_to_shape(array, target_shape)
            if array.dtype != dtype:
                array = array.astype(dtype, copy=False)
        aligned[name] = array if array.ndim == 0 else np.ascontiguousarray(array)
    return aligned


def _coreml_tensors(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        array = np.asarray(value)
        if array.dtype == np.int64:
            result[name] = array.astype(np.int32)
        else:
            result[name] = array
    return result


def _align_coreml_inputs(
    tensors: dict[str, np.ndarray],
    coreml_artifact: Path,
) -> dict[str, np.ndarray]:
    sample_inputs = _coreml_sample_inputs(coreml_artifact)
    if not sample_inputs:
        return tensors
    aligned = dict(tensors)
    for name, spec in sample_inputs.items():
        target_shape = _coreml_shape(spec.get("shape"))
        target_dtype = _coreml_dtype(spec.get("dtype"))
        if name in aligned:
            array = np.asarray(aligned[name])
        else:
            array = _coreml_default_input(
                spec,
                target_shape=target_shape,
                target_dtype=target_dtype,
            )
        if target_shape:
            array = _align_array_to_shape(array, target_shape)
        if target_dtype is not None and array.dtype != target_dtype:
            array = array.astype(target_dtype, copy=False)
        aligned[name] = np.ascontiguousarray(array)
    return aligned


def _align_litert_inputs(
    tensors: dict[str, np.ndarray],
    litert_artifact: Path,
) -> dict[str, np.ndarray]:
    try:
        import tensorflow as tf
    except Exception:
        return tensors
    try:
        interpreter = tf.lite.Interpreter(model_path=str(litert_artifact))
        interpreter.allocate_tensors()
        details = interpreter.get_input_details()
    except Exception:
        return tensors
    if not isinstance(details, list) or not details:
        return tensors
    aligned: dict[str, np.ndarray] = {}
    for detail in details:
        if not isinstance(detail, dict):
            continue
        name = str(detail.get("name") or "")
        if not name:
            continue
        dtype = np.dtype(detail.get("dtype") or np.float32)
        shape = tuple(int(max(1, dim)) for dim in np.asarray(detail.get("shape", [])))
        array = _find_litert_source_tensor(tensors, name)
        if array is None:
            array = _litert_default_input(
                name=name,
                target_shape=shape,
                target_dtype=dtype,
                tensors=tensors,
            )
        if shape:
            array = _align_array_to_shape(np.asarray(array), shape)
        else:
            array = np.asarray(array)
        if array.dtype != dtype:
            array = array.astype(dtype, copy=False)
        aligned[name] = np.ascontiguousarray(array)
    return aligned or tensors


def _find_litert_source_tensor(
    tensors: dict[str, np.ndarray],
    target_name: str,
) -> np.ndarray | None:
    if target_name in tensors:
        return np.asarray(tensors[target_name])
    target_key = _canonical_litert_name(target_name)
    for name, value in tensors.items():
        if _canonical_litert_name(name) == target_key:
            return np.asarray(value)
    alias_map = {
        "input": ("input", "input_values", "audio", "waveform"),
        "input_values": ("input_values", "input", "audio", "waveform"),
        "audio": ("audio", "waveform", "input_values", "input"),
        "waveform": ("waveform", "audio", "input_values", "input"),
        "sr": ("sr", "sampling_rate"),
        "sampling_rate": ("sampling_rate", "sr"),
        "state": ("state",),
    }
    for canonical, aliases in alias_map.items():
        if target_key == canonical:
            for alias in aliases:
                for name, value in tensors.items():
                    if _canonical_litert_name(name) == alias:
                        return np.asarray(value)
    return None


def _canonical_litert_name(name: str) -> str:
    key = name.strip().lower()
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    if ":" in key:
        key = key.split(":", 1)[0]
    if key.startswith("serving_default_"):
        key = key[len("serving_default_") :]
    return key.replace("-", "_")


def _litert_default_input(
    *,
    name: str,
    target_shape: tuple[int, ...],
    target_dtype: np.dtype[Any],
    tensors: dict[str, np.ndarray],
) -> np.ndarray:
    canonical = _canonical_litert_name(name)
    if canonical in {"sr", "sampling_rate"}:
        value = np.asarray([16000], dtype=np.int64)
        return _align_array_to_shape(value, target_shape) if target_shape else value
    if canonical == "state":
        state_value = tensors.get("state")
        if state_value is not None:
            fallback = np.asarray(state_value, dtype=np.float32)
            if fallback.size > 0:
                return fallback
    if target_shape:
        return np.zeros(target_shape, dtype=target_dtype)
    return np.zeros((1,), dtype=target_dtype)


def _coreml_sample_inputs(coreml_artifact: Path) -> dict[str, dict[str, Any]]:
    sample_path = _coreml_sample_input_path(coreml_artifact)
    if sample_path is None:
        return {}
    try:
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, dict):
        raw_inputs = {
            key: value
            for key, value in payload.items()
            if isinstance(value, dict)
            and (value.get("shape") is not None or value.get("dtype") is not None)
        }
    inputs: dict[str, dict[str, Any]] = {}
    for key, value in raw_inputs.items():
        if isinstance(value, dict):
            inputs[str(key)] = dict(value)
    return inputs


def _coreml_sample_input_path(coreml_artifact: Path) -> Path | None:
    path = _path(coreml_artifact)
    if path is None:
        return None
    if path.is_dir():
        sample = path / "sample_input.json"
        if sample.exists():
            return sample
        parent_sample = path.parent / "sample_input.json"
        if parent_sample.exists():
            return parent_sample
        return None
    if path.name == "sample_input.json" and path.exists():
        return path
    sample = path.parent / "sample_input.json"
    if sample.exists():
        return sample
    return None


def _coreml_shape(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list):
        return ()
    dims: list[int] = []
    for value in raw:
        if not isinstance(value, (int, float)):
            return ()
        dims.append(int(value))
    return tuple(dims)


def _coreml_dtype(raw: Any) -> np.dtype[Any] | None:
    if not isinstance(raw, str):
        return None
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "bool": np.uint8,
    }
    dtype = mapping.get(raw)
    return np.dtype(dtype) if dtype is not None else None


def _coreml_default_input(
    spec: dict[str, Any],
    *,
    target_shape: tuple[int, ...],
    target_dtype: np.dtype[Any] | None,
) -> np.ndarray:
    values = spec.get("values")
    if isinstance(values, list):
        dtype = target_dtype or np.float32
        return np.asarray(_flatten_values(values), dtype=dtype)
    dtype = target_dtype or np.float32
    if target_shape:
        return np.zeros(target_shape, dtype=dtype)
    return np.zeros((1,), dtype=dtype)


def _align_array_to_shape(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if array.shape == target_shape:
        return array
    target_ndim = len(target_shape)
    while array.ndim > target_ndim and array.shape[0] == 1:
        array = array[0]
    while array.ndim < target_ndim:
        array = np.expand_dims(array, axis=0)
    if array.shape == target_shape:
        return array
    if array.ndim == target_ndim:
        result = np.zeros(target_shape, dtype=array.dtype)
        slices = tuple(slice(0, min(src, dst)) for src, dst in zip(array.shape, target_shape))
        result[slices] = array[slices]
        return result
    total = int(np.prod(target_shape, dtype=np.int64))
    result = np.zeros((total,), dtype=array.dtype)
    flat = array.reshape(-1)
    length = min(total, flat.size)
    if length > 0:
        result[:length] = flat[:length]
    return result.reshape(target_shape)


def _flatten_values(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if isinstance(value, list):
            result.extend(_flatten_values(value))
        else:
            result.append(value)
    return result


def _align_onnx_pipeline_inputs(
    tensors: dict[str, np.ndarray],
    pipeline_artifact: Path,
    *,
    batch_size: int,
    seq_length: int,
    past_length: int,
) -> dict[str, np.ndarray]:
    import onnx

    spec = json.loads(pipeline_artifact.read_text(encoding="utf-8"))
    aligned = dict(tensors)
    produced: set[str] = set()
    for stage in spec.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        if stage.get("op"):
            _align_pipeline_op(stage, aligned, produced)
            continue
        model_ref = stage.get("model")
        if not isinstance(model_ref, str):
            continue
        model = onnx.load(
            str(_resolve_pipeline_path(pipeline_artifact, model_ref)),
            load_external_data=False,
        )
        input_map = stage.get("inputs") if isinstance(stage.get("inputs"), dict) else {}
        for input_value in model.graph.input:
            graph_name = input_value.name
            pipeline_name = str(input_map.get(graph_name, graph_name))
            if pipeline_name in aligned or pipeline_name in produced:
                continue
            tensor_type = input_value.type.tensor_type
            dtype = _onnx_dtype(tensor_type.elem_type)
            shape = [
                _dim_value(dim, batch_size, seq_length, past_length)
                for dim in tensor_type.shape.dim
            ]
            if not shape:
                shape = [1]
            aligned[pipeline_name] = np.zeros(shape, dtype=dtype)
        output_map = stage.get("outputs") if isinstance(stage.get("outputs"), dict) else {}
        for output_value in model.graph.output:
            produced.add(str(output_map.get(output_value.name, output_value.name)))
    return aligned


def _align_pipeline_op(
    stage: dict[str, Any],
    aligned: dict[str, np.ndarray],
    produced: set[str],
) -> None:
    inputs = stage.get("inputs") if isinstance(stage.get("inputs"), dict) else {}
    outputs = stage.get("outputs") if isinstance(stage.get("outputs"), dict) else {}
    if stage.get("op") == "scatter_embeddings":
        indices_name = str(inputs.get("indices", "image_token_indices"))
        if indices_name not in aligned:
            indices = _image_token_indices(aligned)
            if indices is not None:
                aligned[indices_name] = indices
        produced.add(str(outputs.get("output", "inputs_embeds")))


def _onnx_input_names(model: Any, tensors: dict[str, np.ndarray]) -> list[str]:
    names = [value.name for value in model.graph.input]
    names.extend(name for name in tensors if name not in names)
    return names


def _dim_value(
    dim: Any,
    batch_size: int,
    seq_length: int,
    past_length: int,
) -> int:
    if dim.dim_value > 0:
        return int(dim.dim_value)
    token = str(dim.dim_param).lower()
    if "batch" in token:
        return batch_size
    if "past" in token or "cache" in token:
        return max(0, past_length)
    if "seq" in token or "time" in token or "token" in token:
        return seq_length
    return 1


def _onnx_target_shape(
    tensor_type: Any,
    *,
    source_shape: tuple[int, ...],
    batch_size: int,
    seq_length: int,
    past_length: int,
) -> tuple[int, ...]:
    dims = list(tensor_type.shape.dim)
    if not dims:
        return ()
    shape: list[int] = []
    for index, dim in enumerate(dims):
        if dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        elif index < len(source_shape):
            shape.append(int(source_shape[index]))
        else:
            shape.append(_dim_value(dim, batch_size, seq_length, past_length))
    return tuple(shape)


def _onnx_dtype(elem_type: int) -> np.dtype[Any]:
    mapping = {
        1: np.float32,
        7: np.int64,
        6: np.int32,
        2: np.uint8,
        10: np.float16,
        11: np.float64,
        9: np.bool_,
    }
    return np.dtype(mapping.get(elem_type, np.float32))


def _resolve_pipeline_path(spec_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return spec_path.parent / path


def _payload(
    tensors: dict[str, np.ndarray],
    *,
    out: Path,
    metadata: dict[str, Any],
    sidecar_threshold: int,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    sidecar_dir = out.with_suffix("")
    for name, value in tensors.items():
        array = _supported_array(np.asarray(value))
        spec = {
            "dtype": _dtype_name(array.dtype),
            "shape": list(array.shape),
        }
        if array.size > sidecar_threshold:
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{_safe_name(name)}.{spec['dtype']}.bin"
            path = sidecar_dir / filename
            path.write_bytes(np.ascontiguousarray(array).tobytes())
            spec["file"] = f"{out.stem}/{filename}"
        else:
            spec["values"] = array.tolist()
        inputs[name] = spec
    return {"metadata": metadata, "inputs": inputs}


def _numpy_tensors(batch: Any) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for name, value in dict(batch).items():
        array = _as_numpy(value)
        if array is not None:
            tensors[name] = array
    return tensors


def _as_numpy(value: Any) -> np.ndarray | None:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype == object:
        return None
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
    return array


def _supported_array(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array)
    if array.dtype in {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint8),
    }:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.integer):
        return np.ascontiguousarray(array.astype(np.int64))
    if np.issubdtype(array.dtype, np.floating):
        return np.ascontiguousarray(array.astype(np.float32))
    raise ValueError(f"Unsupported tensor dtype: {array.dtype}")


def _dtype_name(dtype: np.dtype[Any]) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.float16:
        return "float16"
    if dtype == np.float32:
        return "float32"
    if dtype == np.float64:
        return "float64"
    if dtype == np.int32:
        return "int32"
    if dtype == np.int64:
        return "int64"
    if dtype == np.uint8:
        return "uint8"
    if dtype == np.bool_:
        return "bool"
    if np.issubdtype(dtype, np.integer):
        return "int64"
    if np.issubdtype(dtype, np.floating):
        return "float32"
    raise ValueError(f"Unsupported tensor dtype: {dtype}")


def _text_fixture(
    *,
    explicit: str | None,
    file: Path | None,
    fallback: str,
) -> str:
    if explicit is not None:
        return explicit
    if file is not None and file.exists():
        return file.read_text(encoding="utf-8").strip()
    return fallback


def _path(value: Any) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


if __name__ == "__main__":
    main()
