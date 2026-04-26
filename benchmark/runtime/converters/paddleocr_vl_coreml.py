from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "PaddlePaddle/PaddleOCR-VL-1.5"
IMAGE_PLACEHOLDER = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export PaddleOCR-VL-1.5 into a Core ML component pipeline. "
            "The converter writes embed_tokens.mlmodelc, vision_encoder.mlmodelc, "
            "decoder.mlmodelc, and pipeline.json."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--image-file", type=Path, default=ROOT / "benchmark/runtime/fixtures/image.png")
    parser.add_argument("--sequence-length", type=int, default=160)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--full-sequence-logits", action="store_true")
    parser.add_argument("--compute-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--minimum-target", choices=["ios17", "macos14"], default="ios17")
    parser.add_argument("--keep-packages", action="store_true")
    args = parser.parse_args()

    export_paddleocr_vl_coreml(
        model_id=args.model,
        output_dir=args.output_dir,
        prompt=_prompt(args.prompt, args.prompt_file),
        image_file=args.image_file,
        sequence_length=args.sequence_length,
        trust_remote_code=args.trust_remote_code,
        full_sequence_logits=args.full_sequence_logits,
        compute_precision=args.compute_precision,
        torch_dtype=args.torch_dtype,
        minimum_target=args.minimum_target,
        keep_packages=args.keep_packages,
    )


def export_paddleocr_vl_coreml(
    *,
    model_id: str,
    output_dir: Path,
    prompt: str,
    image_file: Path,
    sequence_length: int,
    trust_remote_code: bool,
    full_sequence_logits: bool,
    compute_precision: str,
    torch_dtype: str,
    minimum_target: str,
    keep_packages: bool,
) -> dict[str, Any]:
    import coremltools as ct
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    _patch_transformers_mask_alias()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=_torch_dtype(torch_dtype),
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()
    model.requires_grad_(False)

    batch = _processor_batch(
        processor,
        prompt=prompt,
        image_file=image_file,
        sequence_length=sequence_length,
    )
    text_inputs = _text_inputs(batch)
    vision_inputs = _vision_inputs(batch)
    decoder_inputs = _decoder_inputs(model, batch, sequence_length=sequence_length)

    target = _minimum_target(ct, minimum_target)
    precision = _compute_precision(ct, compute_precision)

    package_root = output_dir / "_mlpackages"
    package_root.mkdir(parents=True, exist_ok=True)
    embed_package_path = package_root / "embed_tokens.mlpackage"
    vision_package_path = package_root / "vision_encoder.mlpackage"
    decoder_package_path = package_root / "decoder.mlpackage"
    embed_path = output_dir / "embed_tokens.mlmodelc"
    vision_path = output_dir / "vision_encoder.mlmodelc"
    decoder_path = output_dir / "decoder.mlmodelc"

    _convert_component(
        ct=ct,
        module=TextEmbeddingWrapper(model),
        example_inputs=(text_inputs["input_ids"],),
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=tuple(text_inputs["input_ids"].shape),
                dtype=np.int32,
            )
        ],
        outputs=[ct.TensorType(name="text_inputs_embeds")],
        package_path=embed_package_path,
        compiled_path=embed_path,
        target=target,
        precision=precision,
    )

    _convert_component(
        ct=ct,
        module=VisionProjectorWrapper(model, vision_inputs["image_grid_thw"]),
        example_inputs=(vision_inputs["pixel_values"],),
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=tuple(vision_inputs["pixel_values"].shape),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="image_embeds")],
        package_path=vision_package_path,
        compiled_path=vision_path,
        target=target,
        precision=precision,
    )

    _convert_component(
        ct=ct,
        module=DecoderWrapper(
            model,
            position_ids=decoder_inputs["position_ids"],
            sequence_length=sequence_length,
            full_sequence_logits=full_sequence_logits,
        ),
        example_inputs=(
            decoder_inputs["inputs_embeds"],
            decoder_inputs["attention_mask"],
            decoder_inputs["position_ids"],
        ),
        inputs=[
            ct.TensorType(
                name="inputs_embeds",
                shape=tuple(decoder_inputs["inputs_embeds"].shape),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=tuple(decoder_inputs["attention_mask"].shape),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="position_ids",
                shape=tuple(decoder_inputs["position_ids"].shape),
                dtype=np.int32,
            ),
        ],
        outputs=[ct.TensorType(name="logits")],
        package_path=decoder_package_path,
        compiled_path=decoder_path,
        target=target,
        precision=precision,
    )

    image_indices = _image_token_indices(batch, int(model.config.image_token_id))
    pipeline = _pipeline_spec(
        embed_path=embed_path,
        vision_path=vision_path,
        decoder_path=decoder_path,
        outputs=["logits"],
    )
    pipeline_path = output_dir / "pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")

    sample_input_path = output_dir / "sample_input.json"
    sample_input_path.write_text(
        json.dumps(
            _sample_input_payload(
                input_ids=text_inputs["input_ids"],
                pixel_values=vision_inputs["pixel_values"],
                attention_mask=decoder_inputs["attention_mask"],
                position_ids=decoder_inputs["position_ids"],
                image_token_indices=image_indices,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "format": "dart_inference.coreml_conversion_report.v1",
        "source_model": model_id,
        "artifact": str(pipeline_path),
        "components": {
            "embed_tokens": str(embed_path),
            "vision_encoder": str(vision_path),
            "decoder": str(decoder_path),
        },
        "sample_input": str(sample_input_path),
        "sequence_length": sequence_length,
        "full_sequence_logits": full_sequence_logits,
        "image_grid_thw": vision_inputs["image_grid_thw"].tolist(),
        "image_token_count": int(image_indices.numel()),
        "minimum_target": minimum_target,
        "compute_precision": compute_precision,
        "torch_dtype": torch_dtype,
        "source_packages_retained": keep_packages,
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    if not keep_packages:
        shutil.rmtree(package_root)
    print(json.dumps(report, indent=2))
    return report


class TextEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.embedding = model.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids.long())


class VisionProjectorWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, image_grid_thw: torch.Tensor) -> None:
        super().__init__()
        embeddings = model.visual.vision_model.embeddings
        self.patch_embedding = embeddings.patch_embedding
        self.layers = model.visual.vision_model.encoder.layers
        self.post_layernorm = model.visual.vision_model.post_layernorm
        self.projector = model.mlp_AR
        grid = tuple(int(value) for value in image_grid_thw[0].tolist())
        self.image_grid = [grid]
        t, h, w = grid
        m1, m2 = self.projector.merge_kernel_size
        self.merge_kernel_h = int(m1)
        self.merge_kernel_w = int(m2)
        self.merged_t = t
        self.merged_h = h // self.merge_kernel_h
        self.merged_w = w // self.merge_kernel_w
        self.sequence_length = t * h * w
        hidden_size = int(model.config.vision_config.hidden_size)
        first_attn = self.layers[0].self_attn
        self.embed_dim = int(first_attn.embed_dim)
        self.num_heads = int(first_attn.num_heads)
        self.head_dim = int(first_attn.head_dim)
        self.half_head_dim = self.head_dim // 2
        with torch.no_grad():
            position_embedding = embeddings.interpolate_pos_encoding(
                torch.zeros(t * h * w, hidden_size),
                h,
                w,
                True,
            ).squeeze(0).repeat(t, 1)
        image_pids = torch.arange(t * h * w, dtype=torch.long) % (h * w)
        height_position_ids = image_pids // w
        width_position_ids = image_pids % w
        cu_seqlens = torch.tensor([0, t * h * w], dtype=torch.int32)
        with torch.no_grad():
            pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
            max_grid_size = int(pids.max().item() + 1)
            rope_emb_max_grid = model.visual.vision_model.encoder.rotary_pos_emb(
                max_grid_size
            )
            rope_emb = rope_emb_max_grid[pids].flatten(1).repeat(1, 2)
            rope_cos = rope_emb.cos()
            rope_sin = rope_emb.sin()
        self.register_buffer(
            "position_embedding",
            position_embedding,
            persistent=False,
        )
        self.register_buffer(
            "height_position_ids",
            height_position_ids,
            persistent=False,
        )
        self.register_buffer(
            "width_position_ids",
            width_position_ids,
            persistent=False,
        )
        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(target_dtype))
        hidden_states = patch_embeds.flatten(-2).squeeze(-1)
        hidden_states = hidden_states.reshape(1, self.sequence_length, -1)
        hidden_states = hidden_states + self.position_embedding.unsqueeze(0)
        rope_emb = (self.rope_cos, self.rope_sin)
        for layer in self.layers:
            hidden_states = self._encoder_layer(layer, hidden_states, rope_emb)
        image_features = self.post_layernorm(hidden_states).squeeze(0)
        return self._project(image_features)

    def _project(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = self.projector.pre_norm(image_features)
        image_features = image_features.reshape(
            self.merged_t,
            self.merged_h,
            self.merge_kernel_h,
            self.merged_w,
            self.merge_kernel_w,
            self.embed_dim,
        )
        image_features = image_features.permute(0, 1, 3, 2, 4, 5).reshape(
            self.merged_t * self.merged_h * self.merged_w,
            self.merge_kernel_h * self.merge_kernel_w * self.embed_dim,
        )
        hidden_states = self.projector.linear_1(image_features)
        hidden_states = self.projector.act(hidden_states)
        return self.projector.linear_2(hidden_states)

    def _encoder_layer(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer.layer_norm1(hidden_states)
        hidden_states = self._attention(layer.self_attn, hidden_states, rope_emb)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.layer_norm2(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return residual + hidden_states

    def _attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        queries = attention.q_proj(hidden_states).reshape(
            1,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        )
        keys = attention.k_proj(hidden_states).reshape(
            1,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        )
        values = attention.v_proj(hidden_states).reshape(
            1,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        )
        queries, keys = self._apply_rotary(queries, keys, rope_emb)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) * attention.scale
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, values)
        output = output.transpose(1, 2).reshape(
            1,
            self.sequence_length,
            self.embed_dim,
        )
        return attention.out_proj(output)

    def _apply_rotary(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        rope_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = rope_emb
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        q_float = queries.float()
        k_float = keys.float()
        q_embed = (q_float * cos) + (self._rotate_half(q_float) * sin)
        k_embed = (k_float * cos) + (self._rotate_half(k_float) * sin)
        return q_embed.to(queries.dtype), k_embed.to(keys.dtype)

    def _rotate_half(self, value: torch.Tensor) -> torch.Tensor:
        first = value[..., : self.half_head_dim]
        second = value[..., self.half_head_dim :]
        return torch.cat((-second, first), dim=-1)


class DecoderWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        position_ids: torch.Tensor,
        sequence_length: int,
        full_sequence_logits: bool,
    ) -> None:
        super().__init__()
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.sequence_length = sequence_length
        self.full_sequence_logits = full_sequence_logits
        self.hidden_size = int(model.config.hidden_size)
        self.num_heads = int(model.config.num_attention_heads)
        self.num_key_value_heads = int(model.config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(model.config.head_dim)
        self.mrope_section = [int(value) * 2 for value in model.config.rope_scaling["mrope_section"]]
        self.half_head_dim = self.head_dim // 2
        causal = torch.full(
            (sequence_length, sequence_length),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
        )
        causal = torch.triu(causal, diagonal=1).reshape(
            1,
            1,
            sequence_length,
            sequence_length,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, sequence_length, self.hidden_size)
            cos, sin = model.model.rotary_emb(dummy, position_ids.long())
            cos = self._select_mrope(cos).unsqueeze(1)
            sin = self._select_mrope(sin).unsqueeze(1)
        self.register_buffer("causal_mask", causal, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = self.lm_head.weight.dtype
        hidden_states = inputs_embeds.to(target_dtype)
        mask = self._attention_mask(attention_mask)
        for layer in self.layers:
            hidden_states = self._decoder_layer(layer, hidden_states, mask)
        hidden_states = self.norm(hidden_states)
        if not self.full_sequence_logits:
            hidden_states = hidden_states[:, -1:, :]
        return self.lm_head(hidden_states)

    def _decoder_layer(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = self._attention(layer.self_attn, hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return residual + hidden_states

    def _attention(
        self,
        attention: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_states = attention.q_proj(hidden_states).reshape(
            1,
            self.sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_states = attention.k_proj(hidden_states).reshape(
            1,
            self.sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = attention.v_proj(hidden_states).reshape(
            1,
            self.sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        query_states, key_states = self._apply_mrope(query_states, key_states)
        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)
        scores = torch.matmul(query_states, key_states.transpose(2, 3)) * attention.scaling
        scores = scores + attention_mask
        weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(
            1,
            self.sequence_length,
            self.num_heads * self.head_dim,
        )
        return attention.o_proj(attn_output)

    def _attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        padding = (1.0 - attention_mask.to(torch.float32)).reshape(
            1,
            1,
            1,
            self.sequence_length,
        )
        padding = padding * torch.finfo(torch.float32).min
        return self.causal_mask + padding

    def _repeat_kv(self, value: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return value
        value = value[:, :, None, :, :].expand(
            1,
            self.num_key_value_heads,
            self.num_key_value_groups,
            self.sequence_length,
            self.head_dim,
        )
        return value.reshape(
            1,
            self.num_heads,
            self.sequence_length,
            self.head_dim,
        )

    def _apply_mrope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_float = query_states.float()
        k_float = key_states.float()
        q_embed = (q_float * self.rope_cos) + (self._rotate_half(q_float) * self.rope_sin)
        k_embed = (k_float * self.rope_cos) + (self._rotate_half(k_float) * self.rope_sin)
        return q_embed.to(query_states.dtype), k_embed.to(key_states.dtype)

    def _rotate_half(self, value: torch.Tensor) -> torch.Tensor:
        first = value[..., : self.half_head_dim]
        second = value[..., self.half_head_dim :]
        return torch.cat((-second, first), dim=-1)

    def _select_mrope(self, value: torch.Tensor) -> torch.Tensor:
        pieces = value.split(self.mrope_section, dim=-1)
        selected = [piece[index % 3] for index, piece in enumerate(pieces)]
        return torch.cat(selected, dim=-1)


def _processor_batch(
    processor: Any,
    *,
    prompt: str,
    image_file: Path,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    from PIL import Image

    image = Image.open(image_file).convert("RGB")
    encoded = processor(
        text=_paddle_prompt(prompt),
        images=image,
        return_tensors="pt",
    )
    return _pad_batch({key: value for key, value in encoded.items()}, sequence_length)


def _pad_batch(
    batch: dict[str, torch.Tensor],
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    current = int(batch["input_ids"].shape[1])
    if current > sequence_length:
        raise SystemExit(
            f"PaddleOCR-VL sample sequence length {current} exceeds "
            f"--sequence-length {sequence_length}."
        )
    pad = sequence_length - current
    if pad == 0:
        return batch
    input_ids = torch.nn.functional.pad(batch["input_ids"], (0, pad), value=0)
    attention_mask = torch.nn.functional.pad(batch["attention_mask"], (0, pad), value=0)
    result = dict(batch)
    result["input_ids"] = input_ids
    result["attention_mask"] = attention_mask
    return result


def _text_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"input_ids": batch["input_ids"].to(torch.int32)}


def _vision_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": batch["pixel_values"].to(torch.float32),
        "image_grid_thw": batch["image_grid_thw"].to(torch.int64),
    }


def _decoder_inputs(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    input_ids = batch["input_ids"].long()
    attention_mask = batch["attention_mask"].long()
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=batch["image_grid_thw"].long(),
        attention_mask=attention_mask,
    )
    position_ids = position_ids[:, :, :sequence_length]
    with torch.no_grad():
        inputs_embeds = model.get_input_embeddings()(input_ids)
    return {
        "inputs_embeds": inputs_embeds.to(torch.float32),
        "attention_mask": attention_mask.to(torch.int32),
        "position_ids": position_ids.to(torch.int32),
    }


def _image_token_indices(batch: dict[str, torch.Tensor], image_token_id: int) -> torch.Tensor:
    ids = batch["input_ids"][0]
    return torch.nonzero(ids == image_token_id, as_tuple=False).flatten().to(torch.int32)


def _convert_component(
    *,
    ct: Any,
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    inputs: list[Any],
    outputs: list[Any],
    package_path: Path,
    compiled_path: Path,
    target: Any,
    precision: Any,
) -> None:
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, strict=False)
        mlmodel = ct.convert(
            traced,
            source="pytorch",
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=target,
            compute_precision=precision,
        )
    if package_path.exists():
        shutil.rmtree(package_path)
    if compiled_path.exists():
        shutil.rmtree(compiled_path)
    mlmodel.save(package_path)
    _compile_mlpackage(package_path, compiled_path)


def _compile_mlpackage(package_path: Path, compiled_path: Path) -> None:
    result = subprocess.run(
        [
            "xcrun",
            "coremlcompiler",
            "compile",
            str(package_path),
            str(compiled_path.parent),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            "coremlcompiler failed for "
            f"{package_path}:\n{result.stdout}\n{result.stderr}"
        )
    if not compiled_path.exists():
        raise SystemExit(
            f"coremlcompiler did not create expected model: {compiled_path}\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _pipeline_spec(
    *,
    embed_path: Path,
    vision_path: Path,
    decoder_path: Path,
    outputs: list[str],
) -> dict[str, Any]:
    return {
        "format": "dart_inference.coreml_pipeline.v1",
        "stages": [
            {
                "name": "embed_tokens",
                "model": embed_path.name,
                "outputs": {"text_inputs_embeds": "text_inputs_embeds"},
            },
            {
                "name": "vision_encoder",
                "model": vision_path.name,
                "outputs": {"image_embeds": "image_embeds"},
            },
            {
                "name": "merge_image_embeds",
                "op": "scatter_embeddings",
                "inputs": {
                    "base": "text_inputs_embeds",
                    "updates": "image_embeds",
                    "indices": "image_token_indices",
                },
                "outputs": {"output": "inputs_embeds"},
            },
            {
                "name": "decoder",
                "model": decoder_path.name,
                "inputs": {
                    "inputs_embeds": "inputs_embeds",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                },
                "outputs": {"logits": "logits"},
            },
        ],
        "outputs": outputs,
    }


def _sample_input_payload(
    *,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    image_token_indices: torch.Tensor,
) -> dict[str, Any]:
    return {
        "metadata": {
            "source": "benchmark/runtime/converters/paddleocr_vl_coreml.py",
            "note": "Sample input matches the fixed Core ML conversion shapes.",
        },
        "inputs": {
            "input_ids": _tensor_json(input_ids),
            "pixel_values": _tensor_json(pixel_values),
            "attention_mask": _tensor_json(attention_mask),
            "position_ids": _tensor_json(position_ids),
            "image_token_indices": _tensor_json(image_token_indices),
        },
    }


def _tensor_json(tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu().numpy()
    dtype = str(array.dtype)
    if dtype == "int64":
        array = array.astype(np.int32)
        dtype = "int32"
    if array.size > 4096:
        encoded = np.ascontiguousarray(array).tobytes()
        import base64

        return {
            "dtype": dtype,
            "shape": list(array.shape),
            "base64": base64.b64encode(encoded).decode("ascii"),
        }
    return {"dtype": dtype, "shape": list(array.shape), "values": array.tolist()}


def _minimum_target(ct: Any, value: str) -> Any:
    if value == "macos14":
        return ct.target.macOS14
    return ct.target.iOS17


def _compute_precision(ct: Any, value: str) -> Any:
    if value == "float32":
        return ct.precision.FLOAT32
    return ct.precision.FLOAT16


def _torch_dtype(value: str) -> torch.dtype:
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _patch_transformers_mask_alias() -> None:
    try:
        from transformers import masking_utils
    except ImportError:
        return
    create_causal_mask = getattr(masking_utils, "create_causal_mask", None)
    if create_causal_mask is None or getattr(create_causal_mask, "_dinf_alias", False):
        return

    def create_causal_mask_compat(*args: Any, **kwargs: Any) -> Any:
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return create_causal_mask(*args, **kwargs)

    create_causal_mask_compat._dinf_alias = True  # type: ignore[attr-defined]
    masking_utils.create_causal_mask = create_causal_mask_compat


def _paddle_prompt(prompt: str) -> str:
    if "<|IMAGE_PLACEHOLDER|>" in prompt:
        return prompt
    return f"{IMAGE_PLACEHOLDER}\n{prompt}"


def _prompt(value: str | None, path: Path | None) -> str:
    if value is not None:
        return value
    if path is not None and path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "OCR this image."


if __name__ == "__main__":
    main()
