#!/usr/bin/env python3
"""
Dump intermediate values from PaddleOCR-VL-1.5 (8-bit) via mlx-vlm for
Dart parity checking.

Usage:
    source /tmp/mlx-test-env/bin/activate
    python tool/dump_paddle_v15_reference.py

Writes .npy files to /tmp/paddle_v15_ref/ for the test image
    /Users/binbinsh/Projects/Personal/chef-de-mise/test-ocr-input.jpg
"""

import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

# ─── paths ───────────────────────────────────────────────────────────────────
MODEL_PATH = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/"
        "models--mlx-community--PaddleOCR-VL-1.5-8bit/"
        "snapshots/37d4c85284434b6e6fd4c03f8b719b1aefaa013c"
    )
)
IMAGE_PATH = Path("/Users/binbinsh/Projects/Personal/chef-de-mise/test-ocr-input.jpg")
OUT_DIR = Path("/tmp/paddle_v15_ref")
OUT_DIR.mkdir(exist_ok=True)

# ─── load model via mlx-vlm ─────────────────────────────────────────────────
from mlx_vlm import load as vlm_load
from mlx_vlm.models.paddleocr_vl.processing_paddleocr_vl import (
    ImageProcessor,
    PaddleOCRVLProcessor,
    smart_resize,
)

print("Loading model …")
model, processor = vlm_load(str(MODEL_PATH))
mx.eval(model.parameters())
print("Model loaded.")

config = json.loads((MODEL_PATH / "config.json").read_text())
vision_cfg = config.get("vision_config", {})
text_cfg = config.get("text_config", {})
patch_size = vision_cfg.get("patch_size", 14)
merge_size = vision_cfg.get("spatial_merge_size", 2)
image_token_id = config.get("image_token_id", 100295)
vision_start_token_id = config.get("vision_start_token_id", 101305)
vision_end_token_id = config.get("vision_end_token_id", 101306)

# ─── preprocess image ────────────────────────────────────────────────────────
print(f"Image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = image.size
print(f"  Original size: {orig_w}x{orig_h}")

# Replicate mlx-vlm preprocessing
ip = processor.image_processor
resized_h, resized_w = smart_resize(
    orig_h,
    orig_w,
    factor=patch_size * merge_size,
    min_pixels=ip.min_pixels,
    max_pixels=ip.max_pixels,
)
print(f"  Resized: {resized_w}x{resized_h}")

grid_h = resized_h // patch_size
grid_w = resized_w // patch_size
print(f"  Grid: {grid_h}x{grid_w} = {grid_h * grid_w} patches")

merged_h = grid_h // merge_size
merged_w = grid_w // merge_size
num_merged = merged_h * merged_w
print(f"  Merged: {merged_h}x{merged_w} = {num_merged} tokens")

# Build the prompt text
prompt_text = (
    "<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|>"
    "<|IMAGE_END|>Extract all text from this image.\nAssistant:\n"
)
print(f"  Prompt: {prompt_text!r}")

# Process through the processor
inputs = processor(images=[image], text=[prompt_text], return_tensors="np")
input_ids = mx.array(inputs["input_ids"])
pixel_values = mx.array(inputs["pixel_values"])
image_grid_thw = mx.array(inputs["image_grid_thw"])

print(f"  input_ids shape: {input_ids.shape}")
print(f"  pixel_values shape: {pixel_values.shape}")
print(f"  image_grid_thw: {image_grid_thw.tolist()}")

# Save input_ids
np.save(OUT_DIR / "input_ids.npy", np.array(input_ids.tolist(), dtype=np.int32))

# Save the preprocessed pixel values for Dart image loading parity
pv_np = np.array(pixel_values.tolist())
np.save(OUT_DIR / "pixel_values.npy", pv_np.astype(np.float32))

# Also save the resized+normalised image in NHWC format for Dart Conv2d input.
# Dart expects [1, H, W, C] (float32, already normalised).
resized_img = image.resize((resized_w, resized_h), Image.BICUBIC)
img_arr = np.array(resized_img, dtype=np.float32) / 255.0
# Apply ImageNet normalisation that mlx-vlm's ImageProcessor uses
mean = np.array(ip.image_mean, dtype=np.float32)
std = np.array(ip.image_std, dtype=np.float32)
img_arr = (img_arr - mean) / std
# NHWC: [1, H, W, C]
img_nhwc = img_arr[np.newaxis, :, :, :]
print(f"  NHWC image: shape={img_nhwc.shape}")
np.save(OUT_DIR / "image_nhwc.npy", img_nhwc)

# ─── step 1: vision embeddings (patch embed + position interpolation) ────────
print("\n=== Vision Embeddings ===")
vis = model.visual

# Call embeddings
hidden_after_embed = vis.embeddings(pixel_values, image_grid_thw)
mx.eval(hidden_after_embed)
print(
    f"  After embeddings: shape={hidden_after_embed.shape} dtype={hidden_after_embed.dtype}"
)
np.save(
    OUT_DIR / "vision_embeddings.npy",
    np.array(hidden_after_embed.tolist(), dtype=np.float32),
)

# ─── step 2: vision rotary pos embedding ─────────────────────────────────────
print("\n=== Vision Rotary Pos Embedding ===")
rotary_pos_emb = vis.rot_pos_emb(image_grid_thw)
mx.eval(rotary_pos_emb)
print(f"  Vision rotary: shape={rotary_pos_emb.shape}")
np.save(
    OUT_DIR / "vision_rotary_pos_emb.npy",
    np.array(rotary_pos_emb.tolist(), dtype=np.float32),
)

# ─── step 3: after first vision layer ────────────────────────────────────────
print("\n=== Vision Layer 0 ===")
# Build cu_seqlens for attention
batch_size = image_grid_thw.shape[0]
cu_seqlens = []
for i in range(batch_size):
    seq_len = image_grid_thw[i, 1] * image_grid_thw[i, 2]
    cu_seqlens.append(mx.repeat(seq_len, image_grid_thw[i, 0]))
cu_seqlens = mx.concatenate(cu_seqlens)
cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

hidden = hidden_after_embed
for i in range(len(vis.layers)):
    hidden = vis.layers[i](hidden, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
    mx.eval(hidden)
    if i == 0:
        print(f"  After layer 0: shape={hidden.shape}")
        np.save(
            OUT_DIR / "vision_after_layer0.npy",
            np.array(hidden.tolist(), dtype=np.float32),
        )
    if i == 2:
        np.save(
            OUT_DIR / "vision_after_layer2.npy",
            np.array(hidden.tolist(), dtype=np.float32),
        )

print(f"  After all {len(vis.layers)} layers: shape={hidden.shape}")
np.save(
    OUT_DIR / "vision_after_all_layers.npy",
    np.array(hidden.tolist(), dtype=np.float32),
)

# Post layernorm
hidden = vis.post_layernorm(hidden)
mx.eval(hidden)
print(f"  After post_layernorm: shape={hidden.shape}")
np.save(
    OUT_DIR / "vision_post_layernorm.npy",
    np.array(hidden.tolist(), dtype=np.float32),
)

# Projector
projected = vis.projector(hidden, image_grid_thw)
mx.eval(projected)
print(f"  After projector: shape={projected.shape}")
np.save(
    OUT_DIR / "vision_projected.npy",
    np.array(projected.tolist(), dtype=np.float32),
)

# ─── step 4: full vision encode (end-to-end) ────────────────────────────────
print("\n=== Full Vision Encode (end-to-end) ===")
full_hidden = vis(pixel_values, image_grid_thw, output_hidden_states=False)
mx.eval(full_hidden)
print(f"  Full encode: shape={full_hidden.shape}")
# Verify it matches step-by-step
diff = mx.abs(full_hidden - projected).max().item()
print(f"  Max diff vs step-by-step: {diff}")

# ─── step 5: text embeddings ────────────────────────────────────────────────
print("\n=== Text Embeddings ===")
text_embeds = model.language_model.model.embed_tokens(input_ids)
mx.eval(text_embeds)
print(f"  Text embeddings: shape={text_embeds.shape}")

# ─── step 6: merged embeddings ──────────────────────────────────────────────
print("\n=== Merged Embeddings ===")
merged_embeds = model.merge_input_ids_with_image_features(
    image_token_id,
    full_hidden,
    text_embeds,
    input_ids,
)
mx.eval(merged_embeds)
print(f"  Merged embeddings: shape={merged_embeds.shape}")
np.save(
    OUT_DIR / "merged_embeddings_first16.npy",
    np.array(merged_embeds[0, :16, :].tolist(), dtype=np.float32),
)
np.save(
    OUT_DIR / "merged_embeddings_last16.npy",
    np.array(merged_embeds[0, -16:, :].tolist(), dtype=np.float32),
)

# ─── step 7: position IDs (M-RoPE) ──────────────────────────────────────────
print("\n=== Position IDs (M-RoPE) ===")
position_ids, rope_deltas = model.language_model.get_rope_index(
    input_ids, image_grid_thw, None, None
)
mx.eval(position_ids)
print(f"  Position IDs: shape={position_ids.shape}")
print(f"  rope_deltas: {rope_deltas}")
np.save(
    OUT_DIR / "position_ids.npy",
    np.array(position_ids.tolist(), dtype=np.int32),
)

# ─── step 8: M-RoPE cos/sin ─────────────────────────────────────────────────
print("\n=== M-RoPE cos/sin ===")
rotary_emb = model.language_model.model.rotary_emb
cos_out, sin_out = rotary_emb(merged_embeds, position_ids)
mx.eval(cos_out)
mx.eval(sin_out)
print(f"  cos shape: {cos_out.shape}")
print(f"  sin shape: {sin_out.shape}")
np.save(
    OUT_DIR / "mrope_cos.npy",
    np.array(cos_out.tolist(), dtype=np.float32),
)
np.save(
    OUT_DIR / "mrope_sin.npy",
    np.array(sin_out.tolist(), dtype=np.float32),
)

# ─── step 9: first LM layer output ──────────────────────────────────────────
print("\n=== First LM Layer ===")
from mlx_vlm.models.base import create_attention_mask

mask = create_attention_mask(merged_embeds, None)
position_embeddings = (cos_out, sin_out)
h = merged_embeds
first_layer = model.language_model.model.layers[0]
h_after_first = first_layer(h, mask, None, position_embeddings)
mx.eval(h_after_first)
print(f"  After LM layer 0: shape={h_after_first.shape}")
np.save(
    OUT_DIR / "lm_after_layer0_first16.npy",
    np.array(h_after_first[0, :16, :].tolist(), dtype=np.float32),
)

# ─── step 10: full forward (first logits) ───────────────────────────────────
print("\n=== Full Forward (logits) ===")

# Reset model state
model.language_model._position_ids = None
model.language_model._rope_deltas = None

logits = model(
    input_ids,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
)
mx.eval(logits)
logits_arr = logits.logits if hasattr(logits, "logits") else logits
print(f"  Logits shape: {logits_arr.shape}")

# Save only the last-position logits (for next-token prediction)
last_logits = logits_arr[0, -1, :]
mx.eval(last_logits)
print(f"  Last logits shape: {last_logits.shape}")
np.save(
    OUT_DIR / "last_logits.npy",
    np.array(last_logits.tolist(), dtype=np.float32),
)

# Top-5 predictions
top5_indices = mx.argsort(last_logits)[-5:][::-1]
mx.eval(top5_indices)
top5_ids = top5_indices.tolist()
print(f"  Top-5 token IDs: {top5_ids}")

# Decode top-5
for tid in top5_ids:
    token_str = processor.tokenizer.decode([tid])
    prob = mx.softmax(last_logits, axis=-1)[tid].item()
    print(f"    {tid}: {token_str!r} (prob={prob:.4f})")

# Save the greedy next token
greedy_token = top5_ids[0]
print(
    f"\n  Greedy first token: {greedy_token} = {processor.tokenizer.decode([greedy_token])!r}"
)

# ─── step 11: full generation ────────────────────────────────────────────────
print("\n=== Full Generation ===")
from mlx_vlm import generate as vlm_generate

# Reset state
model.language_model._position_ids = None
model.language_model._rope_deltas = None

prompt = "<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>Extract all text from this image.\nAssistant:\n"
t0 = time.time()
output = vlm_generate(
    model,
    processor,
    image=str(IMAGE_PATH),
    prompt=prompt,
    max_tokens=1024,
    verbose=False,
)
t1 = time.time()
print(f"  Generation time: {t1 - t0:.2f}s")
# Handle both str and GenerationResult
if hasattr(output, "text"):
    output_text = output.text
elif isinstance(output, str):
    output_text = output
else:
    output_text = str(output)
print(f"  Output length: {len(output_text)} chars")
print(f"  Output:\n---\n{output_text}\n---")

# Save full output text
(OUT_DIR / "full_output.txt").write_text(output_text, encoding="utf-8")

# ─── summary ─────────────────────────────────────────────────────────────────
print(f"\nAll reference files saved to {OUT_DIR}/")
for f in sorted(OUT_DIR.iterdir()):
    if f.suffix == ".npy":
        arr = np.load(f)
        print(f"  {f.name}: shape={arr.shape} dtype={arr.dtype}")
    else:
        print(f"  {f.name}: {f.stat().st_size} bytes")
