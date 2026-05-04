"""Model A: NaViT vision encoder + projector.

Hybrid OCR contract (issue #1):

Inputs (CoreML):
  - pixel_values    : float16, EnumeratedShapes over (1, num_patches, 3*14*14)
  - image_grid_thw  : int32,   shape (3,) — runtime metadata (T, H, W). Used
                      by the host to pick the right enumerated bucket; the
                      traced graph itself is bucket-specialized so the values
                      are not consumed numerically. We thread the tensor
                      through a no-op cast/sum so the converter does not
                      prune the input.

Outputs:
  - image_embeds    : float16, (num_image_tokens, hidden=1024) — the
                      projector output BEFORE scatter into prompt embeddings.
                      ``num_image_tokens`` equals ``bucket.num_merged_tokens``
                      and is fixed per traced subgraph; one mlpackage holds
                      one subgraph per image bucket via EnumeratedShapes.

The Dart-side runner performs the scatter into the prompt embedding
buffer (see ``paddleOcrVlScatterImageEmbeddings`` in ``embed.dart``).
This package no longer holds an ``embed_tokens`` lookup table or a
``torch.where`` scatter — those moved to the host.

The pixel_values input is the post-processor flat layout
``[num_patches, 3, 14, 14]`` reshaped to ``[1, num_patches, 588]`` so all
buckets share a single rank-3 tensor with one variable axis.

The projector internals (rope, position_embedding, merge_kernel) depend
on the *grid* (T, H, W). To keep one mlpackage per CoreML model, we trace
6 separate vision graphs (one per bucket) and combine them via
EnumeratedShapes — which means the scalar buffers used inside the trace
are fine because each enumerated shape produces its own subgraph in MIL.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .enumerated_shapes import (
    PATCH_SIZE,
    SPATIAL_MERGE,
    ImageBucket,
)


@dataclass
class VisionEmbedTraceExample:
    pixel_values: torch.Tensor       # (1, num_patches, 3*14*14) fp16
    image_grid_thw: torch.Tensor     # (3,) int32
    bucket: ImageBucket


class VisionEmbedWrapper(torch.nn.Module):
    """Vision encoder + Projector. Emits ``image_embeds`` only.

    A single instance is bound to a single ``ImageBucket`` because the
    rotary cache, position-embedding interpolation, and merge reshape all
    depend on the (T, H, W) grid. The pipeline traces one wrapper per
    bucket, and the converter assembles them into a single mlpackage with
    EnumeratedShapes on ``num_patches``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        bucket: ImageBucket,
    ) -> None:
        super().__init__()
        self.bucket = bucket

        # ---- Vision tower ------------------------------------------------
        # Bug V: LIB layout. SHIPPED used `model.visual` / `model.mlp_AR`;
        # LIB nests these under `model.model` (the multimodal wrapper).
        embeddings = model.model.visual.vision_model.embeddings
        self.patch_embedding = embeddings.patch_embedding
        self.layers = model.model.visual.vision_model.encoder.layers
        self.post_layernorm = model.model.visual.vision_model.post_layernorm
        self.projector = model.model.projector  # was: model.mlp_AR (SHIPPED)

        first_attn = self.layers[0].self_attn
        self.embed_dim = int(first_attn.embed_dim)
        self.num_heads = int(first_attn.num_heads)
        self.head_dim = int(first_attn.head_dim)
        self.half_head_dim = self.head_dim // 2

        t, h, w = bucket.t, bucket.h, bucket.w
        self.t = t
        self.h = h
        self.w = w
        self.num_patches = bucket.num_patches
        self.num_merged_tokens = bucket.num_merged_tokens

        m1, m2 = self.projector.merge_kernel_size
        self.merge_kernel_h = int(m1)
        self.merge_kernel_w = int(m2)
        assert self.merge_kernel_h == SPATIAL_MERGE
        assert self.merge_kernel_w == SPATIAL_MERGE
        self.merged_t = t
        self.merged_h = h // self.merge_kernel_h
        self.merged_w = w // self.merge_kernel_w

        # Pre-compute pos-emb + rotary for this bucket (frozen buffers).
        with torch.no_grad():
            hidden_size = int(model.config.vision_config.hidden_size)
            # Bug V: LIB's interpolate_pos_encoding signature is
            # (embeddings, height, width). SHIPPED took an extra `is_video`
            # bool. Drop the trailing arg.
            position_embedding = embeddings.interpolate_pos_encoding(
                torch.zeros(t * h * w, hidden_size),
                h,
                w,
            ).squeeze(0).repeat(t, 1)
            image_pids = torch.arange(t * h * w, dtype=torch.long) % (h * w)
            height_position_ids = image_pids // w
            width_position_ids = image_pids % w
            pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
            max_grid_size = int(pids.max().item() + 1)
            rope_emb_max_grid = (
                model.model.visual.vision_model.encoder.rotary_pos_emb(max_grid_size)
            )
            rope_emb = rope_emb_max_grid[pids].flatten(1).repeat(1, 2)
            rope_cos = rope_emb.cos()
            rope_sin = rope_emb.sin()

        self.register_buffer("position_embedding", position_embedding, persistent=False)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # pixel_values: (1, num_patches, 3*14*14) — reshape to model's expected
        # 5D layout (1, num_patches, 3, 14, 14) then run vision tower.
        target_dtype = self.patch_embedding.weight.dtype
        flat = pixel_values.to(target_dtype)
        per_patch = flat.reshape(self.num_patches, 3, PATCH_SIZE, PATCH_SIZE)
        patch_embeds = self.patch_embedding(per_patch)  # (N, hidden, 1, 1)
        hidden_states = patch_embeds.flatten(-2).squeeze(-1)
        hidden_states = hidden_states.reshape(1, self.num_patches, -1)
        hidden_states = hidden_states + self.position_embedding.unsqueeze(0)

        rope_emb = (self.rope_cos, self.rope_sin)
        for layer in self.layers:
            hidden_states = self._encoder_layer(layer, hidden_states, rope_emb)

        image_features = self.post_layernorm(hidden_states).squeeze(0)
        image_embeds = self._project(image_features)  # (M, text_hidden=1024)

        # Thread image_grid_thw through a numerically-zero op so coremltools
        # keeps the input alive in the converted graph. The traced subgraph
        # is bucket-specialized; grid_thw values are runtime metadata only.
        grid_noop = image_grid_thw.to(image_embeds.dtype).sum() * 0.0
        return image_embeds + grid_noop

    # ------------------------------------------------------------------ #
    def _project(self, image_features: torch.Tensor) -> torch.Tensor:
        x = self.projector.pre_norm(image_features)
        x = x.reshape(
            self.merged_t,
            self.merged_h,
            self.merge_kernel_h,
            self.merged_w,
            self.merge_kernel_w,
            self.embed_dim,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            self.merged_t * self.merged_h * self.merged_w,
            self.merge_kernel_h * self.merge_kernel_w * self.embed_dim,
        )
        x = self.projector.linear_1(x)
        x = self.projector.act(x)
        return self.projector.linear_2(x)

    def _encoder_layer(self, layer, hidden_states, rope_emb):
        residual = hidden_states
        hidden_states = layer.layer_norm1(hidden_states)
        hidden_states = self._attention(layer.self_attn, hidden_states, rope_emb)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer.layer_norm2(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        return residual + hidden_states

    def _attention(self, attention, hidden_states, rope_emb):
        N = self.num_patches
        q = attention.q_proj(hidden_states).reshape(1, N, self.num_heads, self.head_dim)
        k = attention.k_proj(hidden_states).reshape(1, N, self.num_heads, self.head_dim)
        v = attention.v_proj(hidden_states).reshape(1, N, self.num_heads, self.head_dim)
        q, k = self._apply_rotary(q, k, rope_emb)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # fp32 attention — vision encoder has 1379 patches per bucket and
        # softmax over fp16 logits drifts visibly (Phase E observed MAE
        # ~0.5 on Stage A vs HF). Upcast q*scale and mat-K then softmax in
        # fp32, like upstream.
        # Bug V: LIB's PaddleOCRVisionAttention has no `.scale` attribute
        # (SHIPPED did). Compute the standard 1/sqrt(head_dim) scale.
        scale_attr = getattr(attention, "scale", None)
        scale = float(scale_attr) if scale_attr is not None else self.head_dim ** -0.5
        q32 = q.float() * scale
        k32 = k.float()
        scores = torch.matmul(q32, k32.transpose(-1, -2))
        weights = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(1, N, self.embed_dim)
        return attention.out_proj(out)

    def _apply_rotary(self, q, k, rope_emb):
        cos, sin = rope_emb
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        qf, kf = q.float(), k.float()
        q_e = (qf * cos) + (self._rotate_half(qf) * sin)
        k_e = (kf * cos) + (self._rotate_half(kf) * sin)
        return q_e.to(q.dtype), k_e.to(k.dtype)

    def _rotate_half(self, x):
        a = x[..., : self.half_head_dim]
        b = x[..., self.half_head_dim :]
        return torch.cat((-b, a), dim=-1)


def make_trace_example(
    bucket: ImageBucket,
    *,
    dtype: torch.dtype = torch.float16,
) -> VisionEmbedTraceExample:
    """Build a self-consistent dummy batch for tracing.

    Non-degenerate values so ``torch.jit.trace`` cannot constant-fold any
    input. Zero pixel_values would let the tracer collapse the entire
    vision tower (every q*0/k*0/v*0 chain folds).
    """
    n = bucket.num_patches
    g = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(
        1, n, 3 * PATCH_SIZE * PATCH_SIZE, generator=g, dtype=torch.float32
    ).to(dtype)
    image_grid_thw = torch.tensor([bucket.t, bucket.h, bucket.w], dtype=torch.int32)
    return VisionEmbedTraceExample(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        bucket=bucket,
    )
