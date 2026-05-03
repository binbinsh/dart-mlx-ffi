"""Model C: decode decoder (stateful KV read+write, single new token).

Inputs (CoreML, iOS18+ stateful mlprogram):
  - inputs_embeds : fp16, (1, 1, 1024)
  - rope_cos / rope_sin : fp16, (1, 1, 1, head_dim)  — host-precomputed mRoPE
  - cur_len       : int32, (1,) — current cache length BEFORE writing this step
                    Range: [1, MAX_KV_LEN-1]
  - kv_len        : int32, (1,) — cur_len + 1; the model uses RangeDim on the
                    attention K/V slice axis. We pass it explicitly so the
                    runtime can match the dim.

States: same 36 K/V buffers as prefill, shared across calls.

Outputs:
  - logits        : fp16, (1, 1, vocab)

The attention computation reads slots ``[0:kv_len)`` of each cache buffer.
Because Core ML can't slice a state buffer by a dynamic length directly, we
mask the padding region instead: read the full ``(1, kv_heads, MAX_KV, hd)``
buffer and add ``-inf`` to attention scores for positions ``>= kv_len``.

This wastes a constant amount of compute per step (matmul over MAX_KV=4096
columns) but trades that for a graph that's free of dynamic shape ops.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .enumerated_shapes import MAX_KV_LEN


@dataclass
class DecodeTraceExample:
    inputs_embeds: torch.Tensor   # (1, 1, hidden) fp16
    rope_cos: torch.Tensor        # (1, 1, 1, head_dim) fp16
    rope_sin: torch.Tensor        # (1, 1, 1, head_dim) fp16
    cur_len: torch.Tensor         # (1,) int32 — slots already filled
    kv_len: torch.Tensor          # (1,) int32 — = cur_len + 1


class DecodeDecoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        # Bug V: LIB layout — see prefill_decoder_model.py for the same
        # rationale. Text decoder is at model.model.language_model and
        # text hyperparameters live on config.text_config.
        self.layers = model.model.language_model.layers
        self.norm = model.model.language_model.norm
        self.lm_head = model.lm_head

        text_cfg = model.config.text_config
        self.hidden_size = int(text_cfg.hidden_size)
        self.num_heads = int(text_cfg.num_attention_heads)
        self.num_kv_heads = int(text_cfg.num_key_value_heads)
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(text_cfg.head_dim)
        self.half_head_dim = self.head_dim // 2
        self.num_layers = len(self.layers)

        # Reusable position vector for masking — constant.
        self.register_buffer(
            "kv_positions",
            torch.arange(MAX_KV_LEN, dtype=torch.int32).reshape(1, 1, 1, MAX_KV_LEN),
            persistent=False,
        )

        for i in range(self.num_layers):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, self.num_kv_heads, MAX_KV_LEN, self.head_dim, dtype=torch.float16),
                persistent=False,
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, self.num_kv_heads, MAX_KV_LEN, self.head_dim, dtype=torch.float16),
                persistent=False,
            )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cur_len: torch.Tensor,
        kv_len: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = self.lm_head.weight.dtype
        h = inputs_embeds.to(target_dtype)
        mask = self._mask_full(kv_len)
        for i, layer in enumerate(self.layers):
            h = self._decoder_layer(layer, i, h, mask, rope_cos, rope_sin, cur_len)
        h = self.norm(h)
        return self.lm_head(h)

    # ------------------------------------------------------------------ #
    def _decoder_layer(self, layer, layer_idx, h, mask, cos, sin, cur_len):
        residual = h
        h = layer.input_layernorm(h)
        h = self._attention(layer.self_attn, layer_idx, h, mask, cos, sin, cur_len)
        h = residual + h
        residual = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        return residual + h

    def _attention(self, attn, layer_idx, h, mask, cos, sin, cur_len):
        # h: (1, 1, hidden) → q (1, num_heads, 1, head_dim)
        q = attn.q_proj(h).reshape(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = attn.k_proj(h).reshape(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = attn.v_proj(h).reshape(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k_new = self._apply_rope(q, k_new, cos, sin)

        # Update K/V cache at slot index = cur_len.  Use the canonical
        # ``state[..., a:b] = x`` slice-assign pattern that coremltools'
        # ``generate_tensor_assignment_ops`` pass recognises and lowers
        # to ``coreml_update_state``.  We use a length-1 slice
        # ``[cur:cur+1]``; cur is a 0-d int tensor extracted from cur_len.
        k_buf = getattr(self, f"k_cache_{layer_idx}")
        v_buf = getattr(self, f"v_cache_{layer_idx}")
        cur = cur_len.to(torch.int32).reshape(())
        end = cur + 1
        k_buf[:, :, cur:end, :] = k_new.to(k_buf.dtype)
        v_buf[:, :, cur:end, :] = v_new.to(v_buf.dtype)

        # Read full buffers; mask later.
        k_full = self._repeat_kv(k_buf.to(q.dtype))   # (1, num_heads, MAX_KV, hd)
        v_full = self._repeat_kv(v_buf.to(q.dtype))

        # Decode attends over MAX_KV=4096 columns — fp16 softmax is
        # particularly fragile here. Mirror upstream's fp32 softmax and
        # pre-scale q so the matmul magnitude stays small.
        scale = float(attn.scaling)
        q32 = q.float() * scale
        k32 = k_full.float()
        scores = torch.matmul(q32, k32.transpose(2, 3))
        scores = scores + mask.float()
        weights = torch.softmax(scores, dim=-1).to(v_full.dtype)
        out = torch.matmul(weights, v_full)            # (1, num_heads, 1, hd)
        out = out.transpose(1, 2).reshape(1, 1, self.num_heads * self.head_dim)
        return attn.o_proj(out)

    def _mask_full(self, kv_len: torch.Tensor) -> torch.Tensor:
        # Positions >= kv_len → -inf, else 0.  Causal is automatic because
        # query length is 1 and we only let it attend to filled positions.
        kv_len_b = kv_len.to(torch.int32).reshape(1, 1, 1, 1)
        keep = (self.kv_positions < kv_len_b).to(torch.float32)
        # 1 → 0, 0 → -inf
        # fp16-safe mask fill: fp32.min casts to -inf in fp16, then
        # 0 * -inf = NaN poisons softmax. fp16.min (-65504) stays finite
        # and acts as -inf for softmax (exp(-65504) underflows to 0).
        return (1.0 - keep) * torch.finfo(torch.float16).min

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.kv_groups == 1:
            return x
        x = x[:, :, None, :, :].expand(
            1, self.num_kv_heads, self.kv_groups, MAX_KV_LEN, self.head_dim
        )
        return x.reshape(1, self.num_heads, MAX_KV_LEN, self.head_dim)

    def _apply_rope(self, q, k, cos, sin):
        cos = cos.float()
        sin = sin.float()
        qf, kf = q.float(), k.float()
        q_e = (qf * cos) + (self._rotate_half(qf) * sin)
        k_e = (kf * cos) + (self._rotate_half(kf) * sin)
        return q_e.to(q.dtype), k_e.to(k.dtype)

    def _rotate_half(self, x):
        a = x[..., : self.half_head_dim]
        b = x[..., self.half_head_dim :]
        return torch.cat((-b, a), dim=-1)


def make_trace_example(
    *, head_dim: int, hidden_size: int, dtype: torch.dtype = torch.float16
) -> DecodeTraceExample:
    """Non-degenerate example tensors for ``torch.jit.trace``.

    See ``prefill_decoder_model.make_trace_example`` for rationale.
    """
    g = torch.Generator().manual_seed(0)
    inputs_embeds = torch.randn(1, 1, hidden_size, generator=g, dtype=torch.float32).to(dtype)
    angles = torch.randn(1, 1, 1, head_dim, generator=g, dtype=torch.float32)
    rope_cos = torch.cos(angles).to(dtype)
    rope_sin = torch.sin(angles).to(dtype)
    # cur_len strictly within (0, MAX_KV_LEN); pick a mid-range value so the
    # padding-mask branch ``(1 - keep) * -inf`` is exercised over a real mix
    # of kept/masked positions.
    cur = 16
    return DecodeTraceExample(
        inputs_embeds=inputs_embeds,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        cur_len=torch.tensor([cur], dtype=torch.int32),
        kv_len=torch.tensor([cur + 1], dtype=torch.int32),
    )
