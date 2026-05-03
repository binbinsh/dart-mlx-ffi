"""Model B: prefill decoder (stateful KV write).

Inputs (CoreML, iOS18+ stateful mlprogram):
  - inputs_embeds      : fp16, EnumeratedShapes (1, P, 1024)  — P ∈ PROMPT_BUCKETS
  - attention_mask     : int32, (1, P)
  - rope_cos / rope_sin: fp16, (3, 1, P, head_dim)  — host-precomputed mRoPE
                         (3 = mrope x/y/t axes; we already do the mrope-section
                         selection on host so the model gets the *selected* table)
  - prompt_len_used    : int32 scalar (1,) — number of real tokens; cache write
                         range is [0:prompt_len_used). Required so decode can
                         continue from the right offset.

States (one StateType per buffer, 18 layers × 2 = 36):
  - k_cache_{i} / v_cache_{i} : fp16, (1, num_kv_heads=2, MAX_KV_LEN=4096, head_dim=128)

Outputs:
  - logits             : fp16, (1, 1, vocab) — logits for the LAST real token

Design notes
------------
* mRoPE selection: instead of passing the per-axis pieces, we ask the host to
  apply ``_select_mrope`` and pass the already-selected ``(1, 1, P, head_dim)``
  cos/sin table broadcast across heads. This keeps the graph simple and avoids
  baking ``mrope_section`` constants per bucket.

* Causal mask + padding mask are *recomputed inside* the model from the
  ``attention_mask`` input plus a baked lower-triangular constant for each P
  bucket. This makes the graph self-contained.

* ``prompt_len_used`` selects the last-token row from the (P, vocab) logits.
  This avoids returning a (1, P, vocab) tensor (P up to 768 × 100k ≈ 153 MiB).

* Writing the cache: for slot index range ``[0:prompt_len_used)`` we use a
  full-length write (``state[..., :P, :] = k_full``) — when ``prompt_len_used <
  P`` the *padding rows* are zeros (host-set) and write benign garbage that the
  decode step never reads (decode reads ``[:cur_len]`` and ``cur_len`` starts
  at ``prompt_len_used``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .enumerated_shapes import MAX_KV_LEN


@dataclass
class PrefillTraceExample:
    inputs_embeds: torch.Tensor    # (1, P, 1024) fp16
    attention_mask: torch.Tensor   # (1, P) int32
    rope_cos: torch.Tensor         # (1, 1, P, head_dim) fp16
    rope_sin: torch.Tensor         # (1, 1, P, head_dim) fp16
    prompt_len_used: torch.Tensor  # (1,) int32
    prompt_len: int


class PrefillDecoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, prompt_len: int) -> None:
        super().__init__()
        # Bug V: LIB layout. Text decoder lives at model.model.language_model
        # (SHIPPED had it directly at model.model). lm_head is still on the
        # outer model. Text-decoder hyperparameters live on
        # config.text_config (the multimodal config has no head_dim etc.).
        self.layers = model.model.language_model.layers
        self.norm = model.model.language_model.norm
        self.lm_head = model.lm_head

        text_cfg = model.config.text_config
        self.prompt_len = int(prompt_len)
        self.hidden_size = int(text_cfg.hidden_size)
        self.num_heads = int(text_cfg.num_attention_heads)
        self.num_kv_heads = int(text_cfg.num_key_value_heads)
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(text_cfg.head_dim)
        self.half_head_dim = self.head_dim // 2
        self.num_layers = len(self.layers)

        # Baked lower-triangular causal mask for the bucket length.
        # Use the fp16 minimum (-65504) instead of fp32 minimum: the
        # converter emits this graph in fp16 and ``fp32.min`` (-3.4e38)
        # casts to ``-inf`` in fp16. Then ``0 * -inf = NaN`` poisons
        # softmax → all positions get the same weight → image-invariant
        # output. ``fp16.min`` stays finite under the cast and behaves
        # like ``-inf`` for softmax purposes (exp(-65504) underflows to 0).
        causal = torch.full(
            (prompt_len, prompt_len),
            torch.finfo(torch.float16).min,
            dtype=torch.float32,
        )
        causal = torch.triu(causal, diagonal=1).reshape(1, 1, prompt_len, prompt_len)
        self.register_buffer("causal_mask", causal, persistent=False)

        # KV-cache state buffers (registered as named buffers so coremltools
        # can pair them with ct.StateType entries via name match).
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
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        prompt_len_used: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = self.lm_head.weight.dtype
        h = inputs_embeds.to(target_dtype)
        mask = self._attention_mask(attention_mask)
        for i, layer in enumerate(self.layers):
            h = self._decoder_layer(layer, i, h, mask, rope_cos, rope_sin)
        h = self.norm(h)
        # Select last *real* token via gather along seq axis.
        # prompt_len_used is shape (1,); we need an int index into dim=1.
        last_idx = prompt_len_used.long().reshape(1) - 1
        last = torch.index_select(h, dim=1, index=last_idx)  # (1, 1, hidden)
        return self.lm_head(last)

    # ------------------------------------------------------------------ #
    def _decoder_layer(self, layer, layer_idx, h, mask, cos, sin):
        residual = h
        h = layer.input_layernorm(h)
        h = self._attention(layer.self_attn, layer_idx, h, mask, cos, sin)
        h = residual + h
        residual = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        return residual + h

    def _attention(self, attn, layer_idx, h, mask, cos, sin):
        P = self.prompt_len
        q = attn.q_proj(h).reshape(1, P, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(h).reshape(1, P, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(h).reshape(1, P, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._apply_rope(q, k, cos, sin)

        # Write K, V into the cache buffers at slots [0:P]. Padding rows
        # (beyond prompt_len_used) carry zeros; decode's mask hides them.
        #
        # IMPORTANT: use a *tensor* begin index (`zero`) so coremltools'
        # ``generate_tensor_assignment_ops`` pass recognises the canonical
        # in-place state-update pattern and lowers to ``coreml_update_state``
        # (the same op decode_decoder_model.py uses). With a Python-int
        # begin (``[:, :, :P, :]``), coremltools instead emits a plain
        # ``slice_update`` that does NOT mutate the state buffer in place —
        # so the prefill-built KV cache is invisible to the decode model
        # at runtime, even though both wrappers compute the same K/V.
        # This was the root cause of the prefill→decode KV desync.
        k_buf = getattr(self, f"k_cache_{layer_idx}")
        v_buf = getattr(self, f"v_cache_{layer_idx}")
        zero = torch.zeros((), dtype=torch.int32)
        end = zero + P
        k_buf[:, :, zero:end, :] = k.to(k_buf.dtype)
        v_buf[:, :, zero:end, :] = v.to(v_buf.dtype)

        k_full = self._repeat_kv(k)
        v_full = self._repeat_kv(v)

        # Compute attention in fp32 to avoid fp16 overflow in softmax —
        # matches upstream HF (``softmax(..., dtype=torch.float32)``).
        # Pre-scale ``q`` so the matmul itself stays in a safe range.
        scale = float(attn.scaling)
        q32 = q.float() * scale
        k32 = k_full.float()
        scores = torch.matmul(q32, k32.transpose(2, 3))
        scores = scores + mask.float()
        weights = torch.softmax(scores, dim=-1).to(v_full.dtype)
        out = torch.matmul(weights, v_full)
        out = out.transpose(1, 2).reshape(1, P, self.num_heads * self.head_dim)
        return attn.o_proj(out)

    def _attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        padding = (1.0 - attention_mask.to(torch.float32)).reshape(1, 1, 1, self.prompt_len)
        padding = padding * torch.finfo(torch.float16).min  # fp16-safe; see __init__ comment
        return self.causal_mask + padding

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.kv_groups == 1:
            return x
        x = x[:, :, None, :, :].expand(
            1, self.num_kv_heads, self.kv_groups, self.prompt_len, self.head_dim
        )
        return x.reshape(1, self.num_heads, self.prompt_len, self.head_dim)

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
    prompt_len: int,
    *,
    head_dim: int,
    hidden_size: int,
    dtype: torch.dtype = torch.float16,
) -> PrefillTraceExample:
    """Build NON-DEGENERATE example tensors for ``torch.jit.trace``.

    All four tensor inputs must contain non-trivial values; otherwise the
    tracer constant-folds them and the converted MIL graph treats them as
    dead inputs (the bug we're avoiding here):

    * ``inputs_embeds``  — random; fp16-realistic magnitude.
    * ``attention_mask`` — mix of 1s (kept) and 0s (padded) so the
      ``(1 - mask) * float_min`` branch is exercised; otherwise the
      padding term folds to 0 and the whole mask becomes a constant.
    * ``rope_cos`` / ``rope_sin`` — values in [-1, 1]; if zero, every
      ``q*cos + rotate_half(q)*sin`` collapses to 0 at trace time and
      RoPE silently disappears from the graph.
    * ``prompt_len_used`` — strictly < prompt_len so the gather-last-token
      branch sees a non-trivial index.
    """
    g = torch.Generator().manual_seed(0)
    inputs_embeds = torch.randn(1, prompt_len, hidden_size, generator=g, dtype=torch.float32).to(dtype)
    pad_len = max(1, prompt_len // 8)
    keep_len = prompt_len - pad_len
    attention_mask = torch.cat(
        [
            torch.ones(1, keep_len, dtype=torch.int32),
            torch.zeros(1, pad_len, dtype=torch.int32),
        ],
        dim=1,
    )
    # RoPE cos/sin live in [-1, 1]; sample angles and take cos/sin so the
    # tracer sees realistic per-position values (no degenerate zeros).
    angles = torch.randn(1, 1, prompt_len, head_dim, generator=g, dtype=torch.float32)
    rope_cos = torch.cos(angles).to(dtype)
    rope_sin = torch.sin(angles).to(dtype)
    prompt_len_used = torch.tensor([max(1, prompt_len // 2)], dtype=torch.int32)
    return PrefillTraceExample(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        prompt_len_used=prompt_len_used,
        prompt_len=prompt_len,
    )
