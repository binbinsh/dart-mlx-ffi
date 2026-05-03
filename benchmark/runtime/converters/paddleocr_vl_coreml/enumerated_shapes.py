"""Bucket shapes shared across the 3 sub-converters.

The image grid buckets come from ADR §5.1 (six (T, H, W) triples). The prompt
length buckets come from ADR §5.2. Decode KV uses a RangeDim per ADR §5.3.

All three sub-converters import from this single source of truth so the
runtime pipeline.json is internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# (T, H, W) — H and W are post-patchify grid units (not pixels). Pixel
# dimensions are H*patch_size and W*patch_size respectively (patch_size=14).
GRID_BUCKETS: Tuple[Tuple[int, int, int], ...] = (
    (1, 28, 28),
    (1, 40, 28),
    (1, 28, 40),
    (1, 56, 28),
    (1, 28, 56),
    (1, 42, 42),
)

PATCH_SIZE = 14
SPATIAL_MERGE = 2

# Prompt length enumerated buckets (ADR §5.2). Right-pad to the smallest
# enclosing bucket. Anything > 768 falls back to MLX in the runtime.
PROMPT_BUCKETS: Tuple[int, ...] = (128, 256, 384, 512, 768)

# KV cache RangeDim upper bound (ADR §5.3).
MAX_KV_LEN = 4096
DECODE_KV_DEFAULT = 512


@dataclass(frozen=True)
class ImageBucket:
    t: int
    h: int  # patch grid units
    w: int  # patch grid units

    @property
    def num_patches(self) -> int:
        """Number of raw vision patches before spatial merge."""
        return self.t * self.h * self.w

    @property
    def num_merged_tokens(self) -> int:
        """Number of fused image tokens after Projector merges 2x2 patches."""
        return self.t * (self.h // SPATIAL_MERGE) * (self.w // SPATIAL_MERGE)

    @property
    def pixel_h(self) -> int:
        return self.h * PATCH_SIZE

    @property
    def pixel_w(self) -> int:
        return self.w * PATCH_SIZE


def all_image_buckets() -> tuple[ImageBucket, ...]:
    return tuple(ImageBucket(t, h, w) for (t, h, w) in GRID_BUCKETS)


def patch_count_buckets() -> tuple[int, ...]:
    """Per-bucket raw patch counts. Vision Model A uses these as the
    EnumeratedShapes set on the ``num_patches`` axis of ``pixel_values``.
    """
    return tuple(b.num_patches for b in all_image_buckets())


def merged_token_buckets() -> tuple[int, ...]:
    return tuple(b.num_merged_tokens for b in all_image_buckets())


def default_image_bucket() -> ImageBucket:
    """Bucket used as the trace example (and the EnumeratedShapes default)."""
    return ImageBucket(*GRID_BUCKETS[0])


def default_prompt_len() -> int:
    return PROMPT_BUCKETS[1]
