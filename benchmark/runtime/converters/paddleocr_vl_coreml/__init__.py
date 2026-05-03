"""PaddleOCR-VL-1.5 CoreML re-architecture (Phase 1).

This package replaces the prior monolithic ``paddleocr_vl_coreml.py`` script
with a 3-mlpackage pipeline: vision+embed (Model A), prefill decoder (Model B,
stateful), and decode decoder (Model C, stateful, single-token).

Authoritative documents:

* ``mise-kitchen-prep/docs/adr/0001-paddleocr-vl-coreml-rearchitecture.md``
* ``mise-kitchen-prep/docs/plans/paddleocr-vl-coreml-rebuild.md``

Pinned tooling:

* coremltools == 9.0
* torch == 2.11.0
* transformers == 5.3.0
"""

from .pipeline import PipelineConfig, build_pipeline, main

__all__ = ["PipelineConfig", "build_pipeline", "main"]
