"""Legacy entry-point — delegates to the new package.

This file used to be a 850-LOC monolithic exporter. It has been replaced by
``benchmark.runtime.converters.paddleocr_vl_coreml`` (a package), per
docs/adr/0001-paddleocr-vl-coreml-rearchitecture.md.

Run the new pipeline with::

    python -m benchmark.runtime.converters.paddleocr_vl_coreml.pipeline \\
        --hf-snapshot <path> --output-dir <path> [--image <path>]

This shim still works for callers that invoke the old script directly: it
forwards argv to the package's ``main`` so existing wrappers don't break.
"""

from __future__ import annotations

import sys


def main() -> None:
    from benchmark.runtime.converters.paddleocr_vl_coreml.pipeline import main as pkg_main

    pkg_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
