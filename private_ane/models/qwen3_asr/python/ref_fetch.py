from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_OUT_DIR = "tmp/Qwen3-ASR-1.7B"
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        args.model,
        local_dir=str(out_dir),
        allow_patterns=ALLOW_PATTERNS,
        max_workers=args.max_workers,
    )
    print(path)


if __name__ == "__main__":
    main()
