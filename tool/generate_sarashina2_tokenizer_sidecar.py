#!/usr/bin/env python3
"""Generate a compact Sarashina2 tokenizer sidecar for the native runtime."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--tokenizer-json",
        type=Path,
        help="Path to sarashina2.2-tts/tokenizer.json.",
    )
    input_group.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing tokenizer.json; output defaults there.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .sara2tok path. Defaults to <model-dir>/tokenizer.sara2tok.",
    )
    return parser.parse_args()


def _hex(value: str) -> str:
    return value.encode("utf-8").hex()


def _validate_byte_fallback_vocab(vocab: list) -> None:
    found: set[int] = set()
    pattern = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
    for item in vocab:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"invalid Unigram vocab item: {item!r}")
        token = str(item[0])
        match = pattern.match(token)
        if match is not None:
            found.add(int(match.group(1), 16))
    missing = sorted(set(range(256)) - found)
    if missing:
        preview = ", ".join(f"<0x{value:02X}>" for value in missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} missing)"
        raise ValueError(f"byte_fallback=true but vocab is missing {preview}{suffix}")


def main() -> None:
    args = parse_args()
    tokenizer_json = args.tokenizer_json or args.model_dir / "tokenizer.json"
    output = args.output or tokenizer_json.with_name("tokenizer.sara2tok")
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    model = data["model"]
    if model.get("type") != "Unigram":
        raise ValueError(f"expected Unigram tokenizer, got {model.get('type')!r}")
    if model.get("byte_fallback") is not True:
        raise ValueError("Sarashina2 tokenizer currently expects byte_fallback=true")
    pre = data.get("pre_tokenizer") or {}
    if pre.get("type") != "Metaspace":
        raise ValueError(f"expected Metaspace pre_tokenizer, got {pre.get('type')!r}")
    if pre.get("split") is not False:
        raise ValueError("Sarashina2 tokenizer currently expects Metaspace split=false")
    if pre.get("prepend_scheme", "never") != "never":
        raise ValueError(
            "Sarashina2 tokenizer currently expects Metaspace prepend_scheme=never"
        )
    if data.get("normalizer") is not None:
        raise ValueError("Sarashina2 tokenizer currently expects no normalizer")
    vocab = model.get("vocab")
    if not isinstance(vocab, list):
        raise ValueError("tokenizer.json model.vocab must be a list")
    _validate_byte_fallback_vocab(vocab)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        f.write("sara2tok\t1\n")
        f.write(f"meta\tunk_id\t{int(model.get('unk_id', 0))}\n")
        f.write("meta\tbyte_fallback\t1\n")
        f.write(f"meta\treplacement_hex\t{_hex(pre.get('replacement', '▁'))}\n")
        f.write(f"meta\tprepend_scheme\t{pre.get('prepend_scheme', 'never')}\n")
        for idx, item in enumerate(vocab):
            token, score = item
            f.write(f"tok\t{idx}\t{float(score):.9g}\t{_hex(token)}\n")
        for item in data.get("added_tokens", []):
            content = item.get("content", "")
            if not content:
                continue
            special = 1 if item.get("special") else 0
            f.write(f"add\t{int(item['id'])}\t{special}\t{_hex(content)}\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
