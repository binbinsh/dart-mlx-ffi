#!/usr/bin/env python3
"""Generate a compact Qwen2 BPE tokenizer sidecar for the native runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NEUTTS_AIR_SPECIALS = {
    "<|TEXT_REPLACE|>",
    "<|TEXT_PROMPT_START|>",
    "<|TEXT_PROMPT_END|>",
    "<|SPEECH_REPLACE|>",
    "<|SPEECH_GENERATION_START|>",
    "<|SPEECH_GENERATION_END|>",
}

COSYVOICE2_SPECIALS = [
    ("<|endoftext|>", 151643),
    ("<|im_start|>", 151644),
    ("<|im_end|>", 151645),
    ("<|endofprompt|>", 151646),
    ("[breath]", 151647),
    ("<strong>", 151648),
    ("</strong>", 151649),
    ("[noise]", 151650),
    ("[laughter]", 151651),
    ("[cough]", 151652),
    ("[clucking]", 151653),
    ("[accent]", 151654),
    ("[quick_breath]", 151655),
    ("<laughter>", 151656),
    ("</laughter>", 151657),
    ("[hissing]", 151658),
    ("[sigh]", 151659),
    ("[vocalized-noise]", 151660),
    ("[lipsmack]", 151661),
    ("[mn]", 151662),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--tokenizer-json",
        type=Path,
        help="Path to a HuggingFace tokenizer.json with a BPE model.",
    )
    input_group.add_argument(
        "--model-dir",
        type=Path,
        help=(
            "Directory containing tokenizer.json or vocab.json + merges.txt; "
            "output defaults there."
        ),
    )
    input_group.add_argument(
        "--tokenizer-dir",
        type=Path,
        help="Directory containing vocab.json and merges.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output .qwen2bpe path. Defaults to the input tokenizer directory."
        ),
    )
    parser.add_argument(
        "--specials",
        choices=["all", "none", "neutts-air", "cosyvoice2"],
        default="all",
        help=(
            "Which added tokens to register as tokenizer specials. Use "
            "'neutts-air' to avoid registering 65k speech tokens in the hot "
            "path, or 'cosyvoice2' for CosyVoice2's runtime-added controls."
        ),
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        help="Additional added-token content to register as a special.",
    )
    return parser.parse_args()


def _hex(value: str) -> str:
    return value.encode("utf-8").hex()


def _merge_parts(entry: object) -> tuple[str, str]:
    if isinstance(entry, list) and len(entry) == 2:
        return str(entry[0]), str(entry[1])
    if isinstance(entry, str):
        left, sep, right = entry.partition(" ")
        if sep:
            return left, right
    raise ValueError(f"invalid BPE merge entry: {entry!r}")


def _selected_specials(items: list[dict], mode: str, extras: set[str]) -> list[dict]:
    if mode == "all":
        selected = list(items)
    elif mode == "none":
        selected = []
    elif mode == "neutts-air":
        selected = [
            item for item in items if item.get("content") in NEUTTS_AIR_SPECIALS
        ]
    elif mode == "cosyvoice2":
        selected = [
            {"content": content, "id": token_id}
            for content, token_id in COSYVOICE2_SPECIALS
        ]
    else:
        raise AssertionError(mode)
    if extras:
        by_content = {item.get("content"): item for item in items}
        by_content.update({item.get("content"): item for item in selected})
        for token in sorted(extras):
            item = by_content.get(token)
            if item is None:
                raise ValueError(f"special token {token!r} not found in added_tokens")
            if item not in selected:
                selected.append(item)
    return selected


def _added_tokens_from_config(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    decoder = data.get("added_tokens_decoder") or {}
    if not isinstance(decoder, dict):
        return []
    tokens: list[dict] = []
    for token_id, item in decoder.items():
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            tokens.append({"id": int(token_id), "content": content})
    return tokens


def _load_tokenizer_model(
    args: argparse.Namespace,
) -> tuple[Path, dict, list, list[dict]]:
    if args.tokenizer_json is not None:
        tokenizer_json = args.tokenizer_json
        data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError(f"expected BPE tokenizer, got {model.get('type')!r}")
        vocab = model.get("vocab")
        merges = model.get("merges")
        added_tokens = data.get("added_tokens") or []
        base_dir = tokenizer_json.parent
    else:
        base_dir = args.tokenizer_dir or args.model_dir
        tokenizer_json = base_dir / "tokenizer.json"
        if tokenizer_json.exists():
            data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
            model = data["model"]
            if model.get("type") != "BPE":
                raise ValueError(f"expected BPE tokenizer, got {model.get('type')!r}")
            vocab = model.get("vocab")
            merges = model.get("merges")
            added_tokens = data.get("added_tokens") or _added_tokens_from_config(
                base_dir / "tokenizer_config.json"
            )
        else:
            vocab_path = base_dir / "vocab.json"
            merges_path = base_dir / "merges.txt"
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            merges = [
                line
                for line in merges_path.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            ]
            added_tokens = _added_tokens_from_config(
                base_dir / "tokenizer_config.json"
            )

    if not isinstance(vocab, dict):
        raise ValueError("BPE vocab must be an object")
    if not isinstance(merges, list):
        raise ValueError("BPE merges must be a list")
    return base_dir, vocab, merges, added_tokens


def main() -> None:
    args = parse_args()
    base_dir, vocab, merges, added_tokens = _load_tokenizer_model(args)
    output = args.output or base_dir / "tokenizer.qwen2bpe"
    max_id = max(
        [int(value) for value in vocab.values()]
        + [int(item["id"]) for item in added_tokens if "id" in item],
        default=-1,
    )
    specials = _selected_specials(
        added_tokens,
        args.specials,
        set(args.special_token),
    )
    for item in specials:
        max_id = max(max_id, int(item["id"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        f.write("qwen2bpe\t1\n")
        f.write(f"meta\tdeclared_vocab_size\t{max_id + 1}\n")
        for token, token_id in sorted(vocab.items(), key=lambda item: int(item[1])):
            f.write(f"v\t{int(token_id)}\t{_hex(token)}\n")
        for entry in merges:
            left, right = _merge_parts(entry)
            f.write(f"m\t{_hex(left)}\t{_hex(right)}\n")
        for item in sorted(specials, key=lambda entry: int(entry["id"])):
            content = item.get("content")
            if not content:
                continue
            f.write(f"s\t{int(item['id'])}\t{_hex(str(content))}\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
