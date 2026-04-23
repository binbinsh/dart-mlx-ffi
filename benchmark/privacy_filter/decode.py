from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

BACKGROUND_LABEL = "O"
SPAN_LABELS = (
    BACKGROUND_LABEL,
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)
BOUNDARY_TAGS = ("B", "I", "E", "S")
TOKEN_LABELS = (BACKGROUND_LABEL,) + tuple(
    f"{tag}-{label}"
    for label in SPAN_LABELS
    if label != BACKGROUND_LABEL
    for tag in BOUNDARY_TAGS
)
VITERBI_BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)
NEG_INF = -1e9


@dataclass(frozen=True)
class LabelInfo:
    token_to_span_label: dict[int, int]
    token_boundary_tags: dict[int, str | None]
    span_class_names: tuple[str, ...]
    background_token_label: int
    background_span_label: int


@dataclass(frozen=True)
class DetectedSpan:
    label: str
    start: int
    end: int
    text: str
    placeholder: str


@dataclass(frozen=True)
class DecodedPrediction:
    token_labels: tuple[int, ...]
    detected_spans: tuple[DetectedSpan, ...]
    redacted_text: str


def load_label_info(root_dir: Path, model_dir: Path) -> LabelInfo:
    for config_path in (root_dir / "config.json", model_dir / "config.json"):
        if not config_path.exists():
            continue
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if isinstance(id2label, dict):
            labels = tuple(str(id2label[str(index)]) for index in range(len(id2label)))
            return build_label_info(labels)
        if int(raw.get("num_labels", -1)) == len(TOKEN_LABELS):
            return build_label_info(TOKEN_LABELS)
    raise ValueError("Could not resolve privacy-filter label names from config.json")


def build_label_info(class_names: Sequence[str]) -> LabelInfo:
    token_to_span_label: dict[int, int] = {}
    token_boundary_tags: dict[int, str | None] = {}
    span_class_names = [BACKGROUND_LABEL]
    span_lookup = {BACKGROUND_LABEL: 0}
    background_token_label: int | None = None

    for index, name in enumerate(class_names):
        if name == BACKGROUND_LABEL:
            background_token_label = index
            token_to_span_label[index] = span_lookup[BACKGROUND_LABEL]
            token_boundary_tags[index] = None
            continue
        if "-" not in name:
            raise ValueError(f"Invalid token label {name!r}")
        boundary, span_name = name.split("-", 1)
        if boundary not in BOUNDARY_TAGS or not span_name:
            raise ValueError(f"Invalid token label {name!r}")
        span_index = span_lookup.get(span_name)
        if span_index is None:
            span_index = len(span_class_names)
            span_lookup[span_name] = span_index
            span_class_names.append(span_name)
        token_to_span_label[index] = span_index
        token_boundary_tags[index] = boundary

    if background_token_label is None:
        raise ValueError("Label names must include O")
    return LabelInfo(
        token_to_span_label=token_to_span_label,
        token_boundary_tags=token_boundary_tags,
        span_class_names=tuple(span_class_names),
        background_token_label=background_token_label,
        background_span_label=span_lookup[BACKGROUND_LABEL],
    )


def load_viterbi_biases(
    root_dir: Path,
    model_dir: Path,
    calibration_path: str | None = None,
) -> dict[str, float]:
    candidates = []
    if calibration_path:
        candidates.append(Path(calibration_path).expanduser())
    candidates.extend(
        [
            root_dir / "viterbi_calibration.json",
            model_dir / "viterbi_calibration.json",
        ],
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        raw = json.loads(candidate.read_text(encoding="utf-8"))
        biases = raw["operating_points"]["default"]["biases"]
        return {key: float(biases.get(key, 0.0)) for key in VITERBI_BIAS_KEYS}
    return {key: 0.0 for key in VITERBI_BIAS_KEYS}


class ViterbiDecoder:
    def __init__(self, label_info: LabelInfo, biases: dict[str, float]) -> None:
        self.label_info = label_info
        self.biases = biases
        num_classes = len(label_info.token_to_span_label)
        self.start_scores = np.full((num_classes,), NEG_INF, dtype=np.float32)
        self.end_scores = np.full((num_classes,), NEG_INF, dtype=np.float32)
        self.transition_scores = np.full(
            (num_classes, num_classes),
            NEG_INF,
            dtype=np.float32,
        )
        self._build_scores(num_classes)

    def decode(self, token_logprobs: np.ndarray) -> list[int]:
        if token_logprobs.ndim != 2:
            raise ValueError("token_logprobs must have shape [seq_len, num_classes]")
        seq_len, num_classes = token_logprobs.shape
        if seq_len == 0:
            return []

        scores = token_logprobs[0] + self.start_scores
        backpointers = np.empty((seq_len - 1, num_classes), dtype=np.int64)
        for index in range(1, seq_len):
            transitions = scores[:, None] + self.transition_scores
            backpointers[index - 1] = np.argmax(transitions, axis=0)
            scores = np.max(transitions, axis=0) + token_logprobs[index]

        if not np.isfinite(scores).any():
            return np.argmax(token_logprobs, axis=1).astype(int).tolist()

        scores = scores + self.end_scores
        last_label = int(np.argmax(scores))
        path = np.empty((seq_len,), dtype=np.int64)
        path[-1] = last_label
        for index in range(seq_len - 2, -1, -1):
            last_label = int(backpointers[index, last_label])
            path[index] = last_label
        return path.astype(int).tolist()

    def _build_scores(self, num_classes: int) -> None:
        info = self.label_info
        for index in range(num_classes):
            tag = info.token_boundary_tags.get(index)
            span = info.token_to_span_label.get(index)
            if tag in {"B", "S"} or index == info.background_token_label:
                self.start_scores[index] = 0.0
            if tag in {"E", "S"} or index == info.background_token_label:
                self.end_scores[index] = 0.0
            for next_index in range(num_classes):
                next_tag = info.token_boundary_tags.get(next_index)
                next_span = info.token_to_span_label.get(next_index)
                if self._is_valid_transition(
                    prev_tag=tag,
                    prev_span=span,
                    next_tag=next_tag,
                    next_span=next_span,
                    next_index=next_index,
                ):
                    self.transition_scores[index, next_index] = self._transition_bias(
                        prev_tag=tag,
                        prev_span=span,
                        next_tag=next_tag,
                        next_span=next_span,
                        prev_index=index,
                        next_index=next_index,
                    )

    def _is_valid_transition(
        self,
        *,
        prev_tag: str | None,
        prev_span: int | None,
        next_tag: str | None,
        next_span: int | None,
        next_index: int,
    ) -> bool:
        next_is_background = self._is_background(next_span, next_index)
        if (next_span is None or next_tag is None) and not next_is_background:
            return False
        if prev_span is None or prev_tag is None:
            return next_is_background or next_tag in {"B", "S"}
        if self._is_background(prev_span, None):
            return next_is_background or next_tag in {"B", "S"}
        if prev_tag in {"E", "S"}:
            return next_is_background or next_tag in {"B", "S"}
        if prev_tag in {"B", "I"}:
            return prev_span == next_span and next_tag in {"I", "E"}
        return False

    def _transition_bias(
        self,
        *,
        prev_tag: str | None,
        prev_span: int | None,
        next_tag: str | None,
        next_span: int | None,
        prev_index: int,
        next_index: int,
    ) -> float:
        prev_is_background = self._is_background(prev_span, prev_index)
        next_is_background = self._is_background(next_span, next_index)
        if prev_is_background:
            if next_is_background:
                return self.biases["transition_bias_background_stay"]
            if next_tag in {"B", "S"}:
                return self.biases["transition_bias_background_to_start"]
            return 0.0
        if prev_tag in {"B", "I"}:
            if next_tag == "I" and prev_span == next_span:
                return self.biases["transition_bias_inside_to_continue"]
            if next_tag == "E" and prev_span == next_span:
                return self.biases["transition_bias_inside_to_end"]
            return 0.0
        if prev_tag in {"E", "S"}:
            if next_is_background:
                return self.biases["transition_bias_end_to_background"]
            if next_tag in {"B", "S"}:
                return self.biases["transition_bias_end_to_start"]
        return 0.0

    def _is_background(self, span: int | None, token_index: int | None) -> bool:
        return span == self.label_info.background_span_label or (
            token_index == self.label_info.background_token_label
        )


def decode_prediction(
    logits: np.ndarray,
    *,
    token_offsets: Sequence[tuple[int, int]],
    text: str,
    label_info: LabelInfo,
    decoder: ViterbiDecoder | None,
    trim_whitespace: bool = True,
) -> DecodedPrediction:
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for span decoding")
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError("logits must have shape [seq_len, num_labels]")
    if logits.shape[0] != len(token_offsets):
        raise ValueError("token_offsets length must match logits sequence length")

    log_probs = _log_softmax(logits.astype(np.float32))
    token_labels = (
        decoder.decode(log_probs)
        if decoder is not None
        else np.argmax(log_probs, axis=-1).astype(int).tolist()
    )
    token_spans = labels_to_spans(
        {index: label for index, label in enumerate(token_labels)},
        label_info,
    )
    char_spans = token_spans_to_char_spans(token_spans, token_offsets)
    if trim_whitespace:
        char_spans = trim_char_spans_whitespace(char_spans, text)
    detected = _select_non_overlapping_spans(
        [
            DetectedSpan(
                label=(
                    label_info.span_class_names[label]
                    if 0 <= label < len(label_info.span_class_names)
                    else f"label_{label}"
                ),
                start=start,
                end=end,
                text=text[start:end],
                placeholder=_label_placeholder(
                    label_info.span_class_names[label]
                    if 0 <= label < len(label_info.span_class_names)
                    else f"label_{label}",
                ),
            )
            for label, start, end in char_spans
            if 0 <= start < end <= len(text)
        ],
    )
    return DecodedPrediction(
        token_labels=tuple(token_labels),
        detected_spans=tuple(detected),
        redacted_text=_redact_text(text, detected),
    )


def labels_to_spans(
    labels_by_index: dict[int, int],
    label_info: LabelInfo,
) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    current_label: int | None = None
    start_index: int | None = None
    previous_index: int | None = None

    for token_index in sorted(labels_by_index):
        label_id = labels_by_index[token_index]
        span_label = label_info.token_to_span_label.get(label_id)
        boundary_tag = label_info.token_boundary_tags.get(label_id)

        if previous_index is not None and token_index != previous_index + 1:
            if current_label is not None and start_index is not None:
                spans.append((current_label, start_index, previous_index + 1))
            current_label = None
            start_index = None

        if span_label is None:
            previous_index = token_index
            continue
        if span_label == label_info.background_span_label:
            if current_label is not None and start_index is not None:
                spans.append((current_label, start_index, token_index))
            current_label = None
            start_index = None
            previous_index = token_index
            continue
        if boundary_tag == "S":
            if current_label is not None and start_index is not None:
                spans.append((current_label, start_index, previous_index + 1))
            spans.append((span_label, token_index, token_index + 1))
            current_label = None
            start_index = None
        elif boundary_tag == "B":
            if current_label is not None and start_index is not None:
                spans.append((current_label, start_index, previous_index + 1))
            current_label = span_label
            start_index = token_index
        elif boundary_tag == "I":
            if current_label != span_label:
                if current_label is not None and start_index is not None:
                    spans.append((current_label, start_index, previous_index + 1))
                current_label = span_label
                start_index = token_index
        elif boundary_tag == "E":
            if current_label == span_label and start_index is not None:
                spans.append((current_label, start_index, token_index + 1))
            else:
                spans.append((span_label, token_index, token_index + 1))
            current_label = None
            start_index = None
        else:
            current_label = None
            start_index = None
        previous_index = token_index

    if current_label is not None and start_index is not None and previous_index is not None:
        spans.append((current_label, start_index, previous_index + 1))
    return spans


def token_spans_to_char_spans(
    spans: Sequence[tuple[int, int, int]],
    token_offsets: Sequence[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    converted: list[tuple[int, int, int]] = []
    for label, token_start, token_end in spans:
        if not (0 <= token_start < token_end <= len(token_offsets)):
            continue
        start = int(token_offsets[token_start][0])
        end = int(token_offsets[token_end - 1][1])
        if end > start:
            converted.append((label, start, end))
    return converted


def trim_char_spans_whitespace(
    spans: Sequence[tuple[int, int, int]],
    text: str,
) -> list[tuple[int, int, int]]:
    trimmed = []
    for label, start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            trimmed.append((label, start, end))
    return trimmed


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _label_placeholder(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    return f"<{normalized or 'REDACTED'}>"


def _select_non_overlapping_spans(
    spans: Sequence[DetectedSpan],
) -> list[DetectedSpan]:
    ordered = sorted(
        spans,
        key=lambda span: (span.start, -(span.end - span.start), span.label),
    )
    kept = []
    cursor = 0
    for span in ordered:
        if span.start < cursor or span.end <= span.start:
            continue
        kept.append(span)
        cursor = span.end
    return kept


def _redact_text(text: str, spans: Sequence[DetectedSpan]) -> str:
    pieces = []
    cursor = 0
    for span in spans:
        pieces.append(text[cursor : span.start])
        pieces.append(span.placeholder)
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)
