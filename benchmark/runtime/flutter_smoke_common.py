from __future__ import annotations

import base64
import json
import selectors
import subprocess
import time
from typing import Any


MARKERS = ("DINF_RUNTIME_SMOKE_RESULT:", "DMF_RUNTIME_SMOKE_RESULT:")
MARKER_BEGINS = (
    "DINF_RUNTIME_SMOKE_RESULT_BEGIN",
    "DMF_RUNTIME_SMOKE_RESULT_BEGIN",
)
MARKER_CHUNKS = (
    "DINF_RUNTIME_SMOKE_RESULT_CHUNK:",
    "DMF_RUNTIME_SMOKE_RESULT_CHUNK:",
)
MARKER_ENDS = (
    "DINF_RUNTIME_SMOKE_RESULT_END",
    "DMF_RUNTIME_SMOKE_RESULT_END",
)


def extract_marker_payload(line: str) -> dict[str, Any] | None:
    marker = _first_marker(line, MARKERS)
    if marker is None:
        return None
    raw = line.split(marker, 1)[1].strip()
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {"passed": False, "error": "Invalid marker JSON", "raw": raw}
    if isinstance(decoded, dict):
        return decoded
    return {"passed": False, "error": "Marker payload is not an object", "raw": raw}


class MarkerParser:
    def __init__(self) -> None:
        self._started = False
        self._expected_chunks: int | None = None
        self._chunks: dict[int, str] = {}

    def feed(self, line: str) -> dict[str, Any] | None:
        inline = extract_marker_payload(line)
        if inline is not None:
            return inline

        begin = _first_marker(line, MARKER_BEGINS)
        if begin is not None:
            self._started = True
            self._chunks.clear()
            self._expected_chunks = self._parse_expected_chunks(line, begin)
            return None

        chunk = _first_marker(line, MARKER_CHUNKS)
        if chunk is not None:
            self._started = True
            self._store_chunk(line, chunk)
            if self._expected_chunks is not None and len(self._chunks) >= self._expected_chunks:
                return self._decode_chunks()
            return None

        if _first_marker(line, MARKER_ENDS) is not None and self._started:
            return self._decode_chunks()

        return None

    def _parse_expected_chunks(self, line: str, marker: str) -> int | None:
        raw = line.split(marker, 1)[1].strip()
        if raw.startswith(":"):
            raw = raw[1:].strip()
        if not raw:
            return None
        try:
            parsed = int(raw)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    def _store_chunk(self, line: str, marker: str) -> None:
        raw = line.split(marker, 1)[1].strip()
        if not raw:
            return
        if ":" not in raw:
            return
        header, chunk = raw.split(":", 1)
        index = None
        expected = None
        if "/" in header:
            left, right = header.split("/", 1)
            try:
                index = int(left)
                expected = int(right)
            except ValueError:
                return
        else:
            try:
                index = int(header)
            except ValueError:
                return
        if index is None or index <= 0:
            return
        if expected is not None and expected > 0:
            self._expected_chunks = expected
        self._chunks[index] = chunk.strip()

    def _decode_chunks(self) -> dict[str, Any]:
        if not self._chunks:
            result = {"passed": False, "error": "No marker chunks received"}
            self._reset()
            return result

        expected = self._expected_chunks
        if expected is None:
            expected = max(self._chunks)
        missing = [i for i in range(1, expected + 1) if i not in self._chunks]
        if missing:
            result = {
                "passed": False,
                "error": "Incomplete marker chunks",
                "chunks_expected": expected,
                "chunks_received": len(self._chunks),
                "missing_chunks": missing,
            }
            self._reset()
            return result

        encoded = "".join(self._chunks[i] for i in range(1, expected + 1))
        try:
            raw_json = base64.b64decode(encoded).decode("utf-8")
        except Exception as exc:  # pragma: no cover - defensive decode guard.
            result = {"passed": False, "error": f"Invalid marker encoding: {exc}"}
            self._reset()
            return result
        try:
            decoded = json.loads(raw_json)
        except json.JSONDecodeError:
            result = {"passed": False, "error": "Invalid marker JSON", "raw": raw_json}
            self._reset()
            return result
        self._reset()
        if isinstance(decoded, dict):
            return decoded
        return {"passed": False, "error": "Marker payload is not an object"}

    def _reset(self) -> None:
        self._started = False
        self._expected_chunks = None
        self._chunks.clear()


def _first_marker(line: str, markers: tuple[str, ...]) -> str | None:
    for marker in markers:
        if marker in line:
            return marker
    return None


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        process.stderr.close()


def capture_runtime_smoke(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: int,
    parser: MarkerParser,
) -> tuple[dict[str, Any] | None, bool, list[str]]:
    marker_payload: dict[str, Any] | None = None
    timed_out = False
    lines: list[str] = []
    deadline = time.time() + timeout_seconds
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                timed_out = True
                break
            events = selector.select(timeout=min(1.0, remaining))
            if not events:
                if process.poll() is not None:
                    break
                continue
            line = process.stdout.readline()
            if line == "":
                if process.poll() is not None:
                    break
                continue
            line = line.rstrip("\n")
            print(line, flush=True)
            lines.append(line)
            marker_payload = parser.feed(line)
            if marker_payload is not None:
                break
    finally:
        selector.close()
    return marker_payload, timed_out, lines
