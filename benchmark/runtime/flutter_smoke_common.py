from __future__ import annotations

import base64
import json
import selectors
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


MARKER = "DMF_RUNTIME_SMOKE_RESULT:"
MARKER_BEGIN = "DMF_RUNTIME_SMOKE_RESULT_BEGIN"
MARKER_CHUNK = "DMF_RUNTIME_SMOKE_RESULT_CHUNK:"
MARKER_END = "DMF_RUNTIME_SMOKE_RESULT_END"
ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ENV_FILE = ROOT / ".dart_mlx_runtime_env.json"


def extract_marker_payload(line: str) -> dict[str, Any] | None:
    if MARKER not in line:
        return None
    raw = line.split(MARKER, 1)[1].strip()
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

        if MARKER_BEGIN in line:
            self._started = True
            self._chunks.clear()
            self._expected_chunks = self._parse_expected_chunks(line)
            return None

        if MARKER_CHUNK in line:
            self._started = True
            self._store_chunk(line)
            if self._expected_chunks is not None and len(self._chunks) >= self._expected_chunks:
                return self._decode_chunks()
            return None

        if MARKER_END in line and self._started:
            return self._decode_chunks()

        return None

    def _parse_expected_chunks(self, line: str) -> int | None:
        raw = line.split(MARKER_BEGIN, 1)[1].strip()
        if raw.startswith(":"):
            raw = raw[1:].strip()
        if not raw:
            return None
        try:
            parsed = int(raw)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    def _store_chunk(self, line: str) -> None:
        raw = line.split(MARKER_CHUNK, 1)[1].strip()
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


def write_runtime_env_file(values: dict[str, str]) -> Path:
    payload = dict(values)
    RUNTIME_ENV_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return RUNTIME_ENV_FILE


def clear_runtime_env_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def capture_runtime_smoke(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: int,
    parser: MarkerParser,
    line_observer: Callable[[str], None] | None = None,
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
            if line_observer is not None:
                line_observer(line)
            marker_payload = parser.feed(line)
            if marker_payload is not None:
                break
    finally:
        selector.close()
    return marker_payload, timed_out, lines
