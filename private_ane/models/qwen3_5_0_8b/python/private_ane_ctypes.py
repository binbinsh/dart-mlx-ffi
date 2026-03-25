from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LIB = ROOT / ".dart_tool" / "lib" / "libdart_mlx_ffi.dylib"


class PrivateAneError(RuntimeError):
    pass


def resolve_library_path() -> Path:
    if DEFAULT_LIB.exists():
        return DEFAULT_LIB
    candidates = sorted(ROOT.glob("**/libdart_mlx_ffi.dylib"))
    if not candidates:
        raise FileNotFoundError("Unable to locate libdart_mlx_ffi.dylib")
    return candidates[0]


def _load_library() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(resolve_library_path()))

    lib.dart_mlx_ane_private_is_enabled.restype = ctypes.c_bool
    lib.dart_mlx_ane_private_clear_error.restype = None
    lib.dart_mlx_ane_private_last_error_copy.restype = ctypes.c_void_p
    lib.dart_mlx_string_free_copy.argtypes = [ctypes.c_void_p]
    lib.dart_mlx_string_free_copy.restype = None
    lib.dart_mlx_free_buffer.argtypes = [ctypes.c_void_p]
    lib.dart_mlx_free_buffer.restype = None

    lib.dart_mlx_ane_private_model_new_mil_ex.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.dart_mlx_ane_private_model_new_mil_ex.restype = ctypes.c_void_p
    lib.dart_mlx_ane_private_model_free.argtypes = [ctypes.c_void_p]
    lib.dart_mlx_ane_private_model_free.restype = None
    lib.dart_mlx_ane_private_model_compile.argtypes = [ctypes.c_void_p]
    lib.dart_mlx_ane_private_model_compile.restype = ctypes.c_int
    lib.dart_mlx_ane_private_model_load.argtypes = [ctypes.c_void_p]
    lib.dart_mlx_ane_private_model_load.restype = ctypes.c_int

    lib.dart_mlx_ane_private_session_new.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]
    lib.dart_mlx_ane_private_session_new.restype = ctypes.c_void_p
    lib.dart_mlx_ane_private_session_free.argtypes = [ctypes.c_void_p]
    lib.dart_mlx_ane_private_session_free.restype = None
    lib.dart_mlx_ane_private_session_write_input_bytes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.dart_mlx_ane_private_session_write_input_bytes.restype = ctypes.c_int
    lib.dart_mlx_ane_private_session_evaluate.argtypes = [ctypes.c_void_p]
    lib.dart_mlx_ane_private_session_evaluate.restype = ctypes.c_int
    lib.dart_mlx_ane_private_session_read_output_bytes_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.dart_mlx_ane_private_session_read_output_bytes_copy.restype = ctypes.c_void_p
    return lib


LIB = _load_library()


def _last_error() -> str:
    ptr = LIB.dart_mlx_ane_private_last_error_copy()
    if not ptr:
        return "Unknown private ANE failure."
    try:
        return ctypes.string_at(ptr).decode("utf-8", errors="replace")
    finally:
        LIB.dart_mlx_string_free_copy(ptr)


def _check_status(name: str, status: int) -> None:
    if status != 0:
        raise PrivateAneError(f"{name} failed: {_last_error()}")


def _check_handle(name: str, handle: int | None) -> int:
    if not handle:
        raise PrivateAneError(f"{name} failed: {_last_error()}")
    return int(handle)


def is_enabled() -> bool:
    return bool(LIB.dart_mlx_ane_private_is_enabled())


class PrivateAneModel:
    def __init__(self, handle: int):
        self.handle = handle

    @classmethod
    def from_mil(
        cls,
        mil_text: str,
        *,
        weights: Iterable[tuple[str, bytes, int]],
    ) -> "PrivateAneModel":
        LIB.dart_mlx_ane_private_clear_error()
        weights = list(weights)
        path_arr = (ctypes.c_char_p * len(weights))()
        data_arr = (ctypes.POINTER(ctypes.c_uint8) * len(weights))()
        len_arr = (ctypes.c_size_t * len(weights))()
        off_arr = (ctypes.c_size_t * len(weights))()
        keepalive = []
        for index, (path, data, offset) in enumerate(weights):
            path_arr[index] = path.encode("utf-8")
            buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            keepalive.append(buf)
            data_arr[index] = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
            len_arr[index] = len(data)
            off_arr[index] = offset
        handle = _check_handle(
            "model_new_mil_ex",
            LIB.dart_mlx_ane_private_model_new_mil_ex(
                mil_text.encode("utf-8"),
                path_arr if weights else None,
                data_arr if weights else None,
                len_arr if weights else None,
                off_arr if weights else None,
                len(weights),
            ),
        )
        model = cls(handle)
        model._keepalive = keepalive
        return model

    def compile(self) -> None:
        LIB.dart_mlx_ane_private_clear_error()
        _check_status("model_compile", LIB.dart_mlx_ane_private_model_compile(self.handle))

    def load(self) -> None:
        LIB.dart_mlx_ane_private_clear_error()
        _check_status("model_load", LIB.dart_mlx_ane_private_model_load(self.handle))

    def create_session(self, *, input_bytes: list[int], output_bytes: list[int]) -> "PrivateAneSession":
        in_arr = (ctypes.c_size_t * len(input_bytes))(*input_bytes)
        out_arr = (ctypes.c_size_t * len(output_bytes))(*output_bytes)
        LIB.dart_mlx_ane_private_clear_error()
        handle = _check_handle(
            "session_new",
            LIB.dart_mlx_ane_private_session_new(
                self.handle,
                in_arr if input_bytes else None,
                len(input_bytes),
                out_arr if output_bytes else None,
                len(output_bytes),
            ),
        )
        return PrivateAneSession(handle, input_bytes=input_bytes, output_bytes=output_bytes)

    def close(self) -> None:
        if self.handle:
            LIB.dart_mlx_ane_private_model_free(self.handle)
            self.handle = 0


class PrivateAneSession:
    def __init__(self, handle: int, *, input_bytes: list[int], output_bytes: list[int]):
        self.handle = handle
        self.input_bytes = list(input_bytes)
        self.output_bytes = list(output_bytes)

    def _coerce_input(self, blob):
        if isinstance(blob, np.ndarray):
            arr = np.ascontiguousarray(blob)
            size = int(arr.nbytes)
            buf = (ctypes.c_uint8 * size).from_buffer(arr)
            return size, arr, buf
        if isinstance(blob, bytearray):
            size = len(blob)
            buf = (ctypes.c_uint8 * size).from_buffer(blob)
            return size, blob, buf
        if isinstance(blob, memoryview):
            mv = blob.cast("B") if blob.format != "B" or blob.ndim != 1 else blob
            if mv.contiguous and not mv.readonly:
                size = mv.nbytes
                buf = (ctypes.c_uint8 * size).from_buffer(mv)
                return size, mv, buf
            data = mv.tobytes()
            size = len(data)
            buf = (ctypes.c_uint8 * size).from_buffer_copy(data)
            return size, buf, buf
        if isinstance(blob, bytes):
            size = len(blob)
            buf = (ctypes.c_uint8 * size).from_buffer_copy(blob)
            return size, buf, buf
        mv = memoryview(blob)
        if mv.contiguous and not mv.readonly:
            mv = mv.cast("B") if mv.format != "B" or mv.ndim != 1 else mv
            size = mv.nbytes
            buf = (ctypes.c_uint8 * size).from_buffer(mv)
            return size, mv, buf
        data = mv.tobytes()
        size = len(data)
        buf = (ctypes.c_uint8 * size).from_buffer_copy(data)
        return size, buf, buf

    def run_one(self, input_blob) -> bytes:
        if len(self.input_bytes) != 1 or len(self.output_bytes) != 1:
            raise ValueError("run_one only supports single-input/single-output sessions")
        return self.run_many([input_blob])[0]

    def run_many(self, input_blobs) -> list[bytes]:
        if len(input_blobs) != len(self.input_bytes):
            raise ValueError(
                f"Expected {len(self.input_bytes)} inputs, got {len(input_blobs)}"
            )
        buffers = []
        LIB.dart_mlx_ane_private_clear_error()
        for index, blob in enumerate(input_blobs):
            size, keepalive, buf = self._coerce_input(blob)
            if size != self.input_bytes[index]:
                raise ValueError(
                    f"Input {index} expected {self.input_bytes[index]} bytes, got {size}"
                )
            buffers.append((keepalive, buf))
            _check_status(
                f"session_write_input[{index}]",
                LIB.dart_mlx_ane_private_session_write_input_bytes(
                    self.handle,
                    index,
                    ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)),
                    size,
                ),
            )
        _check_status("session_evaluate", LIB.dart_mlx_ane_private_session_evaluate(self.handle))
        outputs = []
        for index in range(len(self.output_bytes)):
            out_len = ctypes.c_size_t()
            ptr = LIB.dart_mlx_ane_private_session_read_output_bytes_copy(
                self.handle,
                index,
                ctypes.byref(out_len),
            )
            if not ptr:
                raise PrivateAneError(
                    f"session_read_output[{index}] failed: {_last_error()}"
                )
            try:
                outputs.append(ctypes.string_at(ptr, out_len.value))
            finally:
                LIB.dart_mlx_free_buffer(ptr)
        return outputs

    def close(self) -> None:
        if self.handle:
            LIB.dart_mlx_ane_private_session_free(self.handle)
            self.handle = 0
