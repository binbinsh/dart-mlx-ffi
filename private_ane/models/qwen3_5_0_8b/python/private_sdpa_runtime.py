from __future__ import annotations

from dataclasses import dataclass

from private_ane_ctypes import PrivateAneModel
from private_probe_cache import get as get_probe, make_key as make_probe_key, set as set_probe


def make_sdpa_mil(num_heads: int, key_len: int, head_dim: int) -> str:
    return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> a,
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> b,
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> c
    ) {{
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> out =
            scaled_dot_product_attention(query = a, key = b, value = c)[name = string("sdpa")];
    }} -> (out);
}}
"""


def make_sdpa_prefill_group_mil(num_heads: int, seq_len: int, head_dim: int) -> str:
    bodies = []
    outputs = []
    for prefix in range(1, seq_len + 1):
        p = f"{prefix:02d}"
        q_begin = [0, 0, prefix - 1, 0]
        q_end = [1, num_heads, prefix, head_dim]
        kv_begin = [0, 0, 0, 0]
        kv_end = [1, num_heads, prefix, head_dim]
        out_name = f"o{p}"
        outputs.append(out_name)
        bodies.append(
            f"""        tensor<int32, [4]> qb{p} = const()[name = string("qb{p}"), val = tensor<int32, [4]>([{q_begin[0]},{q_begin[1]},{q_begin[2]},{q_begin[3]}])];
        tensor<int32, [4]> qe{p} = const()[name = string("qe{p}"), val = tensor<int32, [4]>([{q_end[0]},{q_end[1]},{q_end[2]},{q_end[3]}])];
        tensor<int32, [4]> kb{p} = const()[name = string("kb{p}"), val = tensor<int32, [4]>([{kv_begin[0]},{kv_begin[1]},{kv_begin[2]},{kv_begin[3]}])];
        tensor<int32, [4]> ke{p} = const()[name = string("ke{p}"), val = tensor<int32, [4]>([{kv_end[0]},{kv_end[1]},{kv_end[2]},{kv_end[3]}])];
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> q{p} = slice_by_index(begin = qb{p}, end = qe{p}, x = q)[name = string("q{p}")];
        tensor<fp16, [1, {num_heads}, {prefix}, {head_dim}]> k{p} = slice_by_index(begin = kb{p}, end = ke{p}, x = k)[name = string("k{p}")];
        tensor<fp16, [1, {num_heads}, {prefix}, {head_dim}]> v{p} = slice_by_index(begin = kb{p}, end = ke{p}, x = v)[name = string("v{p}")];
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> {out_name} =
            scaled_dot_product_attention(query = q{p}, key = k{p}, value = v{p})[name = string("{out_name}")];"""
        )
    return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {num_heads}, {seq_len}, {head_dim}]> q,
        tensor<fp16, [1, {num_heads}, {seq_len}, {head_dim}]> k,
        tensor<fp16, [1, {num_heads}, {seq_len}, {head_dim}]> v
    ) {{
{chr(10).join(bodies)}
    }} -> ({", ".join(outputs)});
}}
"""


def make_packed_sdpa_mil(*, key_len: int, num_heads: int, head_dim: int) -> str:
    channels = num_heads * head_dim
    total_steps = 1 + (2 * key_len)
    q_shape = [1, num_heads, head_dim, 1]
    kv_shape = [1, num_heads, head_dim, key_len]
    q_begin = [0, 0, 0, 0]
    q_end = [1, channels, 1, 1]
    k_begin = [0, 0, 0, 1]
    k_end = [1, channels, 1, 1 + key_len]
    v_begin = [0, 0, 0, 1 + key_len]
    v_end = [1, channels, 1, total_steps]
    perm = [0, 1, 3, 2]
    return f"""program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3510.2.1"}}, {{"coremlc-version", "3505.4.1"}}, {{"coremltools-component-milinternal", ""}}, {{"coremltools-version", "9.0"}}}})]
{{
    func main<ios18>(
        tensor<fp16, [1, {channels}, 1, {total_steps}]> packed
    ) {{
        tensor<int32, [4]> q_begin = const()[name = string("q_begin"), val = tensor<int32, [4]>([{q_begin[0]},{q_begin[1]},{q_begin[2]},{q_begin[3]}])];
        tensor<int32, [4]> q_end = const()[name = string("q_end"), val = tensor<int32, [4]>([{q_end[0]},{q_end[1]},{q_end[2]},{q_end[3]}])];
        tensor<int32, [4]> k_begin = const()[name = string("k_begin"), val = tensor<int32, [4]>([{k_begin[0]},{k_begin[1]},{k_begin[2]},{k_begin[3]}])];
        tensor<int32, [4]> k_end = const()[name = string("k_end"), val = tensor<int32, [4]>([{k_end[0]},{k_end[1]},{k_end[2]},{k_end[3]}])];
        tensor<int32, [4]> v_begin = const()[name = string("v_begin"), val = tensor<int32, [4]>([{v_begin[0]},{v_begin[1]},{v_begin[2]},{v_begin[3]}])];
        tensor<int32, [4]> v_end = const()[name = string("v_end"), val = tensor<int32, [4]>([{v_end[0]},{v_end[1]},{v_end[2]},{v_end[3]}])];
        tensor<fp16, [1, {channels}, 1, 1]> q_flat = slice_by_index(begin = q_begin, end = q_end, x = packed)[name = string("q_flat")];
        tensor<fp16, [1, {channels}, 1, {key_len}]> k_flat = slice_by_index(begin = k_begin, end = k_end, x = packed)[name = string("k_flat")];
        tensor<fp16, [1, {channels}, 1, {key_len}]> v_flat = slice_by_index(begin = v_begin, end = v_end, x = packed)[name = string("v_flat")];
        tensor<int32, [4]> q_shape = const()[name = string("q_shape"), val = tensor<int32, [4]>([{q_shape[0]},{q_shape[1]},{q_shape[2]},{q_shape[3]}])];
        tensor<int32, [4]> kv_shape = const()[name = string("kv_shape"), val = tensor<int32, [4]>([{kv_shape[0]},{kv_shape[1]},{kv_shape[2]},{kv_shape[3]}])];
        tensor<int32, [4]> perm = const()[name = string("perm"), val = tensor<int32, [4]>([{perm[0]},{perm[1]},{perm[2]},{perm[3]}])];
        tensor<fp16, [1, {num_heads}, {head_dim}, 1]> q_reshaped = reshape(shape = q_shape, x = q_flat)[name = string("q_reshaped")];
        tensor<fp16, [1, {num_heads}, {head_dim}, {key_len}]> k_reshaped = reshape(shape = kv_shape, x = k_flat)[name = string("k_reshaped")];
        tensor<fp16, [1, {num_heads}, {head_dim}, {key_len}]> v_reshaped = reshape(shape = kv_shape, x = v_flat)[name = string("v_reshaped")];
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> q =
            transpose(perm = perm, x = q_reshaped)[name = string("q")];
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> k =
            transpose(perm = perm, x = k_reshaped)[name = string("k")];
        tensor<fp16, [1, {num_heads}, {key_len}, {head_dim}]> v =
            transpose(perm = perm, x = v_reshaped)[name = string("v")];
        tensor<fp16, [1, {num_heads}, 1, {head_dim}]> out =
            scaled_dot_product_attention(query = q, key = k, value = v)[name = string("sdpa")];
    }} -> (out);
}}
"""


def pack_sdpa_input(q: "np.ndarray", k: "np.ndarray", v: "np.ndarray") -> "np.ndarray":
    import numpy as np

    q_flat = np.transpose(q, (0, 1, 3, 2)).reshape(q.shape[0], -1, 1, 1)
    key_len = k.shape[2]
    k_flat = np.transpose(k, (0, 1, 3, 2)).reshape(k.shape[0], -1, 1, key_len)
    v_flat = np.transpose(v, (0, 1, 3, 2)).reshape(v.shape[0], -1, 1, key_len)
    return np.concatenate([q_flat, k_flat, v_flat], axis=3).astype("float16", copy=False)


@dataclass
class _SdpaEntry:
    model: PrivateAneModel
    session: object
    output_bytes: int

    def close(self) -> None:
        self.session.close()
        self.model.close()


@dataclass
class _SdpaPrefillEntry:
    model: PrivateAneModel
    session: object
    output_bytes: int
    output_count: int

    def close(self) -> None:
        self.session.close()
        self.model.close()


@dataclass
class _PackedSdpaEntry:
    model: PrivateAneModel
    session: object

    def close(self) -> None:
        self.session.close()
        self.model.close()


class PrivateSdpaRuntime:
    def __init__(
        self,
        entries: dict[int, _SdpaEntry],
        *,
        num_heads: int,
        head_dim: int,
    ):
        self.entries = entries
        self.prefill_entries: dict[int, _SdpaPrefillEntry] = {}
        self.packed_entries: dict[int, _PackedSdpaEntry] = {}
        self.packed_disabled: set[int] = set()
        self.num_heads = num_heads
        self.head_dim = head_dim

    @classmethod
    def build(
        cls,
        *,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
    ) -> "PrivateSdpaRuntime":
        entries = {}
        q_bytes = num_heads * head_dim * 2
        for prefix in range(1, max_seq_len + 1):
            kv_bytes = num_heads * prefix * head_dim * 2
            out_bytes = num_heads * head_dim * 2
            model = PrivateAneModel.from_mil(
                make_sdpa_mil(num_heads, prefix, head_dim),
                weights=[],
            )
            model.compile()
            model.load()
            session = model.create_session(
                input_bytes=[q_bytes, kv_bytes, kv_bytes],
                output_bytes=[out_bytes],
            )
            entries[prefix] = _SdpaEntry(
                model=model,
                session=session,
                output_bytes=out_bytes,
            )
        return cls(entries, num_heads=num_heads, head_dim=head_dim)

    def _build_prefill_entry(self, seq_len: int) -> _SdpaPrefillEntry:
        qkv_bytes = self.num_heads * seq_len * self.head_dim * 2
        out_bytes = self.num_heads * self.head_dim * 2
        model = PrivateAneModel.from_mil(
            make_sdpa_prefill_group_mil(self.num_heads, seq_len, self.head_dim),
            weights=[],
        )
        model.compile()
        model.load()
        session = model.create_session(
            input_bytes=[qkv_bytes, qkv_bytes, qkv_bytes],
            output_bytes=[out_bytes] * seq_len,
        )
        entry = _SdpaPrefillEntry(
            model=model,
            session=session,
            output_bytes=out_bytes,
            output_count=seq_len,
        )
        self.prefill_entries[seq_len] = entry
        return entry

    def _build_packed_entry(self, key_len: int) -> _PackedSdpaEntry:
        channels = self.num_heads * self.head_dim
        total_steps = 1 + (2 * key_len)
        input_bytes = channels * total_steps * 2
        output_bytes = channels * 2
        model = PrivateAneModel.from_mil(
            make_packed_sdpa_mil(
                key_len=key_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            ),
            weights=[],
        )
        model.compile()
        model.load()
        session = model.create_session(
            input_bytes=[input_bytes],
            output_bytes=[output_bytes],
        )
        entry = _PackedSdpaEntry(model=model, session=session)
        self.packed_entries[key_len] = entry
        return entry

    def prepare_prefill(self, seq_len: int) -> bool:
        if seq_len <= 1 or seq_len in self.prefill_entries:
            return True
        try:
            self._build_prefill_entry(seq_len)
            return True
        except Exception:
            return False

    def run(self, prefix: int, q_blob: bytes, k_blob: bytes, v_blob: bytes) -> bytes:
        entry = self.entries[prefix]
        return entry.session.run_many([q_blob, k_blob, v_blob])[0]

    def run_prefill(self, seq_len: int, q_blob: bytes, k_blob: bytes, v_blob: bytes) -> list[bytes]:
        entry = self.prefill_entries.get(seq_len)
        if entry is None:
            entry = self._build_prefill_entry(seq_len)
        return entry.session.run_many([q_blob, k_blob, v_blob])

    def prepare_packed(self, key_len: int) -> bool:
        if key_len <= 1:
            return True
        probe_key = make_probe_key(
            "packed_sdpa",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            key_len=key_len,
        )
        if key_len in self.packed_disabled:
            return False
        cached = get_probe(probe_key)
        if cached is False:
            self.packed_disabled.add(key_len)
            return False
        try:
            entry = self.packed_entries.get(key_len)
            if entry is None:
                entry = self._build_packed_entry(key_len)
            channels = self.num_heads * self.head_dim
            total_steps = 1 + (2 * key_len)
            import numpy as np

            zero = np.zeros((1, channels, 1, total_steps), dtype=np.float16)
            entry.session.run_one(zero)
            set_probe(probe_key, True)
            return True
        except Exception:
            self.packed_disabled.add(key_len)
            entry = self.packed_entries.pop(key_len, None)
            if entry is not None:
                entry.close()
            set_probe(probe_key, False)
            return False

    def run_packed(self, key_len: int, packed) -> bytes:
        if key_len in self.packed_disabled:
            raise RuntimeError(f"Packed SDPA disabled for key_len={key_len}")
        entry = self.packed_entries.get(key_len)
        if entry is None:
            entry = self._build_packed_entry(key_len)
        try:
            return entry.session.run_one(packed)
        except Exception:
            self.packed_disabled.add(key_len)
            entry.close()
            self.packed_entries.pop(key_len, None)
            raise

    def close(self) -> None:
        for entry in self.entries.values():
            entry.close()
        for entry in self.prefill_entries.values():
            entry.close()
        for entry in self.packed_entries.values():
            entry.close()
