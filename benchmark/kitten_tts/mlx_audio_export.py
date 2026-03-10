from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

try:
    from ..common import add_vendor_to_path, cleanup_mlx
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import add_vendor_to_path, cleanup_mlx

add_vendor_to_path("mlx-audio")

import mlx.core as mx
from mlx_audio.tts.utils import load
from mlx_audio.tts.models.interpolate import interpolate
from mlx_audio.tts.models.kitten_tts.quant import maybe_fake_quant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--input-ids-json", required=True)
    parser.add_argument("--ref-s-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _f02uv(sine_gen, f0: mx.array) -> mx.array:
    return mx.array(f0 > sine_gen.voiced_threshold, dtype=mx.float32)


def _f02sine_det(sine_gen, f0_values: mx.array, rand_ini: mx.array) -> mx.array:
    rad_values = (f0_values / sine_gen.sampling_rate) % 1
    rand0 = mx.array(rand_ini)
    rand0[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand0
    rad_values = interpolate(
        rad_values.transpose(0, 2, 1),
        scale_factor=1 / sine_gen.upsample_scale,
        mode="linear",
    ).transpose(0, 2, 1)
    phase = mx.cumsum(rad_values, axis=1) * 2 * math.pi
    phase = interpolate(
        phase.transpose(0, 2, 1) * sine_gen.upsample_scale,
        scale_factor=sine_gen.upsample_scale,
        mode="linear",
    ).transpose(0, 2, 1)
    return mx.sin(phase)


def _sine_gen_det(
    sine_gen,
    f0: mx.array,
    rand_ini: mx.array,
    noise: mx.array,
) -> tuple[mx.array, mx.array]:
    fn = f0 * mx.arange(1, sine_gen.harmonic_num + 2)[None, None, :]
    sine_waves = _f02sine_det(sine_gen, fn, rand_ini) * sine_gen.sine_amp
    uv = _f02uv(sine_gen, f0)
    noise_amp = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
    mixed_noise = noise_amp * noise
    return sine_waves * uv + mixed_noise, uv


def _source_det(
    source_module,
    f0: mx.array,
    rand_ini: mx.array,
    noise: mx.array,
) -> tuple[mx.array, mx.array]:
    sine_wavs, uv = _sine_gen_det(source_module.l_sin_gen, f0, rand_ini, noise)
    sine_wavs = maybe_fake_quant(
        sine_wavs, getattr(source_module.l_linear, "activation_quant", False)
    )
    sine_merge = mx.tanh(source_module.l_linear(sine_wavs))
    return sine_merge, uv


def _generator_det(
    generator,
    hidden: mx.array,
    style: mx.array,
    f0_curve: mx.array,
    rand_ini: mx.array,
    noise: mx.array,
) -> mx.array:
    f0 = generator.f0_upsamp(f0_curve[:, None].transpose(0, 2, 1))
    har_source, _ = _source_det(generator.m_source, f0, rand_ini, noise)
    har_source = mx.squeeze(har_source.transpose(0, 2, 1), axis=1)
    har_spec, har_phase = generator.stft.transform(har_source)
    har = mx.concatenate([har_spec, har_phase], axis=1).swapaxes(2, 1)
    for index in range(generator.num_upsamples):
        hidden = mx.where(hidden > 0, hidden, hidden * 0.1)
        har_q = maybe_fake_quant(
            har, getattr(generator.noise_convs[index], "activation_quant", False)
        )
        x_source = generator.noise_convs[index](har_q).swapaxes(2, 1)
        x_source = generator.noise_res[index](x_source, style)
        hidden = generator.ups[index](hidden.swapaxes(2, 1), mx.conv_transpose1d)
        hidden = hidden.swapaxes(2, 1)
        if index == generator.num_upsamples - 1:
            hidden = generator.reflection_pad(hidden)
        hidden = hidden + x_source

        mixed = None
        for kernel_index in range(generator.num_kernels):
            block = generator.resblocks[index * generator.num_kernels + kernel_index](
                hidden, style
            )
            mixed = block if mixed is None else mixed + block
        hidden = mixed / generator.num_kernels
    hidden = mx.where(hidden > 0, hidden, hidden * 0.01)
    hidden = generator.conv_post(hidden.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    spec = mx.exp(hidden[:, : generator.post_n_fft // 2 + 1, :])
    phase = mx.sin(hidden[:, generator.post_n_fft // 2 + 1 :, :])
    return generator.stft.inverse(spec, phase)[0]


def _decoder_det(
    decoder,
    asr: mx.array,
    f0_pred: mx.array,
    n_pred: mx.array,
    style: mx.array,
    rand_ini: mx.array,
    noise: mx.array,
) -> mx.array:
    f0_curve = f0_pred
    f0 = decoder.F0_conv(f0_pred[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(
        2, 1
    )
    n = decoder.N_conv(n_pred[:, None, :].swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    hidden = mx.concatenate([asr, f0, n], axis=1)
    hidden = decoder.encode(hidden, style)
    asr_res = decoder.asr_res[0](asr.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)
    use_residual = True
    for block in decoder.decode:
        if use_residual:
            hidden = mx.concatenate([hidden, asr_res, f0, n], axis=1)
        hidden = block(hidden, style)
        if getattr(block, "upsample_type", "none") != "none":
            use_residual = False
    return _generator_det(
        decoder.generator,
        hidden,
        style,
        f0_curve,
        rand_ini,
        noise,
    )


def forward_front_base(
    model,
    input_ids: mx.array,
    ref_s: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    input_lengths = mx.array([input_ids.shape[-1]])
    text_mask = mx.arange(int(input_lengths.max()))[None, ...]
    text_mask = mx.repeat(text_mask, input_lengths.shape[0], axis=0).astype(
        input_lengths.dtype
    )
    text_mask = text_mask + 1 > input_lengths[:, None]
    bert_out, _ = model.bert(input_ids, attention_mask=(~text_mask).astype(mx.int32))
    bert_out = maybe_fake_quant(
        bert_out, getattr(model.bert_encoder, "activation_quant", False)
    )
    d_en = model.bert_encoder(bert_out).transpose(0, 2, 1)
    prosody_style = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, prosody_style, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration_logits = model.predictor.duration_proj(x)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    decoder_style = ref_s[:, :128]
    return d, t_en, prosody_style, decoder_style, duration_logits


def build_pred_dur(duration_logits: mx.array) -> mx.array:
    duration = mx.sigmoid(duration_logits).sum(axis=-1)
    return mx.clip(mx.round(duration), a_min=1, a_max=None).astype(mx.int32)[0]


def build_alignment(pred_dur: mx.array, token_count: int) -> mx.array:
    indices = mx.concatenate([mx.repeat(mx.array(i), int(n)) for i, n in enumerate(pred_dur)])
    pred_aln_trg = mx.zeros((token_count, indices.shape[0]))
    pred_aln_trg[indices, mx.arange(indices.shape[0])] = 1
    return pred_aln_trg[None, :]


def forward_front_tail(
    model,
    d: mx.array,
    t_en: mx.array,
    prosody_style: mx.array,
    alignment: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    en = d.transpose(0, 2, 1) @ alignment
    f0_pred, n_pred = model.predictor.F0Ntrain(en, prosody_style)
    asr = t_en @ alignment
    return asr, f0_pred, n_pred


def forward_pred_dur(
    model,
    input_ids: mx.array,
    ref_s: mx.array,
) -> mx.array:
    _, _, _, _, duration_logits = forward_front_base(model, input_ids, ref_s)
    return build_pred_dur(duration_logits)


def forward_full_aligned(
    model,
    input_ids: mx.array,
    ref_s: mx.array,
    alignment: mx.array,
    rand_ini: mx.array,
    noise: mx.array,
) -> mx.array:
    d, t_en, prosody_style, decoder_style, _ = forward_front_base(
        model,
        input_ids,
        ref_s,
    )
    asr, f0_pred, n_pred = forward_front_tail(
        model,
        d,
        t_en,
        prosody_style,
        alignment,
    )
    return forward_decoder(
        model,
        asr,
        f0_pred,
        n_pred,
        decoder_style,
        rand_ini,
        noise,
    )


def forward_decoder(
    model,
    asr: mx.array,
    f0_pred: mx.array,
    n_pred: mx.array,
    style: mx.array,
    rand_ini: mx.array,
    noise: mx.array,
) -> mx.array:
    return _decoder_det(model.decoder, asr, f0_pred, n_pred, style, rand_ini, noise)


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_ids = mx.array(json.loads(args.input_ids_json), dtype=mx.int32)
    ref_s = mx.array(json.loads(args.ref_s_json), dtype=mx.float32)

    model = load(str(snapshot_path), lazy=False)
    try:
        d, t_en, prosody_style, decoder_style, duration_logits = forward_front_base(
            model,
            input_ids,
            ref_s,
        )
        pred_dur = build_pred_dur(duration_logits)
        alignment = build_alignment(pred_dur, input_ids.shape[1])
        asr, f0_pred, n_pred = forward_front_tail(
            model,
            d,
            t_en,
            prosody_style,
            alignment,
        )
        upsample_scale = int(model.decoder.generator.m_source.l_sin_gen.upsample_scale)
        harmonic_dim = int(model.decoder.generator.m_source.l_sin_gen.dim)
        noise_length = int(f0_pred.shape[1] * upsample_scale)

        mx.random.seed(0)
        rand_ini = mx.random.normal((f0_pred.shape[0], harmonic_dim))
        noise = mx.random.normal((f0_pred.shape[0], noise_length, harmonic_dim))

        pred_dur_path = output_dir / "pred_dur.mlxfn"
        full_aligned_path = output_dir / "full_aligned.mlxfn"
        for path in (pred_dur_path, full_aligned_path):
            if path.exists():
                path.unlink()
        mx.export_function(
            str(pred_dur_path),
            lambda ids, ref: forward_pred_dur(model, ids, ref),
            input_ids,
            ref_s,
        )
        mx.export_function(
            str(full_aligned_path),
            lambda ids, ref, align_x, seed0, seed1: forward_full_aligned(
                model,
                ids,
                ref,
                align_x,
                seed0,
                seed1,
            ),
            input_ids,
            ref_s,
            alignment,
            rand_ini,
            noise,
        )

        meta = {
            "snapshot_path": str(snapshot_path.resolve()),
            "input_shape": list(input_ids.shape),
            "ref_shape": list(ref_s.shape),
            "d_shape": list(d.shape),
            "text_shape": list(t_en.shape),
            "prosody_style_shape": list(prosody_style.shape),
            "decoder_style_shape": list(decoder_style.shape),
            "pred_dur_shape": list(pred_dur.shape),
            "pred_dur_sum": int(mx.sum(pred_dur).tolist()),
            "asr_shape": list(asr.shape),
            "f0_shape": list(f0_pred.shape),
            "n_shape": list(n_pred.shape),
            "alignment_shape": list(alignment.shape),
            "noise_length": noise_length,
            "harmonic_dim": harmonic_dim,
            "upsample_scale": upsample_scale,
            "sample_rate": int(model.sample_rate),
        }
        (output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(meta))
    finally:
        del model
        cleanup_mlx(mx)


if __name__ == "__main__":
    main()
