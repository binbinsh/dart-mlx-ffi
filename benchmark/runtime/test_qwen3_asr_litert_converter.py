from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR / "converters"))

import qwen3_asr_onnx_to_litert
from qwen3_asr_onnx_to_litert import convert_qwen3_asr_onnx_to_litert


class Qwen3AsrLiteRtConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_qwen3_asr_litert_test_"))
        self.snapshot = self.tmp / "snapshot"
        self.snapshot.mkdir()
        for name in (
            "encoder.int4.onnx",
            "decoder_init.int4.onnx",
            "decoder_step.int4.onnx",
            "decoder_weights.int4.data",
            "embed_tokens.bin",
            "tokenizer.json",
        ):
            (self.snapshot / name).write_bytes(b"data")
        (self.snapshot / "config.json").write_text("{}", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_converts_component_bundle_and_writes_runner_manifest(self) -> None:
        def fake_convert_source(**kwargs: object) -> dict[str, object]:
            output_dir = Path(str(kwargs["output_dir"]))
            model_dir = output_dir / "onnx2tf_out_source1"
            model_dir.mkdir(parents=True, exist_ok=True)
            selected = model_dir / "model_dynamic_range_quant.tflite"
            selected.write_bytes(b"TFL3")
            source = kwargs["source"]
            return {
                "success": True,
                "repo": source.repo,
                "artifact": source.artifact,
                "onnx_path": source.artifact,
                "onnx2tf_output_dir": str(model_dir),
                "selected_tflite": str(selected),
                "runner": "fake",
            }

        with mock.patch.object(
            qwen3_asr_onnx_to_litert,
            "_convert_source",
            side_effect=fake_convert_source,
        ), mock.patch.object(
            qwen3_asr_onnx_to_litert,
            "_write_static_encoder_input_onnx",
            side_effect=lambda source, destination, **_: Path(destination).write_bytes(
                Path(source).read_bytes()
            ),
        ), mock.patch.object(
            qwen3_asr_onnx_to_litert,
            "_write_encoder_layout_prf",
            side_effect=lambda _source, destination, **_: Path(destination).write_text(
                "{\"format_version\":1,\"operations\":[]}\n",
                encoding="utf-8",
            ),
        ), mock.patch.object(
            qwen3_asr_onnx_to_litert,
            "_write_decoder_compatible_onnx",
            side_effect=lambda source, destination: Path(destination).write_bytes(
                Path(source).read_bytes()
            ),
        ), mock.patch.object(
            qwen3_asr_onnx_to_litert,
            "_write_decoder_layout_prf",
            side_effect=lambda _source, destination: Path(destination).write_text(
                "{\"format_version\":1,\"operations\":[]}\n",
                encoding="utf-8",
            ),
        ):
            report = convert_qwen3_asr_onnx_to_litert(
                repo="andrewleech/qwen3-asr-1.7b-onnx",
                revision=None,
                output_dir=self.tmp / "out",
                snapshot_dir=self.snapshot,
                patch_onnx2tf=False,
                fallback_isolated_onnx2tf2=False,
            )

        out = self.tmp / "out"
        bundle_path = out / "qwen3_asr_litert_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "converted")
        self.assertEqual(
            bundle["format"],
            "dart_mlx_ffi.qwen3_asr_litert_bundle.v1",
        )
        self.assertEqual(bundle["runner"], "Qwen3AsrNativeRunner.loadLiteRtBundle")
        self.assertEqual(bundle["components"]["encoder"], "encoder.tflite")
        self.assertTrue((out / "encoder.tflite").is_file())
        self.assertTrue((out / "decoder_init.tflite").is_file())
        self.assertTrue((out / "decoder_step.tflite").is_file())
        self.assertTrue((out / "embed_tokens.bin").is_file())
        self.assertTrue((out / "tokenizer.json").is_file())
        self.assertEqual(Path(str(report["artifact"])).resolve(), bundle_path.resolve())

    def test_default_download_patterns_do_not_fetch_fp32_fallback(self) -> None:
        patterns = qwen3_asr_onnx_to_litert._allow_patterns(prefer_fp32=False)

        self.assertIn("encoder.int4.onnx", patterns)
        self.assertIn("decoder_weights.int4.data", patterns)
        self.assertNotIn("encoder.onnx", patterns)
        self.assertNotIn("decoder_weights.data", patterns)

    def test_missing_snapshot_files_are_downloaded_individually(self) -> None:
        snapshot = self.tmp / "partial_snapshot"
        snapshot.mkdir()
        for name in (
            *qwen3_asr_onnx_to_litert.REQUIRED_SIDECAR_FILES,
            "tokenizer.json",
        ):
            (snapshot / name).write_bytes(b"data")
        for name in ("decoder_init.int4.onnx", "decoder_step.int4.onnx"):
            (snapshot / name).write_bytes(b"data")
        downloaded: list[str] = []

        def fake_download(**kwargs: object) -> str:
            filename = str(kwargs["filename"])
            downloaded.append(filename)
            (snapshot / filename).write_bytes(b"data")
            return str(snapshot / filename)

        with mock.patch.object(
            qwen3_asr_onnx_to_litert,
            "hf_hub_download_with_fallback",
            side_effect=fake_download,
        ):
            qwen3_asr_onnx_to_litert._ensure_required_snapshot_files(
                snapshot=snapshot,
                repo="andrewleech/qwen3-asr-1.7b-onnx",
                revision=None,
                prefer_fp32=False,
                prefer_local=False,
                endpoint="https://hf-mirror.com",
                fallback_endpoint="https://hf-mirror.com",
            )

        self.assertEqual(
            sorted(downloaded),
            ["decoder_weights.int4.data", "encoder.int4.onnx"],
        )

    def test_encoder_conversion_defaults_to_static_channel_first_mel(self) -> None:
        args = qwen3_asr_onnx_to_litert._component_onnx2tf_extra_args(
            "encoder",
            [],
        )

        self.assertIn("-kt", args)
        self.assertIn("mel", args)
        self.assertEqual(
            qwen3_asr_onnx_to_litert._encoder_mel_frames_from_args([]),
            3000,
        )
        self.assertEqual(
            qwen3_asr_onnx_to_litert._component_onnx2tf_extra_args(
                "decoder_step",
                [],
            ),
            ["-nuo", "-nuonag"],
        )

    def test_user_encoder_shape_override_is_preserved(self) -> None:
        args = qwen3_asr_onnx_to_litert._component_onnx2tf_extra_args(
            "encoder",
            ["-ois", "mel:1,128,1200"],
            ["-prf", "generated.json"],
        )

        self.assertEqual(args.count("-ois"), 1)
        self.assertIn("mel:1,128,1200", args)
        self.assertIn("generated.json", args)
        self.assertEqual(
            qwen3_asr_onnx_to_litert._encoder_mel_frames_from_args(args),
            1200,
        )

    def test_user_prf_override_is_preserved(self) -> None:
        args = qwen3_asr_onnx_to_litert._component_onnx2tf_extra_args(
            "encoder",
            ["-prf", "custom.json"],
            ["-prf", "generated.json"],
        )

        self.assertIn("custom.json", args)
        self.assertNotIn("generated.json", args)

    def test_decoder_init_conversion_uses_static_prompt_inputs(self) -> None:
        args = qwen3_asr_onnx_to_litert._component_onnx2tf_extra_args(
            "decoder_init",
            [],
            [
                "-ois",
                "input_ids:1,128",
                "position_ids:1,128",
                "audio_features:1,104,2048",
                "audio_offset:1",
            ],
        )

        self.assertIn("-ois", args)
        self.assertIn("input_ids:1,128", args)
        self.assertIn("audio_features:1,104,2048", args)
        self.assertIn("-nuo", args)
        self.assertIn("-nuonag", args)
        self.assertIn("-kat", args)
        self.assertIn("audio_offset", args)

    def test_user_decoder_init_shape_override_is_preserved(self) -> None:
        args = ["-ois", "input_ids:1,64", "audio_features:1,80,2048"]

        self.assertEqual(
            qwen3_asr_onnx_to_litert._decoder_init_static_shape_from_args(args),
            (64, 80),
        )
        self.assertEqual(
            qwen3_asr_onnx_to_litert._component_onnx2tf_extra_args(
                "decoder_init",
                args,
                ["-ois", "input_ids:1,128"],
            ).count("-ois"),
            1,
        )

    def test_decoder_graphdef_failure_gets_actionable_error(self) -> None:
        log = self.tmp / "graphdef.log"
        log.write_text(
            "google.protobuf.message.DecodeError: Error parsing message with "
            "type 'tensorflow.GraphDef'\nValueError: Invalid GraphDef\n",
            encoding="utf-8",
        )

        error = qwen3_asr_onnx_to_litert._component_failure_error(
            "decoder_init",
            {
                "error": "onnx2tf failed",
                "attempt_runs": [{"log_path": str(log)}],
            },
        )

        self.assertIn("single-graph protobuf limits", error)
        self.assertIn("chunked pipeline", error)

    def test_rewrites_qwen3_asr_encoder_custom_ops(self) -> None:
        from onnx import TensorProto, helper

        graph = helper.make_graph(
            [
                helper.make_node(
                    "BiasGelu",
                    ["x", "bias"],
                    ["gelu_out"],
                    name="BiasGelu_0",
                ),
                helper.make_node(
                    "SkipLayerNormalization",
                    ["gelu_out", "skip", "gamma", "beta", "ln_bias"],
                    ["norm_out", "", "", "residual_out"],
                    name="SkipLayerNorm_0",
                    epsilon=1e-5,
                    axis=-1,
                    stash_type=1,
                ),
            ],
            "qwen3_asr_rewrite_test",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4]),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4]),
                helper.make_tensor_value_info("skip", TensorProto.FLOAT, [1, 4]),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [4]),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, [4]),
                helper.make_tensor_value_info("ln_bias", TensorProto.FLOAT, [4]),
            ],
            [helper.make_tensor_value_info("norm_out", TensorProto.FLOAT, [1, 4])],
        )
        model = helper.make_model(graph)

        qwen3_asr_onnx_to_litert._rewrite_encoder_custom_ops(model)

        ops = [node.op_type for node in model.graph.node]
        outputs = [output for node in model.graph.node for output in node.output]
        self.assertNotIn("BiasGelu", ops)
        self.assertNotIn("SkipLayerNormalization", ops)
        self.assertIn("Gelu", ops)
        self.assertIn("LayerNormalization", ops)
        self.assertIn("residual_out", outputs)

    def test_rewrites_qwen3_asr_decoder_custom_ops(self) -> None:
        import numpy as np
        from onnx import TensorProto, helper
        from onnx import numpy_helper

        graph = helper.make_graph(
            [
                helper.make_node(
                    "SimplifiedLayerNormalization",
                    ["x", "gamma"],
                    ["norm_out"],
                    name="RMSNorm_0",
                    epsilon=1e-6,
                    axis=-1,
                    stash_type=1,
                ),
                helper.make_node(
                    "MatMulNBits",
                    ["norm_out", "packed_q4", "scales", "zero_points"],
                    ["matmul_out"],
                    name="MatMulNBits_0",
                    domain="com.microsoft",
                    K=4,
                    N=1,
                    bits=4,
                    block_size=4,
                ),
            ],
            "qwen3_asr_decoder_rewrite_test",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4]),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [4]),
            ],
            [helper.make_tensor_value_info("matmul_out", TensorProto.FLOAT, [1, 1])],
            initializer=[
                numpy_helper.from_array(
                    np.array([[[0x21, 0x43]]], dtype=np.uint8),
                    name="packed_q4",
                ),
                numpy_helper.from_array(
                    np.array([[0.5]], dtype=np.float32),
                    name="scales",
                ),
                numpy_helper.from_array(
                    np.array([[1]], dtype=np.uint8),
                    name="zero_points",
                ),
            ],
        )
        model = helper.make_model(graph)

        qwen3_asr_onnx_to_litert._rewrite_decoder_custom_ops(model)

        ops = [node.op_type for node in model.graph.node]
        outputs = [output for node in model.graph.node for output in node.output]
        initializers = [tensor.name for tensor in model.graph.initializer]
        self.assertNotIn("SimplifiedLayerNormalization", ops)
        self.assertNotIn("MatMulNBits", ops)
        self.assertIn("ReduceMean", ops)
        self.assertIn("Sqrt", ops)
        self.assertIn("MatMul", ops)
        self.assertIn("matmul_out", outputs)
        self.assertIn("RMSNorm_0_axes", initializers)
        self.assertNotIn("packed_q4", initializers)
        self.assertIn("MatMulNBits_0_dequant_weight_f16", initializers)

    def test_dequantizes_qwen3_asr_matmulnbits_weight(self) -> None:
        import numpy as np
        import onnx
        from onnx import numpy_helper

        weight = qwen3_asr_onnx_to_litert._dequantize_matmulnbits_weight(
            onnx,
            weight_tensor=numpy_helper.from_array(
                np.array([[[0x21, 0x43]]], dtype=np.uint8),
                name="packed_q4",
            ),
            scale_tensor=numpy_helper.from_array(
                np.array([[0.5]], dtype=np.float32),
                name="scales",
            ),
            zero_point_tensor=numpy_helper.from_array(
                np.array([[1]], dtype=np.uint8),
                name="zero_points",
            ),
            attrs={"K": 4, "N": 1, "bits": 4, "block_size": 4},
        )

        self.assertEqual(weight.shape, (4, 1))
        self.assertEqual(weight.dtype, np.float16)
        self.assertEqual(weight[:, 0].astype(np.float32).tolist(), [0, 0.5, 1, 1.5])


if __name__ == "__main__":
    unittest.main()
