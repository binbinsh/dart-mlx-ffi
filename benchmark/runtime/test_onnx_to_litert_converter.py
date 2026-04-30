from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNTIME_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME_DIR / "converters"))

from onnx_to_litert import (
    _build_onnx2tf_runners,
    _find_auto_prf_path,
    _parse_source_candidate,
    _patch_external_data_resave_source,
    _patch_sequence_lookup_source,
    _patch_slice_source,
    _patch_sequenceempty_source,
    _patch_trilu_source,
    _patch_unsqueeze_source,
    _pick_tflite,
    _resolve_onnx_artifact,
    _run_onnx2tf_attempt,
    _uv_archive_onnx2tf_module_paths,
)


class OnnxToLiteRtConverterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dmf_onnx_to_litert_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_pick_tflite_prefers_dynamic_range_file(self) -> None:
        plain = self.tmp / "model.tflite"
        drq = self.tmp / "model_dynamic_range_quant.tflite"
        plain.write_bytes(b"tflite")
        drq.write_bytes(b"tflite")

        picked = _pick_tflite(self.tmp)
        self.assertEqual(picked, drq)

    def test_pick_tflite_can_prefer_float16_file(self) -> None:
        drq = self.tmp / "model_dynamic_range_quant.tflite"
        fp16 = self.tmp / "model_float16.tflite"
        drq.write_bytes(b"tflite")
        fp16.write_bytes(b"tflite")

        picked = _pick_tflite(self.tmp, preference="float16")
        self.assertEqual(picked, fp16)

    def test_resolve_onnx_artifact_uses_existing_local_path(self) -> None:
        artifact = self.tmp / "model.onnx"
        artifact.write_bytes(b"onnx")

        resolved = _resolve_onnx_artifact(
            repo="ignored/repo",
            artifact=str(artifact),
            revision=None,
            prefer_local=True,
        )
        self.assertEqual(resolved, artifact.resolve())

    def test_parse_source_candidate(self) -> None:
        candidate = _parse_source_candidate("repo/name::onnx/model.onnx")
        self.assertEqual(candidate.repo, "repo/name")
        self.assertEqual(candidate.artifact, "onnx/model.onnx")

    def test_find_auto_prf_path(self) -> None:
        auto = self.tmp / "auto.json"
        auto.write_text("{}", encoding="utf-8")
        log = (
            "Auto-generated replacement JSON saved to: "
            f"{auto}\n"
        )
        self.assertEqual(_find_auto_prf_path(log, output_dir=self.tmp), str(auto))

    def test_patch_sequenceempty_source(self) -> None:
        original = (
            "sequence_dtype = "
            "ONNX_DTYPES_TO_TF_DTYPES(graph_node.attrs.get('dtype', 1))"
        )
        patched, changed = _patch_sequenceempty_source(original)
        self.assertTrue(changed)
        self.assertIn(
            "ONNX_DTYPES_TO_TF_DTYPES[graph_node.attrs.get('dtype', 1)]",
            patched,
        )

    def test_patch_unsqueeze_source(self) -> None:
        original = """

@print_node_info
def make_node():
    axes = tf_layers_dict[graph_node_input_2.name]['tf_node'] \\
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    if input_tensor.shape != tf.TensorShape(None):
        input_tensor_shape = list(input_tensor.shape)
        tensor_rank = len(input_tensor_shape)
    elif graph_node_output.shape is not None:
        input_tensor_shape = [
            dim for idx, dim in enumerate(graph_node_output.shape) if idx not in axes
        ]
        input_tensor_shape = [
            dim if not isinstance(dim, str) else None for dim in input_tensor_shape
        ]
        tensor_rank = len(input_tensor_shape)
"""
        patched, changed = _patch_unsqueeze_source(original)
        self.assertTrue(changed)
        self.assertIn("def _lookup_unsqueeze_axes(", patched)
        self.assertIn("_lookup_unsqueeze_axes(tf_layers_dict, graph_node_input_2)", patched)
        self.assertIn("else:", patched)
        self.assertIn("input_tensor_shape = [None]", patched)

    def test_patch_slice_source(self) -> None:
        original = """            cliped_values = tf.cast(
                tf.subtract(
                    1,
                    tf.clip_by_value(t=begin_, clip_value_min=0, clip_value_max=1),
                ),
                dtype=tf.int32,
            )
            cliped_values = tf.cast(
                tf.subtract(
                    1,
                    tf.clip_by_value(t=end_, clip_value_min=0, clip_value_max=1),
                ),
                dtype=tf.int32,
            )
"""
        patched, changed = _patch_slice_source(original)
        self.assertTrue(changed)
        self.assertIn("begin_clipped = tf.clip_by_value", patched)
        self.assertIn("end_clipped = tf.clip_by_value", patched)

    def test_patch_sequence_lookup_source(self) -> None:
        original = """import onnx_graphsurgeon as gs

@print_node_info
def make_node(*, graph_node: gs.Node, tf_layers_dict: dict, **kwargs: dict):
    input_sequence = tf_layers_dict[graph_node_input_1.name]['tf_node'] \\
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
"""
        patched, changed = _patch_sequence_lookup_source(original)
        self.assertTrue(changed)
        self.assertIn("def _lookup_tf_node(", patched)
        self.assertIn("_lookup_tf_node(tf_layers_dict, graph_node_input_1)", patched)

    def test_patch_trilu_dynamic_shape_source(self) -> None:
        original = """    if k is not None:
        if k > tensor_shape[-1]:
            k = tensor_shape[-1]
        elif k < 0 - tensor_shape[-2]:
            k = 0 - tensor_shape[-2]
"""
        patched, changed = _patch_trilu_source(original)

        self.assertTrue(changed)
        self.assertIn("tensor_shape[-1] is not None", patched)
        self.assertIn("tensor_shape[-2] is not None", patched)

    def test_patch_external_data_resave_source(self) -> None:
        original = "                        onnx.save(estimated_graph, f=input_onnx_file_path)"
        patched, changed = _patch_external_data_resave_source(original)

        self.assertTrue(changed)
        self.assertIn("onnx.save_model(", patched)
        self.assertIn("save_as_external_data=True", patched)

    def test_build_onnx2tf_runners_includes_isolated_fallback(self) -> None:
        runners = _build_onnx2tf_runners(
            fallback_isolated_onnx2tf2=True,
            isolated_onnx2tf2_version="2.4.0",
            isolated_tensorflow_version="2.19.0",
            isolated_tf_keras_version="2.19.0",
            isolated_workdir=self.tmp,
        )
        self.assertEqual(runners[0].id, "local")
        self.assertEqual(len(runners), 3)
        self.assertEqual(runners[1].id, "isolated_onnx2tf_2.4.0")
        self.assertEqual(runners[1].cwd, self.tmp.resolve())
        self.assertIn("--no-project", runners[1].command_prefix)
        self.assertEqual(runners[1].extra_args, [])
        self.assertEqual(
            runners[2].id,
            "isolated_onnx2tf_2.4.0_tf_converter",
        )
        self.assertEqual(runners[2].cwd, self.tmp.resolve())
        self.assertEqual(runners[2].extra_args, ["-tb", "tf_converter", "-agje"])
        self.assertIn("PYTHONNOUSERSITE=1", runners[2].command_prefix)

    def test_build_onnx2tf_runners_without_isolated_fallback(self) -> None:
        runners = _build_onnx2tf_runners(
            fallback_isolated_onnx2tf2=False,
            isolated_onnx2tf2_version="2.4.0",
            isolated_tensorflow_version="2.19.0",
            isolated_tf_keras_version="2.19.0",
            isolated_workdir=None,
        )
        self.assertEqual([runner.id for runner in runners], ["local"])

    def test_uv_archive_paths_honor_uv_cache_dir(self) -> None:
        module = (
            self.tmp
            / "uv-cache"
            / "archive-v0"
            / "pkg"
            / "lib"
            / "python3.12"
            / "site-packages"
            / "onnx2tf"
            / "ops"
            / "Unsqueeze.py"
        )
        module.parent.mkdir(parents=True)
        module.write_text("# patched target", encoding="utf-8")

        with mock.patch.dict(
            "os.environ",
            {"UV_CACHE_DIR": str(self.tmp / "uv-cache")},
            clear=False,
        ):
            paths = _uv_archive_onnx2tf_module_paths("Unsqueeze.py")

        self.assertIn(module, paths)

    def test_run_onnx2tf_attempt_marks_timeout(self) -> None:
        log = self.tmp / "attempt.log"
        result = _run_onnx2tf_attempt(
            command=[sys.executable, "-c", "import time; time.sleep(2)"],
            output_dir=self.tmp,
            log_path=log,
            cwd=None,
            attempt_timeout_seconds=1,
        )
        self.assertEqual(result["returncode"], 124)
        self.assertTrue(result["timed_out"])
        text = log.read_text(encoding="utf-8")
        self.assertIn("onnx2tf attempt timed out", text)


if __name__ == "__main__":
    unittest.main()
