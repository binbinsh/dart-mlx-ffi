from __future__ import annotations

import argparse
import json
import os
import platform as host_platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent
RUNNERS = RUNTIME_DIR / "runners"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one runtime matrix cell and emit a promotion verdict."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--platform", default=_default_platform())
    parser.add_argument("--engine", required=True, choices=["coreml", "onnx", "litert", "mlx"])
    parser.add_argument("--artifact", required=True)
    parser.add_argument(
        "--baseline-engine",
        default="coreml-llm",
        choices=["coreml-llm", "mlx", "coreml", "onnx", "litert"],
    )
    parser.add_argument("--baseline-artifact")
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--baseline-publish-report", type=Path)
    parser.add_argument("--baseline-publish-model-id")
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--raw-baseline-report", type=Path)
    parser.add_argument("--config", type=Path, default=RUNTIME_DIR / "models.yaml")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=RUNTIME_DIR / "fixtures" / "tiny_input.json",
    )
    parser.add_argument("--prompt")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=RUNTIME_DIR / "fixtures" / "text_prompt.txt",
    )
    parser.add_argument(
        "--task",
        choices=[
            "text",
            "function",
            "embedding",
            "vlm",
            "audio",
            "tts",
            "vad",
            "tensor",
        ],
        default="text",
    )
    parser.add_argument("--tools-file", type=Path)
    parser.add_argument("--tools-json")
    parser.add_argument("--embedding-query")
    parser.add_argument("--embedding-query-file", type=Path)
    parser.add_argument("--embedding-dim")
    parser.add_argument("--image-file", type=Path)
    parser.add_argument("--audio-file", type=Path)
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--iters", default="5")
    parser.add_argument("--max-tokens", default="64")
    parser.add_argument("--num-threads")
    parser.add_argument("--provider")
    parser.add_argument("--delegate")
    parser.add_argument("--coreml-mode", choices=["decode", "prefill"])
    parser.add_argument("--litert-section-index")
    parser.add_argument("--hf-cache-root")
    parser.add_argument("--require-provider", action="store_true")
    parser.add_argument("--require-delegate", action="store_true")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "benchmark" / "out" / "runtime",
    )
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cell = RuntimeCell(args)
    result = cell.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.allow_fail and not result["passed"]:
        raise SystemExit(1)


class RuntimeCell:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_dir = args.out_root / args.model_id / args.platform
        baseline_name = _safe_name(args.baseline_engine)
        if args.baseline_engine == args.engine:
            baseline_name = f"{baseline_name}_baseline"
        self.baseline_path = self.out_dir / f"{baseline_name}.json"
        self.candidate_path = self.out_dir / f"{_safe_name(args.engine)}.json"
        self.report_path = self.out_dir / "report.json"
        self.verdict_path = self.out_dir / "verdict.json"

    def run(self) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        commands = []
        if self.args.baseline_report:
            self.baseline_path = self.args.baseline_report
        else:
            baseline_cmd = self._baseline_command()
            commands.append(baseline_cmd)
            self._run(baseline_cmd)

        if self.args.candidate_report:
            self.candidate_path = self.args.candidate_report
        else:
            candidate_cmd = self._candidate_command()
            commands.append(candidate_cmd)
            self._run(candidate_cmd)

        combine_cmd = [
            sys.executable,
            str(RUNTIME_DIR / "combine_reports.py"),
            "--baseline",
            str(self.baseline_path),
            "--candidate",
            str(self.candidate_path),
            "--out",
            str(self.report_path),
            "--model-id",
            self.args.model_id,
        ]
        commands.append(combine_cmd)
        self._run(combine_cmd)

        compare_cmd = [
            sys.executable,
            str(RUNTIME_DIR / "compare.py"),
            str(self.report_path),
            "--out",
            str(self.verdict_path),
        ]
        compare_cmd.extend(_threshold_args(self.args.config))
        compare_cmd.extend(_device_profile_args(self.args))
        commands.append(compare_cmd)
        if self.args.dry_run:
            return {
                "model_id": self.args.model_id,
                "platform": self.args.platform,
                "engine": self.args.engine,
                "baseline_engine": self.args.baseline_engine,
                "passed": True,
                "dry_run": True,
                "paths": self._paths(),
                "commands": [_display_command(cmd) for cmd in commands],
            }
        compare = self._run(compare_cmd, check=False)
        verdict = _read_json(self.verdict_path).get("verdict", {})
        return {
            "model_id": self.args.model_id,
            "platform": self.args.platform,
            "engine": self.args.engine,
            "baseline_engine": self.args.baseline_engine,
            "passed": bool(verdict.get("passed")) and compare.returncode == 0,
            "paths": self._paths(),
            "commands": [_display_command(cmd) for cmd in commands],
        }

    def _paths(self) -> dict[str, str]:
        return {
            "baseline": str(self.baseline_path),
            "candidate": str(self.candidate_path),
            "report": str(self.report_path),
            "verdict": str(self.verdict_path),
        }

    def _baseline_command(self) -> list[str]:
        engine = self.args.baseline_engine
        artifact = self.args.baseline_artifact or self.args.artifact
        if engine == "coreml-llm":
            cmd = [
                sys.executable,
                str(RUNNERS / "coreml_llm_runner.py"),
                "--model-id",
                self.args.model_id,
                "--platform",
                self.args.platform,
                "--artifact",
                artifact,
                "--warmup",
                self.args.warmup,
                "--iters",
                self.args.iters,
                "--max-tokens",
                self.args.max_tokens,
                "--task",
                self.args.task,
                "--out",
                str(self.baseline_path),
            ]
            if self.args.tools_file:
                cmd.extend(["--tools-file", str(self.args.tools_file)])
            if self.args.tools_json:
                cmd.extend(["--tools-json", self.args.tools_json])
            if self.args.embedding_query:
                cmd.extend(["--embedding-query", self.args.embedding_query])
            if self.args.embedding_query_file:
                cmd.extend(
                    ["--embedding-query-file", str(self.args.embedding_query_file)]
                )
            if self.args.embedding_dim:
                cmd.extend(["--embedding-dim", self.args.embedding_dim])
            if self.args.image_file:
                cmd.extend(["--image-file", str(self.args.image_file)])
            if self.args.raw_baseline_report:
                cmd.extend(["--raw-report", str(self.args.raw_baseline_report)])
            elif self.args.prompt is not None:
                cmd.extend(["--prompt", self.args.prompt])
            elif not (
                self.args.task == "embedding"
                and (self.args.embedding_query or self.args.embedding_query_file)
            ):
                cmd.extend(["--prompt-file", str(self.args.prompt_file)])
            return cmd
        if engine == "mlx":
            cmd = [
                sys.executable,
                str(RUNNERS / "mlx_runner.py"),
                "--model-id",
                self.args.model_id,
                "--platform",
                self.args.platform,
                "--artifact",
                artifact,
                "--out",
                str(self.baseline_path),
            ]
            if self.args.raw_baseline_report:
                cmd.extend(["--raw-report", str(self.args.raw_baseline_report)])
            if self.args.baseline_publish_report:
                cmd.extend(
                    ["--publish-report", str(self.args.baseline_publish_report)]
                )
            if self.args.baseline_publish_model_id:
                cmd.extend(
                    ["--publish-model-id", self.args.baseline_publish_model_id]
                )
            return cmd
        if engine in {"coreml", "onnx", "litert"}:
            return self._native_command(
                engine=engine,
                artifact=artifact,
                out_path=self.baseline_path,
            )
        raise ValueError(f"Unsupported baseline engine: {engine}")

    def _candidate_command(self) -> list[str]:
        if self.args.engine in {"coreml", "onnx", "litert"}:
            return self._native_command(
                engine=self.args.engine,
                artifact=self.args.artifact,
                out_path=self.candidate_path,
            )
        cmd = [
            sys.executable,
            str(RUNNERS / "mlx_runner.py"),
            "--model-id",
            self.args.model_id,
            "--platform",
            self.args.platform,
            "--artifact",
            self.args.artifact,
            "--out",
            str(self.candidate_path),
        ]
        return cmd

    def _native_command(
        self,
        *,
        engine: str,
        artifact: str,
        out_path: Path,
    ) -> list[str]:
        runner = {
            "coreml": "coreml_runner.py",
            "onnx": "ort_runner.py",
            "litert": "litert_runner.py",
        }[engine]
        cmd = [
            sys.executable,
            str(RUNNERS / runner),
            "--model-id",
            self.args.model_id,
            "--platform",
            self.args.platform,
            "--artifact",
            artifact,
            "--out",
            str(out_path),
            "--input-json",
            str(self.args.input_json),
            "--warmup",
            self.args.warmup,
            "--iters",
            self.args.iters,
        ]
        if self.args.num_threads:
            cmd.extend(["--num-threads", self.args.num_threads])
        if self.args.hf_cache_root:
            cmd.extend(["--hf-cache-root", self.args.hf_cache_root])
        if engine == "coreml" and self.args.coreml_mode:
            cmd.extend(["--coreml-mode", self.args.coreml_mode])
        if engine == "onnx":
            if self.args.provider:
                cmd.extend(["--provider", self.args.provider])
            if self.args.require_provider:
                cmd.append("--require-provider")
        if engine == "litert":
            if self.args.delegate:
                cmd.extend(["--delegate", self.args.delegate])
            if self.args.litert_section_index:
                cmd.extend(["--litert-section-index", self.args.litert_section_index])
            if self.args.require_delegate:
                cmd.append("--require-delegate")
        return cmd

    def _run(self, cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
        if self.args.dry_run:
            print(_display_command(cmd))
            return subprocess.CompletedProcess(cmd, 0)
        env = dict(os.environ)
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return subprocess.run(cmd, cwd=ROOT, check=check, env=env)


def _default_platform() -> str:
    system = host_platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system or "unknown"


def _safe_name(value: str) -> str:
    return value.replace("-", "_")


def _display_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def _read_json(path: Path) -> dict[str, Any]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return decoded


def _threshold_args(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    thresholds = (config.get("support_policy") or {}).get("thresholds") or {}
    mapping = {
        "min_speed_ratio": "--min-speed-ratio",
        "max_ttft_ratio": "--max-ttft-ratio",
        "max_peak_memory_ratio": "--max-peak-memory-ratio",
        "min_embedding_cosine": "--min-embedding-cosine",
        "max_embedding_l2": "--max-embedding-l2",
        "max_abs_diff": "--max-abs-diff",
        "required_coreml_preferred_device": "--required-coreml-preferred-device",
    }
    args: list[str] = []
    for key, flag in mapping.items():
        if thresholds.get(key) is not None:
            args.extend([flag, str(thresholds[key])])
    if thresholds.get("require_device_profile"):
        args.append("--require-device-profile")
    return args


def _device_profile_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    if args.engine == "onnx" and args.require_provider and args.provider:
        result.extend(["--required-provider", args.provider])
    if args.engine == "litert" and args.require_delegate and args.delegate:
        result.extend(["--required-delegate", args.delegate])
    return result


if __name__ == "__main__":
    main()
