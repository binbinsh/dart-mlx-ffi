from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Android runtime matrix cell through adb."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--platform", default="android", choices=["android"])
    parser.add_argument("--engine", default="litert", choices=["litert", "onnx"])
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--baseline-engine", choices=["litert", "onnx"])
    parser.add_argument("--baseline-artifact")
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--config", type=Path, default=RUNTIME_DIR / "models.yaml")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=RUNTIME_DIR / "fixtures" / "tiny_input.json",
    )
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--task", default="text")
    parser.add_argument("--tools-file", type=Path)
    parser.add_argument("--tools-json")
    parser.add_argument("--embedding-query")
    parser.add_argument("--embedding-query-file", type=Path)
    parser.add_argument("--embedding-dim")
    parser.add_argument("--image-file", type=Path)
    parser.add_argument("--audio-file", type=Path)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "benchmark" / "out" / "runtime",
    )
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--remote-baseline-report")
    parser.add_argument("--remote-candidate-report")
    parser.add_argument("--device-id")
    parser.add_argument(
        "--remote-dir",
        default="/data/local/tmp/dart_mlx_ffi_runtime",
    )
    parser.add_argument("--device-runner")
    parser.add_argument("--device-command")
    parser.add_argument("--push", action="append", default=[])
    parser.add_argument("--pull", action="append", default=[])
    parser.add_argument("--warmup", default="1")
    parser.add_argument("--iters", default="5")
    parser.add_argument("--max-tokens", default="64")
    parser.add_argument("--num-threads")
    parser.add_argument("--delegate")
    parser.add_argument("--provider")
    parser.add_argument("--coreml-mode")
    parser.add_argument("--litert-section-index")
    parser.add_argument("--hf-cache-root")
    parser.add_argument("--require-delegate", action="store_true")
    parser.add_argument("--require-provider", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runner = AdbRuntimeCell(args)
    result = runner.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not args.allow_fail and not result["passed"]:
        raise SystemExit(1)


class AdbRuntimeCell:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_dir = args.out_root / args.model_id / args.platform
        self.baseline_engine = args.baseline_engine or args.engine
        self.local_baseline = args.baseline_report or (
            self.out_dir / f"{self.baseline_engine}_adb_baseline.json"
        )
        self.local_candidate = (
            args.candidate_report or self.out_dir / f"{args.engine}_adb.json"
        )
        self.remote_baseline = args.remote_baseline_report or (
            f"{args.remote_dir.rstrip('/')}/{self.baseline_engine}_baseline.json"
        )
        self.remote_candidate = args.remote_candidate_report or (
            f"{args.remote_dir.rstrip('/')}/{args.engine}_candidate.json"
        )
        self.commands: list[list[str]] = []
        self.remote_files: dict[str, str] = {}
        self._temp_files: list[Path] = []

    def run(self) -> dict[str, Any]:
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._require_adb()
            self._shell(["mkdir", "-p", self.args.remote_dir])

            remote_input = self._push_input_json(self.args.input_json)
            self.remote_files = self._push_named_fixtures()
            remote_artifact = self._resolve_artifact()
            remote_baseline_artifact = self._resolve_baseline_artifact()
            remote_runner = self._resolve_runner()
            for spec in self.args.push:
                self._push_spec(spec)

            if self.args.baseline_report is None:
                command = self._device_command(
                    artifact=remote_baseline_artifact,
                    input_json=remote_input,
                    runner=remote_runner,
                    engine=self.baseline_engine,
                    out_report=self.remote_baseline,
                )
                if command is not None:
                    self._shell(["sh", "-c", command])
                self._pull_file(self.remote_baseline, self.local_baseline)

            command = self._device_command(
                artifact=remote_artifact,
                input_json=remote_input,
                runner=remote_runner,
                engine=self.args.engine,
                out_report=self.remote_candidate,
            )
            if command is not None:
                self._shell(["sh", "-c", command])

            self._pull_file(self.remote_candidate, self.local_candidate)
            for spec in self.args.pull:
                self._pull_spec(spec)

            compare_cmd = self._compare_command()
            self._run(compare_cmd, check=not self.args.allow_fail)
            passed = True
            verdict_path = self.out_dir / "verdict.json"
            if verdict_path.exists():
                verdict = json.loads(verdict_path.read_text(encoding="utf-8")).get(
                    "verdict", {}
                )
                passed = bool(verdict.get("passed"))
            return {
                "model_id": self.args.model_id,
                "platform": self.args.platform,
                "engine": self.args.engine,
                "passed": passed,
                "paths": {
                    "candidate": str(self.local_candidate),
                    "baseline": str(self.local_baseline),
                    "report": str(self.out_dir / "report.json"),
                    "verdict": str(verdict_path),
                },
                "remote": {
                    "device_id": self.args.device_id,
                    "remote_dir": self.args.remote_dir,
                    "baseline_report": self.remote_baseline,
                    "candidate_report": self.remote_candidate,
                    "files": self.remote_files,
                },
                "commands": [_display_command(cmd) for cmd in self.commands],
            }
        finally:
            self._cleanup_temp_files()

    def _require_adb(self) -> None:
        if shutil.which("adb") is None:
            raise RuntimeError("adb is not available on PATH")

    def _push_input_json(self, source: Path) -> str:
        rewritten = self._rewrite_input_sidecars(source)
        return self._push_file(rewritten or source, "input.json")

    def _rewrite_input_sidecars(self, source: Path) -> Path | None:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        inputs = payload.get("inputs")
        if inputs is None:
            inputs = payload
        if not isinstance(inputs, dict):
            return None

        remote_asset_dir = self._remote_join("input_assets")
        sidecar_index = 0
        changed = False
        for spec in inputs.values():
            if not isinstance(spec, dict):
                continue
            key = _sidecar_key(spec)
            if key is None:
                continue
            raw = spec[key]
            if not isinstance(raw, str) or _is_device_path(raw):
                continue
            local = _resolve_local_data_path(source.parent, raw)
            if not local.exists():
                raise RuntimeError(f"Missing input sidecar: {local}")
            if not changed:
                self._shell(["mkdir", "-p", remote_asset_dir])
            remote = f"{remote_asset_dir}/{sidecar_index}_{local.name}"
            sidecar_index += 1
            self._adb(["push", str(local), remote])
            spec[key] = remote
            changed = True

        if not changed:
            return None
        temp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="dmf_adb_input_",
            delete=False,
            encoding="utf-8",
        )
        with temp:
            json.dump(payload, temp, indent=2, ensure_ascii=False)
            temp.write("\n")
        temp_path = Path(temp.name)
        self._temp_files.append(temp_path)
        return temp_path

    def _push_named_fixtures(self) -> dict[str, str]:
        files: dict[str, str] = {}
        for key in (
            "prompt_file",
            "tools_file",
            "embedding_query_file",
            "image_file",
            "audio_file",
        ):
            value = getattr(self.args, key)
            files[key] = str(value or "")
            files[f"remote_{key}"] = self._push_optional_fixture(value, key)
        return files

    def _push_optional_fixture(self, value: Path | None, key: str) -> str:
        if value is None:
            return ""
        raw = str(value)
        if _is_device_path(raw):
            return raw
        local = value.expanduser()
        if not local.exists():
            raise RuntimeError(f"Missing {key}: {local}")
        remote_dir = self._remote_join("fixtures")
        self._shell(["mkdir", "-p", remote_dir])
        remote = f"{remote_dir}/{key}_{local.name}"
        self._adb(["push", str(local), remote])
        return remote

    def _cleanup_temp_files(self) -> None:
        for path in self._temp_files:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _resolve_artifact(self) -> str:
        artifact = Path(self.args.artifact).expanduser()
        if artifact.exists():
            remote = f"{self.args.remote_dir.rstrip('/')}/{artifact.name}"
            self._adb(["push", str(artifact), remote])
            return remote
        return self.args.artifact

    def _resolve_baseline_artifact(self) -> str:
        artifact = Path(self.args.baseline_artifact or self.args.artifact).expanduser()
        if artifact.exists():
            remote = f"{self.args.remote_dir.rstrip('/')}/{artifact.name}"
            self._adb(["push", str(artifact), remote])
            return remote
        return self.args.baseline_artifact or self.args.artifact

    def _resolve_runner(self) -> str | None:
        runner = self.args.device_runner
        if runner is None:
            return None
        path = Path(runner).expanduser()
        if path.exists():
            remote = f"{self.args.remote_dir.rstrip('/')}/{path.name}"
            self._adb(["push", str(path), remote])
            self._shell(["chmod", "+x", remote])
            return remote
        return runner

    def _push_file(self, source: Path, name: str) -> str:
        remote = self._remote_join(name)
        self._adb(["push", str(source), remote])
        return remote

    def _remote_join(self, *parts: str) -> str:
        return "/".join([self.args.remote_dir.rstrip("/"), *parts])

    def _push_spec(self, spec: str) -> None:
        source, remote = _split_mapping(spec)
        self._adb(["push", source, remote])

    def _pull_file(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._adb(["pull", remote, str(local)])

    def _pull_spec(self, spec: str) -> None:
        remote, local = _split_mapping(spec)
        self._pull_file(remote, Path(local))

    def _device_command(
        self,
        *,
        artifact: str,
        input_json: str,
        runner: str | None,
        engine: str,
        out_report: str,
    ) -> str | None:
        values = {
            "model_id": self.args.model_id,
            "platform": self.args.platform,
            "engine": engine,
            "artifact": artifact,
            "input_json": input_json,
            "candidate_report": out_report,
            "out_report": out_report,
            "remote_dir": self.args.remote_dir,
            "task": self.args.task,
            "warmup": self.args.warmup,
            "iters": self.args.iters,
            "max_tokens": self.args.max_tokens,
            "delegate": self.args.delegate or "",
            "provider": self.args.provider or "",
            "coreml_mode": self.args.coreml_mode or "",
            "litert_section_index": self.args.litert_section_index or "",
            "hf_cache_root": self.args.hf_cache_root or "",
            "num_threads": self.args.num_threads or "",
            "runner": runner or "",
            "tools_json": self.args.tools_json or "",
            "embedding_query": self.args.embedding_query or "",
            "embedding_dim": self.args.embedding_dim or "",
            **self.remote_files,
        }
        if self.args.device_command:
            return self.args.device_command.format(**values)
        if runner is None:
            return None
        parts = [
            runner,
            "--model-id",
            self.args.model_id,
            "--platform",
            self.args.platform,
            "--engine",
            engine,
            "--artifact",
            artifact,
            "--input-json",
            input_json,
            "--warmup",
            self.args.warmup,
            "--iters",
            self.args.iters,
            "--out",
            out_report,
        ]
        if self.args.num_threads:
            parts.extend(["--num-threads", self.args.num_threads])
        if self.args.hf_cache_root:
            parts.extend(["--hf-cache-root", self.args.hf_cache_root])
        if self.args.coreml_mode:
            parts.extend(["--coreml-mode", self.args.coreml_mode])
        if self.args.litert_section_index:
            parts.extend(["--litert-section-index", self.args.litert_section_index])
        if self.args.delegate:
            parts.extend(["--delegate", self.args.delegate])
        if self.args.provider:
            parts.extend(["--provider", self.args.provider])
        if self.args.require_delegate:
            parts.append("--require-delegate")
        if self.args.require_provider:
            parts.append("--require-provider")
        return (
            f"LD_LIBRARY_PATH={shlex.quote(self.args.remote_dir)} "
            + " ".join(shlex.quote(part) for part in parts)
        )

    def _compare_command(self) -> list[str]:
        cmd = [
            sys.executable,
            str(RUNTIME_DIR / "run_matrix.py"),
            "--model-id",
            self.args.model_id,
            "--platform",
            self.args.platform,
            "--engine",
            self.args.engine,
            "--artifact",
            self.args.artifact,
            "--baseline-engine",
            self.baseline_engine,
            "--baseline-report",
            str(self.local_baseline),
            "--candidate-report",
            str(self.local_candidate),
            "--config",
            str(self.args.config),
            "--out-root",
            str(self.args.out_root),
            "--task",
            self.args.task,
            "--warmup",
            self.args.warmup,
            "--iters",
            self.args.iters,
            "--max-tokens",
            self.args.max_tokens,
        ]
        if self.args.prompt_file:
            cmd.extend(["--prompt-file", str(self.args.prompt_file)])
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
        if self.args.audio_file:
            cmd.extend(["--audio-file", str(self.args.audio_file)])
        if self.args.delegate:
            cmd.extend(["--delegate", self.args.delegate])
        if self.args.provider:
            cmd.extend(["--provider", self.args.provider])
        if self.args.coreml_mode:
            cmd.extend(["--coreml-mode", self.args.coreml_mode])
        if self.args.litert_section_index:
            cmd.extend(["--litert-section-index", self.args.litert_section_index])
        if self.args.require_delegate:
            cmd.append("--require-delegate")
        if self.args.require_provider:
            cmd.append("--require-provider")
        if self.args.allow_fail:
            cmd.append("--allow-fail")
        return cmd

    def _adb(self, arguments: list[str]) -> subprocess.CompletedProcess:
        cmd = ["adb"]
        if self.args.device_id:
            cmd.extend(["-s", self.args.device_id])
        cmd.extend(arguments)
        return self._run(cmd)

    def _shell(self, arguments: list[str]) -> subprocess.CompletedProcess:
        return self._adb(["shell", *arguments])

    def _run(
        self,
        cmd: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        self.commands.append(cmd)
        if self.args.dry_run:
            print(_display_command(cmd))
            return subprocess.CompletedProcess(cmd, 0)
        env = dict(os.environ)
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return subprocess.run(cmd, cwd=ROOT, check=check, env=env)


def _split_mapping(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Expected mapping in local:remote form: {spec}")
    left, right = spec.split(":", 1)
    if not left or not right:
        raise ValueError(f"Invalid mapping: {spec}")
    return left, right


def _sidecar_key(spec: dict[str, Any]) -> str | None:
    if isinstance(spec.get("file"), str):
        return "file"
    if isinstance(spec.get("path"), str):
        return "path"
    return None


def _resolve_local_data_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _is_device_path(value: str) -> bool:
    prefixes = (
        "/data/",
        "/sdcard/",
        "/storage/",
        "/mnt/",
        "content://",
    )
    return value.startswith(prefixes)


def _display_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


if __name__ == "__main__":
    main()
