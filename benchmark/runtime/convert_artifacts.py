from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent
DEFAULT_RECIPES = RUNTIME_DIR / "conversion_recipes.yaml"
DEFAULT_OUT = RUNTIME_DIR / "artifacts.converted.yaml"
DEFAULT_HF_CATALOG = RUNTIME_DIR / "hf_artifacts.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert source Hugging Face models into local runtime artifacts "
            "and write an artifact map consumable by benchmark/runtime/run_all.py."
        )
    )
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument("--base-artifacts", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tools-root", type=Path)
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--engine", action="append", choices=["onnx", "coreml", "litert"])
    parser.add_argument("--platform", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip the converter when a matching artifact already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the recipe output directory before running the converter.",
    )
    parser.add_argument(
        "--no-fetch-tools",
        action="store_true",
        help="Fail if an external converter tool repo has not already been cloned.",
    )
    parser.add_argument(
        "--artifact-health-check",
        choices=["none", "run"],
        default="run",
        help="Load converted artifacts through the native health gate before unblocking platforms.",
    )
    parser.add_argument(
        "--allow-health-fail",
        action="store_true",
        help="Write failed health records instead of aborting conversion.",
    )
    parser.add_argument(
        "--allow-conversion-fail",
        action="store_true",
        help="Write failed conversion records and continue with later recipes.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.0,
        help="Skip conversion when the output filesystem has less free space.",
    )
    args = parser.parse_args()

    recipes = _read_yaml(args.recipes)
    converter = ArtifactConverter(
        recipes=recipes,
        recipes_path=args.recipes,
        base_artifacts=args.base_artifacts,
        out_path=args.out,
        output_root=args.output_root,
        tools_root=args.tools_root,
        model_filter=set(args.model_id or []),
        engine_filter=set(args.engine or []),
        platform_filter=set(args.platform or []),
        dry_run=args.dry_run,
        reuse_existing=args.reuse_existing,
        overwrite=args.overwrite,
        fetch_tools=not args.no_fetch_tools,
        artifact_health_check=args.artifact_health_check,
        allow_health_fail=args.allow_health_fail,
        allow_conversion_fail=args.allow_conversion_fail,
        min_free_gb=args.min_free_gb,
    )
    result = converter.run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


class ArtifactConverter:
    def __init__(
        self,
        *,
        recipes: dict[str, Any],
        recipes_path: Path,
        base_artifacts: Path | None,
        out_path: Path,
        output_root: Path | None,
        tools_root: Path | None,
        model_filter: set[str],
        engine_filter: set[str],
        platform_filter: set[str],
        dry_run: bool,
        reuse_existing: bool,
        overwrite: bool,
        fetch_tools: bool,
        artifact_health_check: str,
        allow_health_fail: bool,
        allow_conversion_fail: bool,
        min_free_gb: float,
    ) -> None:
        self.recipes = recipes
        self.recipes_path = recipes_path
        self.base_artifacts = base_artifacts
        self.out_path = out_path
        self.output_root = _resolve_path(
            output_root
            or Path(str(recipes.get("output_root") or "benchmark/artifacts/converted"))
        )
        self.tools_root = _resolve_path(
            tools_root
            or Path(str(recipes.get("tools_root") or "benchmark/artifacts/tools"))
        )
        self.model_filter = model_filter
        self.engine_filter = engine_filter
        self.platform_filter = platform_filter
        self.dry_run = dry_run
        self.reuse_existing = reuse_existing
        self.overwrite = overwrite
        self.fetch_tools = fetch_tools
        self.artifact_health_check = artifact_health_check
        self.allow_health_fail = allow_health_fail
        self.allow_conversion_fail = allow_conversion_fail
        self.min_free_gb = min_free_gb

    def run(self) -> dict[str, Any]:
        records = []
        artifact_map = self._base_artifact_map()
        for model_id, model in self._models():
            for engine, recipe in self._recipes_for(model):
                if self.engine_filter and engine not in self.engine_filter:
                    continue
                plan = self._plan(model_id, model, engine, recipe)
                selected_platforms = [
                    platform
                    for platform in plan["platforms"]
                    if not self.platform_filter or platform in self.platform_filter
                ]
                if not selected_platforms:
                    continue
                plan["platforms"] = selected_platforms
                if self.dry_run:
                    records.append({**plan, "state": "planned"})
                    continue
                record = self._execute(plan)
                records.append(record)
                if record["state"] in {"converted", "reused"}:
                    self._merge_artifact(artifact_map, record)
                elif record["state"] in {"conversion_failed", "preflight_skipped"}:
                    self._merge_blocker(artifact_map, record)

        if not self.dry_run:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_path.write_text(
                yaml.safe_dump(artifact_map, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        return {
            "version": 1,
            "recipes": str(self.recipes_path),
            "out": None if self.dry_run else str(self.out_path),
            "dry_run": self.dry_run,
            "converted_count": sum(1 for item in records if item["state"] == "converted"),
            "planned_count": sum(1 for item in records if item["state"] == "planned"),
            "reused_count": sum(1 for item in records if item["state"] == "reused"),
            "health_failed_count": sum(
                1 for item in records if item["state"] == "health_failed"
            ),
            "conversion_failed_count": sum(
                1 for item in records if item["state"] == "conversion_failed"
            ),
            "preflight_skipped_count": sum(
                1 for item in records if item["state"] == "preflight_skipped"
            ),
            "records": records,
        }

    def _models(self) -> list[tuple[str, dict[str, Any]]]:
        raw = self.recipes.get("models") or {}
        models = [
            (str(model_id), model)
            for model_id, model in raw.items()
            if isinstance(model, dict)
        ]
        if self.model_filter:
            models = [
                (model_id, model)
                for model_id, model in models
                if model_id in self.model_filter
            ]
        return models

    def _recipes_for(self, model: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        raw = model.get("recipes") or {}
        if not isinstance(raw, dict):
            return []
        presets = self.recipes.get("presets") or {}
        result = []
        for engine, recipe in raw.items():
            if not isinstance(recipe, dict):
                continue
            preset_name = recipe.get("preset")
            preset = presets.get(preset_name) if isinstance(preset_name, str) else {}
            merged = {**(preset or {}), **recipe}
            merged.pop("preset", None)
            result.append((str(engine), merged))
        return result

    def _plan(
        self,
        model_id: str,
        model: dict[str, Any],
        engine: str,
        recipe: dict[str, Any],
    ) -> dict[str, Any]:
        source_model = str(recipe.get("source_model") or model.get("source_model"))
        if not source_model:
            raise SystemExit(f"Missing source_model for {model_id}/{engine}")
        raw_timeout = recipe.get("timeout_seconds")
        timeout_seconds: int | None = None
        if raw_timeout is not None and raw_timeout != "":
            try:
                timeout_seconds = int(raw_timeout)
            except (TypeError, ValueError) as error:
                raise SystemExit(
                    "timeout_seconds must be a positive integer for "
                    f"{model_id}/{engine}, got: {raw_timeout!r}"
                ) from error
            if timeout_seconds <= 0:
                raise SystemExit(
                    "timeout_seconds must be a positive integer for "
                    f"{model_id}/{engine}, got: {raw_timeout!r}"
                )
        output_dir = self.output_root / model_id / engine
        exporter = str(recipe.get("exporter") or "custom")
        extra_args, ignored_extra_args = _normalized_extra_args(
            exporter,
            recipe.get("extra_args") or [],
        )
        with_args = _normalized_with_packages(recipe.get("extra_with") or [])
        context = {
            "model_id": model_id,
            "engine": engine,
            "source_model": source_model,
            "output_dir": str(output_dir),
            "export_task": str(recipe.get("export_task") or recipe.get("task") or ""),
            "opset": str(recipe.get("opset") or ""),
            "extra_args": extra_args,
            "with_args": with_args,
        }
        for key, value in recipe.items():
            if isinstance(value, (str, int, float, bool)):
                context.setdefault(str(key), str(value))
        tool = self._tool_context(recipe)
        context.update(tool)
        return {
            "model_id": model_id,
            "engine": engine,
            "task": str(model.get("task") or recipe.get("runtime_task") or "text"),
            "source_model": source_model,
            "exporter": exporter,
            "export_task": context["export_task"],
            "output_dir": str(output_dir),
            "platforms": [str(item) for item in recipe.get("platforms") or []],
            "provider_by_platform": recipe.get("provider_by_platform") or {},
            "delegate_by_platform": recipe.get("delegate_by_platform") or {},
            "artifact_candidates": [
                str(item) for item in recipe.get("artifact_candidates") or []
            ],
            "command": _expand_command(recipe.get("command") or [], context),
            "env": _expand_env(recipe.get("env"), context),
            "timeout_seconds": timeout_seconds,
            "report_path": str(output_dir / "conversion_record.json"),
            "ignored_extra_args": ignored_extra_args,
            **tool,
        }

    def _execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(plan["output_dir"])
        artifact = _find_artifact(output_dir, plan["artifact_candidates"])
        if artifact is not None and self.reuse_existing:
            return self._record_with_health(
                plan,
                artifact=artifact,
                state="reused",
                returncode=0,
            )
        preflight = self._preflight(plan)
        if preflight is not None:
            return preflight
        self._ensure_tool(plan)
        if output_dir.exists() and any(output_dir.iterdir()):
            if self.overwrite:
                shutil.rmtree(output_dir)
            else:
                raise SystemExit(
                    f"Output directory is not empty: {output_dir}. "
                    "Pass --reuse-existing or --overwrite."
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        if not plan["command"]:
            raise SystemExit(f"Missing converter command for {plan['model_id']}/{plan['engine']}")
        log_path = output_dir / "conversion.log"
        timeout_seconds = plan.get("timeout_seconds")
        run_env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            **_converter_cache_env(),
            **{
                str(key): str(value)
                for key, value in (plan.get("env") or {}).items()
            },
        }
        with log_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    plan["command"],
                    cwd=ROOT,
                    env=run_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                timeout_record = self._record_failure(
                    plan,
                    state="conversion_failed",
                    returncode=124,
                    reason=(
                        f"Converter timed out after {timeout_seconds}s. "
                        f"See {_rel(log_path)}."
                    ),
                    log_path=log_path,
                    extra={
                        "timeout_seconds": timeout_seconds,
                        "timed_out": True,
                    },
                )
                if self.allow_conversion_fail:
                    return timeout_record
                raise SystemExit(timeout_record["reason"])
        if completed.returncode != 0:
            classified = _classify_conversion_failure(log_path, completed.returncode)
            record = self._record_failure(
                plan,
                state="conversion_failed",
                returncode=completed.returncode,
                reason=(
                    f"Converter exited with code {completed.returncode}. "
                    f"See {_rel(log_path)}."
                ),
                log_path=log_path,
                extra=classified,
            )
            if self.allow_conversion_fail:
                return record
            raise SystemExit(record["reason"])
        artifact = _find_artifact(output_dir, plan["artifact_candidates"])
        if artifact is None:
            record = self._record_failure(
                plan,
                state="conversion_failed",
                returncode=completed.returncode,
                reason=(
                    "Converter finished but no artifact matched "
                    f"{plan['artifact_candidates']} in {output_dir}."
                ),
                log_path=log_path,
            )
            if self.allow_conversion_fail:
                return record
            raise SystemExit(record["reason"])
        return self._record_with_health(
            plan,
            artifact=artifact,
            state="converted",
            returncode=completed.returncode,
        )

    def _preflight(self, plan: dict[str, Any]) -> dict[str, Any] | None:
        if self.min_free_gb <= 0:
            return None
        output_dir = Path(plan["output_dir"])
        usage = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else ROOT)
        free_gb = usage.free / 1_000_000_000
        if free_gb >= self.min_free_gb:
            return None
        return self._record_failure(
            plan,
            state="preflight_skipped",
            returncode=0,
            reason=(
                f"Skipped because free space is {free_gb:.1f} GB, below "
                f"--min-free-gb {self.min_free_gb:.1f}."
            ),
            extra={"free_gb": free_gb, "min_free_gb": self.min_free_gb},
        )

    def _record_failure(
        self,
        plan: dict[str, Any],
        *,
        state: str,
        returncode: int,
        reason: str,
        log_path: Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._record(
            plan,
            artifact=None,
            state=state,
            returncode=returncode,
            reason=reason,
            log_path=log_path,
            extra=extra,
        )

    def _record_with_health(
        self,
        plan: dict[str, Any],
        *,
        artifact: Path,
        state: str,
        returncode: int,
    ) -> dict[str, Any]:
        health_checks = self._health_checks(plan, artifact)
        health_passed = all(item.get("passed") for item in health_checks)
        record_state = state if health_passed else "health_failed"
        record = self._record(
            plan,
            artifact=artifact,
            state=record_state,
            returncode=returncode,
            health_checks=health_checks,
        )
        if health_checks and not health_passed and not self.allow_health_fail:
            raise SystemExit(
                f"Artifact health check failed for {plan['model_id']}/{plan['engine']}. "
                f"See {plan['report_path']}"
            )
        return record

    def _health_checks(self, plan: dict[str, Any], artifact: Path) -> list[dict[str, Any]]:
        if self.artifact_health_check == "none":
            return []
        checks = []
        for platform in plan["platforms"]:
            report = Path(plan["output_dir"]) / f"artifact_health_{platform}.json"
            cmd = _artifact_health_command(plan, artifact, platform, report)
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    **_converter_cache_env(),
                },
                check=False,
            )
            checks.append(
                {
                    "platform": platform,
                    "command": cmd,
                    "report": _rel(report),
                    "returncode": completed.returncode,
                    "passed": completed.returncode == 0,
                }
            )
        return checks

    def _record(
        self,
        plan: dict[str, Any],
        *,
        artifact: Path | None,
        state: str,
        returncode: int,
        reason: str | None = None,
        log_path: Path | None = None,
        extra: dict[str, Any] | None = None,
        health_checks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        record = {
            **plan,
            "state": state,
            "returncode": returncode,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if artifact is not None:
            record["artifact"] = str(artifact)
        if reason is not None:
            record["reason"] = reason
        if log_path is not None:
            record["log_path"] = _rel(log_path)
        if extra:
            record.update(extra)
        if health_checks is not None:
            record["health_checks"] = health_checks
            record["health_passed"] = all(item.get("passed") for item in health_checks)
        report_path = Path(plan["report_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return record

    def _merge_artifact(
        self,
        artifact_map: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        models = artifact_map.setdefault("models", {})
        model = models.setdefault(
            record["model_id"],
            {
                "source_model": record["source_model"],
                "task": record["task"],
                "artifact_coverage": "converted",
                "platforms": {},
            },
        )
        model.setdefault("source_model", record["source_model"])
        model.setdefault("task", record["task"])
        if model.get("blocked_platforms"):
            model["artifact_coverage"] = "partial"
        else:
            model["artifact_coverage"] = "converted"
        platforms = model.setdefault("platforms", {})
        health_by_platform = {
            str(item["platform"]): item for item in record.get("health_checks") or []
        }
        for platform in record["platforms"]:
            cell = {
                "engine": record["engine"],
                "artifact": _rel(Path(record["artifact"])),
                "artifact_source": "converted",
                "source_uri": f"converted://{record['model_id']}/{record['engine']}",
                "converted_from": record["source_model"],
                "conversion_report": _rel(Path(record["report_path"])),
                "unblock_platform": True,
                "task": record["task"],
            }
            provider = (record.get("provider_by_platform") or {}).get(platform)
            delegate = (record.get("delegate_by_platform") or {}).get(platform)
            if provider:
                cell["provider"] = provider
            if delegate:
                cell["delegate"] = delegate
            health = health_by_platform.get(platform)
            if health:
                cell["artifact_health_report"] = health["report"]
                cell["artifact_health_passed"] = bool(health.get("passed"))
            platforms[platform] = cell

    def _merge_blocker(
        self,
        artifact_map: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        models = artifact_map.setdefault("models", {})
        model = models.setdefault(
            record["model_id"],
            {
                "source_model": record["source_model"],
                "task": record["task"],
                "artifact_coverage": "partial",
                "platforms": {},
            },
        )
        model.setdefault("source_model", record["source_model"])
        model.setdefault("task", record["task"])
        model["artifact_coverage"] = "partial"
        reason = str(record.get("reason") or "Conversion did not produce an artifact.")
        failure_class = record.get("failure_class")
        failure_reason = record.get("failure_reason")
        for platform in record["platforms"]:
            platforms = model.get("platforms") or {}
            platform_cell = (
                platforms.get(platform)
                if isinstance(platforms, dict)
                else None
            )
            if isinstance(platform_cell, dict) and platform_cell.get("artifact"):
                blocked_engines = model.setdefault("blocked_engines", {})
                engine_blockers = blocked_engines.setdefault(platform, {})
                engine_blockers[record["engine"]] = reason
                engine_reports = model.setdefault("blocked_engine_reports", {})
                platform_reports = engine_reports.setdefault(platform, {})
                platform_reports[record["engine"]] = _rel(Path(record["report_path"]))
                if isinstance(failure_class, str) and failure_class:
                    engine_failure_classes = model.setdefault(
                        "blocked_engine_failure_classes",
                        {},
                    )
                    platform_classes = engine_failure_classes.setdefault(platform, {})
                    platform_classes[record["engine"]] = failure_class
                if isinstance(failure_reason, str) and failure_reason:
                    engine_failure_reasons = model.setdefault(
                        "blocked_engine_failure_reasons",
                        {},
                    )
                    platform_reasons = engine_failure_reasons.setdefault(platform, {})
                    platform_reasons[record["engine"]] = failure_reason
                if record.get("log_path"):
                    engine_logs = model.setdefault("blocked_engine_logs", {})
                    platform_logs = engine_logs.setdefault(platform, {})
                    platform_logs[record["engine"]] = str(record["log_path"])
                continue

            blocked = model.setdefault("blocked_platforms", {})
            reports = model.setdefault("blocked_platform_reports", {})
            blocked[platform] = reason
            reports[platform] = _rel(Path(record["report_path"]))
            if isinstance(failure_class, str) and failure_class:
                classes = model.setdefault("blocked_platform_failure_classes", {})
                classes[platform] = failure_class
            if isinstance(failure_reason, str) and failure_reason:
                reasons = model.setdefault("blocked_platform_failure_reasons", {})
                reasons[platform] = failure_reason
            if record.get("log_path"):
                logs = model.setdefault("blocked_platform_logs", {})
                logs[platform] = str(record["log_path"])

    def _base_artifact_map(self) -> dict[str, Any]:
        if self.base_artifacts:
            return _read_yaml(self.base_artifacts)
        catalog_path = _resolve_path(
            Path(str(self.recipes.get("source_artifact_catalog") or DEFAULT_HF_CATALOG))
        )
        catalog = _read_yaml(catalog_path) if catalog_path.exists() else {}
        return {
            "version": 1,
            "source_catalog": _rel(self.recipes_path),
            "defaults": catalog.get("defaults") or {},
            "models": {},
        }

    def _tool_context(self, recipe: dict[str, Any]) -> dict[str, str]:
        repo = recipe.get("tool_repo")
        if not repo:
            return {}
        name = str(recipe.get("tool_name") or _safe_tool_name(str(repo)))
        tool_dir = self.tools_root / name
        result = {
            "tool_repo": str(repo),
            "tool_name": name,
            "tool_dir": str(tool_dir),
        }
        if recipe.get("tool_ref"):
            result["tool_ref"] = str(recipe["tool_ref"])
        return result

    def _ensure_tool(self, plan: dict[str, Any]) -> None:
        repo = plan.get("tool_repo")
        if not repo:
            return
        tool_dir = Path(str(plan["tool_dir"]))
        if tool_dir.exists():
            return
        if not self.fetch_tools:
            raise SystemExit(
                f"Missing converter tool repo {repo} at {tool_dir}. "
                "Remove --no-fetch-tools or clone it manually."
            )
        tool_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1", str(repo), str(tool_dir)]
        _run_tool_command(cmd)
        ref = plan.get("tool_ref")
        if ref:
            _run_tool_command(["git", "checkout", str(ref)], cwd=tool_dir)


def _expand_command(command: list[Any], context: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for item in command:
        text = str(item)
        if text.startswith("{") and text.endswith("}"):
            key = text[1:-1]
            value = context.get(key)
            if isinstance(value, list):
                result.extend(str(arg) for arg in value)
                continue
            if value is None:
                continue
            result.append(str(value))
            continue
        result.append(text.format(**context))
    return result


def _expand_env(raw_env: Any, context: dict[str, Any]) -> dict[str, str]:
    if not isinstance(raw_env, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in raw_env.items():
        key_text = str(key).strip()
        if not key_text or value is None:
            continue
        result[key_text] = str(value).format(**context)
    return result


def _converter_cache_env(
    environ: dict[str, str] | None = None,
    *,
    benchmark_root: Path | None = None,
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    root = benchmark_root or (ROOT / "benchmark")
    defaults = {
        "UV_CACHE_DIR": root / ".uv_cache",
        "HF_HOME": root / ".hf_home",
        "XDG_CACHE_HOME": root / ".cache",
    }
    result = {
        key: str(path)
        for key, path in defaults.items()
        if not source.get(key)
    }
    if not source.get("HF_HUB_DISABLE_XET"):
        result["HF_HUB_DISABLE_XET"] = "1"
    return result


def _classify_conversion_failure(log_path: Path, returncode: int) -> dict[str, Any]:
    text = ""
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    lowered = text.lower()
    result: dict[str, Any] = {}
    failure_reason: str | None = None
    if returncode == 124:
        result["failure_class"] = "conversion_timeout"
        failure_reason = "Converter exceeded timeout."
    elif "mutex lock failed" in lowered:
        result["failure_class"] = "exporter_runtime_crash"
        failure_reason = "Exporter crashed while acquiring a runtime mutex."
    elif "llvm error: inconsistency in registered commandline options" in lowered:
        result["failure_class"] = "exporter_runtime_crash"
        failure_reason = "Exporter crashed due to LLVM command-line option collision."
    elif "no module named 'onnx2tf.ops.loop'" in lowered:
        result["failure_class"] = "onnx2tf_unsupported_operator_loop"
        failure_reason = "onnx2tf does not implement Loop op conversion."
    elif "loop op is not yet implemented." in lowered:
        result["failure_class"] = "onnx2tf_unsupported_operator_loop"
        failure_reason = "onnx2tf does not implement Loop op conversion."
    elif "onnx2tf/ops/sequenceempty.py" in lowered and "dict' object is not callable" in lowered:
        result["failure_class"] = "onnx2tf_sequenceempty_bug"
        failure_reason = "onnx2tf SequenceEmpty dtype lookup bug (dict called like function)."
    elif "onnx2tf/ops/unsqueeze.py" in lowered and "input_tensor_shape" in lowered and "unboundlocalerror" in lowered:
        result["failure_class"] = "onnx2tf_unsqueeze_shape_bug"
        failure_reason = (
            "onnx2tf Unsqueeze shape fallback bug "
            "(input_tensor_shape is unbound for unknown-rank tensors)."
        )
    elif "keyerror:" in lowered and "onnx2tf/ops/if.py" in lowered:
        result["failure_class"] = "onnx2tf_if_subgraph_binding_bug"
        failure_reason = (
            "onnx2tf failed resolving If subgraph tensors (KeyError in onnx2tf/ops/If.py)."
        )
    elif "keyerror:" in lowered and "onnx2tf/ops/gather.py" in lowered:
        result["failure_class"] = "onnx2tf_graph_binding_bug"
        failure_reason = (
            "onnx2tf failed resolving Gather input tensors (KeyError in onnx2tf/ops/Gather.py)."
        )
    elif "read this and deal with it. https://github.com/pinto0309/onnx2tf#parameter-replacement" in lowered:
        result["failure_class"] = "onnx2tf_parameter_replacement_required"
        failure_reason = "onnx2tf requested parameter-replacement JSON."
    elif "concat input dtypes must be compatible in flatbuffer_direct" in lowered:
        result["failure_class"] = "onnx2tf_concat_dtype_mismatch"
        failure_reason = (
            "onnx2tf flatbuffer_direct failed due to mixed input dtypes in Concat."
        )
    elif "onnx2tf/ops/slice.py" in lowered and "attributeerror: 'tuple' object has no attribute 'rank'" in lowered:
        result["failure_class"] = "onnx2tf_slice_shape_rank_bug"
        failure_reason = (
            "onnx2tf Slice bug: tuple shape object does not expose .rank."
        )
    elif "onnx2tf/ops/slice.py" in lowered and "input 'y' of 'sub' op has type int64 that does not match type int32" in lowered:
        result["failure_class"] = "onnx2tf_slice_dtype_mismatch"
        failure_reason = "onnx2tf Slice lowering produced int32/int64 subtraction mismatch."
    elif "no such file or directory: 'onnxsim'" in lowered:
        result["failure_class"] = "converter_dependency_missing"
        failure_reason = "onnxsim executable is missing."
    elif "onnx_op_name:" in lowered and "onnx2tf" in lowered:
        result["failure_class"] = "onnx2tf_conversion_failed"
        failure_reason = "onnx2tf failed on a specific ONNX operator."
    elif "cannot import name 'check_model_inputs'" in lowered:
        result["failure_class"] = "transformers_api_mismatch"
        failure_reason = "Exporter expects a Transformers API removed in current version."
    elif "object has no attribute 'text_config'" in lowered:
        result["failure_class"] = "transformers_config_mismatch"
        failure_reason = "Model config is incompatible with current Transformers exporter."
    elif "unrecognized model in" in lowered and "`model_type` key" in lowered:
        result["failure_class"] = "model_architecture_unsupported"
        failure_reason = "Model architecture is not recognized by the exporter."
    elif "out of memory" in lowered or "cuda out of memory" in lowered:
        result["failure_class"] = "conversion_oom"
        failure_reason = "Conversion process ran out of memory."
    elif returncode == 134:
        result["failure_class"] = "exporter_runtime_crash"
        failure_reason = "Exporter aborted with SIGABRT (return code 134)."
    elif returncode != 0:
        result["failure_class"] = "conversion_failed"

    log_tail = _log_tail(text)
    if failure_reason:
        result["failure_reason"] = failure_reason
    elif log_tail:
        result["failure_reason"] = log_tail
    return result


def _log_tail(text: str, *, max_lines: int = 12, max_chars: int = 1200) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


def _normalized_extra_args(
    exporter: str,
    extra_args: list[Any],
) -> tuple[list[str], list[str]]:
    args = [str(arg) for arg in extra_args]
    if exporter != "coreml-llm":
        return args, []
    unsupported_flags = {"--trust-remote-code"}
    kept: list[str] = []
    ignored: list[str] = []
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in unsupported_flags:
            ignored.append(arg)
            continue
        if any(arg.startswith(f"{flag}=") for flag in unsupported_flags):
            ignored.append(arg)
            continue
        kept.append(arg)
        if arg in unsupported_flags and index + 1 < len(args):
            skip_next = True
    return kept, ignored


def _normalized_with_packages(extra_with: list[Any]) -> list[str]:
    result: list[str] = []
    for package in extra_with:
        text = str(package).strip()
        if not text:
            continue
        if text == "--with":
            continue
        if text.startswith("--with "):
            text = text[len("--with ") :].strip()
        if text.startswith("--with="):
            text = text[len("--with=") :].strip()
        if not text:
            continue
        result.extend(["--with", text])
    return result


def _artifact_health_command(
    plan: dict[str, Any],
    artifact: Path,
    platform: str,
    report: Path,
) -> list[str]:
    engine = plan["engine"]
    if engine == "onnx":
        cmd = ["uv", "run", "--group", "onnx-convert", "python"]
    else:
        cmd = [sys.executable]
    cmd.extend(
        [
            str(RUNTIME_DIR / "artifact_health.py"),
            "--engine",
            engine,
            "--platform",
            platform,
            "--artifact",
            str(artifact),
            "--out",
            str(report),
        ]
    )
    provider = (plan.get("provider_by_platform") or {}).get(platform)
    delegate = (plan.get("delegate_by_platform") or {}).get(platform)
    if engine == "onnx" and provider:
        cmd.extend(["--provider", str(provider)])
    if engine == "litert" and delegate:
        cmd.extend(["--delegate", str(delegate)])
    return cmd


def _find_artifact(output_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns or ["*"]:
        matches = sorted(
            path for path in output_dir.glob(pattern) if _is_runtime_artifact(path)
        )
        if matches:
            return matches[0]
    return None


def _is_runtime_artifact(path: Path) -> bool:
    if path.is_file():
        if path.suffix.lower() == ".json":
            return _is_pipeline_artifact(path)
        return path.suffix.lower() in {
            ".onnx",
            ".tflite",
            ".task",
            ".litertlm",
        }
    if path.is_dir():
        return path.suffix.lower() in {".mlmodelc", ".mlpackage"}
    return False


def _is_pipeline_artifact(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    return data.get("format") in {
        "dart_inference.coreml_pipeline.v1",
        "dart_inference.onnx_pipeline.v1",
        "dart_inference.litert_pipeline.v1",
        "dart_mlx_ffi.coreml_pipeline.v1",
        "dart_mlx_ffi.onnx_pipeline.v1",
        "dart_mlx_ffi.litert_pipeline.v1",
    }


def _resolve_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return ROOT / expanded


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_yaml(path: Path) -> dict[str, Any]:
    decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


def _safe_tool_name(repo: str) -> str:
    name = repo.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name.lower()


def _run_tool_command(cmd: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(cmd, cwd=cwd or ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"Tool setup command failed with exit code {completed.returncode}: "
            + " ".join(cmd)
        )


if __name__ == "__main__":
    main()
