from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent
DEFAULT_RECIPES = RUNTIME_DIR / "conversion_recipes.yaml"
DEFAULT_OUT = RUNTIME_DIR / "artifacts.converted.yaml"
DEFAULT_HF_CATALOG = RUNTIME_DIR / "hf_artifacts.yaml"

from convert_artifacts_support import (
    _artifact_health_command,
    _classify_conversion_failure,
    _converter_cache_env,
    _existing_log_path,
    _expand_command,
    _expand_env,
    _find_artifact,
    _normalized_extra_args,
    _normalized_with_packages,
    _read_yaml,
    _rel,
    _resolve_path,
    _run_tool_command,
    _safe_tool_name,
    _seed_models_from_catalog,
)


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
    parser.add_argument(
        "--timeout-seconds-override",
        type=int,
        help=(
            "Override recipe timeout_seconds for every conversion in this run. "
            "Useful for capping long-running exports during probes."
        ),
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
        timeout_seconds_override=args.timeout_seconds_override,
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
        timeout_seconds_override: int | None = None,
        import_existing_records: bool = True,
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
        self.timeout_seconds_override = timeout_seconds_override
        self.import_existing_records = import_existing_records
        if (
            self.timeout_seconds_override is not None
            and self.timeout_seconds_override <= 0
        ):
            raise SystemExit("--timeout-seconds-override must be a positive integer.")

    def run(self) -> dict[str, Any]:
        records = []
        artifact_map = self._base_artifact_map()
        imported_records = (
            self._merge_existing_records(artifact_map)
            if self.import_existing_records and not self.dry_run
            else []
        )
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
            "imported_record_count": len(imported_records),
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
        if self.timeout_seconds_override is not None:
            timeout_seconds = self.timeout_seconds_override
        output_dir = self.output_root / model_id / engine
        exporter = str(recipe.get("exporter") or "custom")
        extra_args, ignored_extra_args = _normalized_extra_args(
            exporter,
            recipe.get("extra_args") or [],
        )
        extra_with = _normalized_with_packages(recipe.get("extra_with"))
        with_args: list[str] = []
        for package in extra_with:
            with_args.extend(["--with", package])
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
            "extra_with": extra_with,
            "preflight_blocked": bool(recipe.get("preflight_blocked")),
            "preflight_failure_class": recipe.get("preflight_failure_class"),
            "preflight_failure_reason": recipe.get("preflight_failure_reason"),
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
        cache_env = _converter_cache_env()
        run_env = {
            **os.environ,
            **cache_env,
            "PYTHONDONTWRITEBYTECODE": "1",
            **{
                str(key): str(value)
                for key, value in (plan.get("env") or {}).items()
            },
        }
        for key in (
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "PYTHONHOME",
            "CONDA_PREFIX",
            "__PYVENV_LAUNCHER__",
            "PYTHONEXECUTABLE",
        ):
            run_env.pop(key, None)
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
                        "failure_class": "conversion_timeout",
                        "failure_reason": "Converter exceeded timeout.",
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
        if plan.get("preflight_blocked"):
            failure_class = str(
                plan.get("preflight_failure_class") or "conversion_preflight_blocked"
            )
            failure_reason = str(
                plan.get("preflight_failure_reason")
                or "Recipe preflight blocked this conversion."
            )
            return self._record_failure(
                plan,
                state="preflight_skipped",
                returncode=0,
                reason=f"Skipped by recipe preflight: {failure_reason}",
                extra={
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                },
            )
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
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
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
            if record.get("runner"):
                cell["runner"] = record["runner"]
            health = health_by_platform.get(platform)
            if health:
                cell["artifact_health_report"] = health["report"]
                cell["artifact_health_passed"] = bool(health.get("passed"))
            platforms[platform] = cell
            self._clear_blockers_for_platform(
                model=model,
                platform=platform,
                engine=record["engine"],
            )

        if model.get("blocked_platforms"):
            model["artifact_coverage"] = "partial"
        else:
            model["artifact_coverage"] = "converted"

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
            if (
                isinstance(platforms, dict)
                and isinstance(platform_cell, dict)
                and self._is_stale_converted_cell(
                    platform_cell=platform_cell,
                    record=record,
                )
            ):
                platforms.pop(platform, None)
                if not platforms:
                    model.pop("platforms", None)
                platform_cell = None
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

    def _is_stale_converted_cell(
        self,
        *,
        platform_cell: dict[str, Any],
        record: dict[str, Any],
    ) -> bool:
        if str(platform_cell.get("engine") or "") != str(record.get("engine") or ""):
            return False
        if str(platform_cell.get("artifact_source") or "") != "converted":
            return False
        expected_source = f"converted://{record['model_id']}/{record['engine']}"
        source_uri = str(platform_cell.get("source_uri") or "")
        return source_uri == expected_source or source_uri == ""

    def _clear_blockers_for_platform(
        self,
        *,
        model: dict[str, Any],
        platform: str,
        engine: str,
    ) -> None:
        self._clear_platform_engine_entry(
            model=model,
            key="blocked_engines",
            platform=platform,
            engine=engine,
        )
        self._clear_platform_engine_entry(
            model=model,
            key="blocked_engine_reports",
            platform=platform,
            engine=engine,
        )
        self._clear_platform_engine_entry(
            model=model,
            key="blocked_engine_failure_classes",
            platform=platform,
            engine=engine,
        )
        self._clear_platform_engine_entry(
            model=model,
            key="blocked_engine_failure_reasons",
            platform=platform,
            engine=engine,
        )
        self._clear_platform_engine_entry(
            model=model,
            key="blocked_engine_logs",
            platform=platform,
            engine=engine,
        )
        self._clear_platform_entry(model=model, key="blocked_platforms", platform=platform)
        self._clear_platform_entry(
            model=model,
            key="blocked_platform_reports",
            platform=platform,
        )
        self._clear_platform_entry(
            model=model,
            key="blocked_platform_failure_classes",
            platform=platform,
        )
        self._clear_platform_entry(
            model=model,
            key="blocked_platform_failure_reasons",
            platform=platform,
        )
        self._clear_platform_entry(
            model=model,
            key="blocked_platform_logs",
            platform=platform,
        )

    def _clear_platform_engine_entry(
        self,
        *,
        model: dict[str, Any],
        key: str,
        platform: str,
        engine: str,
    ) -> None:
        root = model.get(key)
        if not isinstance(root, dict):
            return
        platform_map = root.get(platform)
        if not isinstance(platform_map, dict):
            return
        platform_map.pop(engine, None)
        if not platform_map:
            root.pop(platform, None)
        if not root:
            model.pop(key, None)

    def _clear_platform_entry(
        self,
        *,
        model: dict[str, Any],
        key: str,
        platform: str,
    ) -> None:
        root = model.get(key)
        if not isinstance(root, dict):
            return
        root.pop(platform, None)
        if not root:
            model.pop(key, None)

    def _base_artifact_map(self) -> dict[str, Any]:
        if self.base_artifacts:
            return _read_yaml(self.base_artifacts)
        catalog_path = _resolve_path(
            Path(str(self.recipes.get("source_artifact_catalog") or DEFAULT_HF_CATALOG))
        )
        catalog = _read_yaml(catalog_path) if catalog_path.exists() else {}
        inherited_models = {}
        if bool(self.recipes.get("seed_models_from_catalog")):
            inherited_models = _seed_models_from_catalog(catalog)
        return {
            "version": 1,
            "source_catalog": _rel(catalog_path),
            "defaults": catalog.get("defaults") or {},
            "models": inherited_models,
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

    def _merge_existing_records(self, artifact_map: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.output_root.exists():
            return []
        imported: list[dict[str, Any]] = []
        known_model_ids = self._known_model_ids(artifact_map)
        for path in sorted(self.output_root.glob("**/conversion_record.json")):
            record = self._read_existing_record(path, artifact_map)
            if record is None:
                continue
            if record["model_id"] not in known_model_ids:
                continue
            state = str(record.get("state") or "")
            if state in {"converted", "reused"}:
                artifact_text = record.get("artifact")
                if not isinstance(artifact_text, str) or not artifact_text:
                    continue
                artifact = Path(artifact_text)
                if not artifact.exists():
                    continue
                self._merge_artifact(artifact_map, record)
                imported.append(
                    {
                        "model_id": record["model_id"],
                        "engine": record["engine"],
                        "state": state,
                        "report_path": _rel(path),
                    }
                )
                continue
            if state in {"conversion_failed", "health_failed"} or (
                state == "preflight_skipped" and bool(record.get("preflight_blocked"))
            ):
                self._merge_blocker(artifact_map, record)
                imported.append(
                    {
                        "model_id": record["model_id"],
                        "engine": record["engine"],
                        "state": state,
                        "report_path": _rel(path),
                    }
                )
        return imported

    def _known_model_ids(self, artifact_map: dict[str, Any]) -> set[str]:
        known = {
            str(model_id)
            for model_id in (artifact_map.get("models") or {}).keys()
            if str(model_id)
        }
        known.update(
            str(model_id)
            for model_id in (self.recipes.get("models") or {}).keys()
            if str(model_id)
        )
        return known

    def _read_existing_record(
        self,
        path: Path,
        artifact_map: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(decoded, dict):
            return None
        model_id = str(decoded.get("model_id") or "").strip()
        engine = str(decoded.get("engine") or "").strip()
        if not model_id or not engine:
            return None
        platforms = decoded.get("platforms")
        if isinstance(platforms, list):
            normalized_platforms = [str(item) for item in platforms if str(item)]
        else:
            platform = str(decoded.get("platform") or "").strip()
            normalized_platforms = [platform] if platform else []
        if not normalized_platforms:
            return None
        model_defaults = (
            ((artifact_map.get("models") or {}).get(model_id) or {})
            if isinstance(artifact_map.get("models"), dict)
            else {}
        )
        source_model = str(
            decoded.get("source_model") or model_defaults.get("source_model") or ""
        )
        task = str(decoded.get("task") or model_defaults.get("task") or "text")
        record = dict(decoded)
        record["model_id"] = model_id
        record["engine"] = engine
        record["platforms"] = normalized_platforms
        record["source_model"] = source_model
        record["task"] = task
        record.setdefault("report_path", str(path))
        if (
            record.get("state") == "health_failed"
            and not isinstance(record.get("reason"), str)
        ):
            checks = record.get("health_checks")
            if isinstance(checks, list):
                failed = [
                    str(item.get("platform"))
                    for item in checks
                    if isinstance(item, dict) and not bool(item.get("passed"))
                ]
                if failed:
                    record["reason"] = (
                        "Artifact health check failed on "
                        + ", ".join(sorted(set(failed)))
                        + "."
                    )
        if str(record.get("state") or "") in {"conversion_failed", "preflight_skipped"}:
            log_path = _existing_log_path(record, path)
            if log_path is not None and log_path.exists():
                classified = _classify_conversion_failure(
                    log_path,
                    int(record.get("returncode") or 1),
                )
                if classified.get("failure_class"):
                    record.update(classified)
        return record

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

if __name__ == "__main__":
    main()
