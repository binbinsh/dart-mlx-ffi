from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from hf_download import hf_hub_download_with_fallback, snapshot_download_with_fallback


DEFAULT_FALLBACK_REPOS = ("inclusionAI/Ming-Lite-Omni",)

_EXTERNAL_MODULES = {
    "argparse",
    "collections",
    "copy",
    "dataclasses",
    "functools",
    "gc",
    "itertools",
    "json",
    "math",
    "numpy",
    "os",
    "pathlib",
    "re",
    "scipy",
    "sys",
    "time",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "typing",
}


def prepare_patched_source(
    *,
    source_model: str,
    work_dir: Path,
    fallback_repos: Iterable[str] = DEFAULT_FALLBACK_REPOS,
    revision: str | None = None,
    endpoint: str | None = None,
    fallback_endpoint: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Download source snapshot and inject missing dynamic modules if needed."""
    work_dir.mkdir(parents=True, exist_ok=True)
    source_dir = work_dir / "source_model"
    snapshot_download_with_fallback(
        repo_id=source_model,
        revision=revision,
        local_dir=str(source_dir),
        local_dir_use_symlinks="auto",
        endpoint=endpoint,
        fallback_endpoint=fallback_endpoint,
    )
    queue = list(_auto_map_files(source_dir))
    seen = set(queue)
    downloaded: list[str] = []
    missing: list[str] = []
    repositories = [source_model, *[repo for repo in fallback_repos if repo]]

    while queue:
        filename = queue.pop(0)
        target = source_dir / filename
        if target.exists():
            for dependency in _local_dependencies(target):
                if dependency not in seen:
                    queue.append(dependency)
                    seen.add(dependency)
            continue

        downloaded_path = _download_first_available(
            repositories=repositories,
            filename=filename,
            endpoint=endpoint,
            fallback_endpoint=fallback_endpoint,
        )
        if downloaded_path is None:
            missing.append(filename)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(downloaded_path).read_bytes())
        downloaded.append(filename)
        for dependency in _local_dependencies(target):
            if dependency not in seen:
                queue.append(dependency)
                seen.add(dependency)

    rewritten = _rewrite_local_imports(source_dir)
    report: dict[str, object] = {
        "source_model": source_model,
        "source_dir": str(source_dir),
        "fallback_repos": repositories[1:],
        "patched_files": sorted(downloaded),
        "missing_files": sorted(set(missing)),
        "rewritten_files": rewritten,
    }
    return source_dir, report


def _auto_map_files(source_dir: Path) -> list[str]:
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return []
    decoded = json.loads(config_path.read_text(encoding="utf-8"))
    auto_map = decoded.get("auto_map")
    if not isinstance(auto_map, dict):
        return []
    files: set[str] = set()
    for value in auto_map.values():
        for token in _flatten_auto_map_value(value):
            module = token.split(".", 1)[0].strip()
            if module:
                files.add(f"{module}.py")
    return sorted(files)


def _flatten_auto_map_value(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _download_first_available(
    *,
    repositories: list[str],
    filename: str,
    endpoint: str | None = None,
    fallback_endpoint: str | None = None,
) -> str | None:
    for repo in repositories:
        try:
            return hf_hub_download_with_fallback(
                repo_id=repo,
                filename=filename,
                endpoint=endpoint,
                fallback_endpoint=fallback_endpoint,
            )
        except Exception:
            continue
    return None


def _local_dependencies(path: Path) -> list[str]:
    dependencies: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("from "):
            module = stripped[5:].split(" import ", 1)[0].strip()
        elif stripped.startswith("import "):
            module = stripped[7:].split(" as ", 1)[0].split(",", 1)[0].strip()
        else:
            continue
        module = module.lstrip(".")
        if not module:
            continue
        root = module.split(".", 1)[0].strip()
        if not root or root in _EXTERNAL_MODULES:
            continue
        dependencies.add(f"{root}.py")
    return sorted(dependencies)


def _rewrite_local_imports(source_dir: Path) -> list[str]:
    local_modules = {path.stem for path in source_dir.glob("*.py")}
    rewritten: list[str] = []
    from_pattern = re.compile(r"^(\s*)from\s+([A-Za-z_]\w*)\s+import\s+(.+)$")
    import_pattern = re.compile(r"^(\s*)import\s+([A-Za-z_]\w*)\s*$")
    for path in sorted(source_dir.glob("*.py")):
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        changed = False
        for index, line in enumerate(lines):
            from_match = from_pattern.match(line)
            if from_match is not None:
                module = from_match.group(2)
                if module in local_modules:
                    lines[index] = (
                        f"{from_match.group(1)}from .{module} import {from_match.group(3)}"
                    )
                    changed = True
                    continue
            import_match = import_pattern.match(line)
            if import_match is not None:
                module = import_match.group(2)
                if module in local_modules:
                    lines[index] = f"{import_match.group(1)}from . import {module}"
                    changed = True
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            rewritten.append(path.name)
    return rewritten
