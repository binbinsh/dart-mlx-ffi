from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from matrix_config import artifact_coverage, blocked_platforms


def main() -> None:
    parser = argparse.ArgumentParser(description="List runtime benchmark models")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("models.yaml"),
    )
    parser.add_argument("--support-level", default=None)
    parser.add_argument("--artifact-coverage", default=None)
    parser.add_argument("--blocked-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models = config.get("first_wave") or []
    if args.support_level:
        models = [
            model
            for model in models
            if model.get("support_level") == args.support_level
        ]
    if args.artifact_coverage:
        models = [
            model
            for model in models
            if artifact_coverage(model) == args.artifact_coverage
        ]
    if args.blocked_only:
        models = [model for model in models if blocked_platforms(model)]
    print(json.dumps(models, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
