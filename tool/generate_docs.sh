#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
dart doc --output docs/api
echo "Generated docs at docs/api/index.html"
