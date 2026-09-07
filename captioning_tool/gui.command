#!/bin/bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$ROOT/auto_captioning_tool.command" --gui "$@"
