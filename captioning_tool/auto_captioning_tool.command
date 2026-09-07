#!/bin/bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
UV="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV" && -x "$HOME/.local/bin/uv" ]]; then UV="$HOME/.local/bin/uv"; fi
if [[ -z "$UV" ]]; then
  echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ and run again.' >&2
  exit 1
fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This launcher requires an Apple Silicon Mac.' >&2
  exit 1
fi
RUNTIME="$ROOT/runtime/mac"
if [[ ! -x "$RUNTIME/bin/python" ]]; then "$UV" venv --python 3.13 "$RUNTIME"; fi
"$UV" pip sync --python "$RUNTIME/bin/python" "$ROOT/requirements-mac.lock"
export PYTHONNOUSERSITE=1
if [[ $# -eq 0 ]]; then
  read -r -p 'Input image folder: ' INPUT_DIR
  exec "$RUNTIME/bin/python" -u "$ROOT/auto_captioning_tool.py" \
    --profile compact --device mps --stage captions --no-training-config \
    --prompt "$ROOT/prompts/neutral/general.txt" --input "$INPUT_DIR"
fi
exec "$RUNTIME/bin/python" -u "$ROOT/auto_captioning_tool.py" \
  --profile compact --device mps --stage captions --no-training-config \
  --prompt "$ROOT/prompts/neutral/general.txt" "$@"
