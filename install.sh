#!/bin/sh
# book2audio installer. One command:
#   curl -LsSf https://raw.githubusercontent.com/alexandre-darmon/book2audio/main/install.sh | sh
# Running it again updates book2audio to the latest version.
set -eu

SOURCE="${BOOK2AUDIO_SOURCE:-git+https://github.com/alexandre-darmon/book2audio}"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "book2audio needs a Mac with Apple Silicon (M1 or later)."
  [ -n "${BOOK2AUDIO_FORCE:-}" ] || exit 1
fi

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv, which manages Python and the libraries..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Installing book2audio..."
uv tool install --reinstall --python 3.12 "$SOURCE"
uv tool update-shell >/dev/null 2>&1 || true

echo
echo "Done. Open a new Terminal window and type:  book2audio"
echo "A page opens in your browser. The voice model downloads the first time you use it."
