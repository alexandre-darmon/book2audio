#!/bin/zsh
# Double-click this file to open book2audio in your browser.
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if ! command -v uv >/dev/null; then
  echo "First launch: installing uv (it manages Python and the libraries)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v ffmpeg >/dev/null; then
  echo "Note: ffmpeg is missing, so no audiobook file will be built. Install it with: brew install ffmpeg"
fi

# caffeinate keeps the Mac awake while a book is being read
exec caffeinate -i uv run book2audio.py --ui
