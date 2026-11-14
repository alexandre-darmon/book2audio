#!/bin/sh
# book2audio installer. One command:
#   curl -LsSf https://raw.githubusercontent.com/alexandre-darmon/book2audio/main/install.sh | sh
# Running it again updates book2audio to the latest version.
set -eu

SOURCE="${BOOK2AUDIO_SOURCE:-git+https://github.com/alexandre-darmon/book2audio}"
RAW="https://raw.githubusercontent.com/alexandre-darmon/book2audio/main"

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

# A small app in ~/Applications, so book2audio can be started with a double-click.
# It is created here, on this Mac, and only starts the command installed above.
make_app() {
  APP="$HOME/Applications/book2audio.app"
  rm -rf "$APP"
  mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
  cat > "$APP/Contents/MacOS/book2audio" <<'LAUNCH'
#!/bin/sh
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
mkdir -p "$HOME/Library/Logs"
exec book2audio --ui >> "$HOME/Library/Logs/book2audio.log" 2>&1
LAUNCH
  chmod +x "$APP/Contents/MacOS/book2audio"
  cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>book2audio</string>
  <key>CFBundleDisplayName</key><string>book2audio</string>
  <key>CFBundleIdentifier</key><string>net.alexdarmon.book2audio</string>
  <key>CFBundleExecutable</key><string>book2audio</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSUIElement</key><true/>
</dict></plist>
PLIST
  # Icon: optional. Any failure here leaves a working app with a plain icon.
  SET="$(mktemp -d)/icon.iconset"
  mkdir -p "$SET"
  if curl -LsSf "$RAW/assets/book2audio-icon-1024.png" -o "$SET/src.png" 2>/dev/null; then
    for s in 16 32 128 256 512; do
      sips -z "$s" "$s" "$SET/src.png" --out "$SET/icon_${s}x${s}.png" >/dev/null 2>&1 || true
      d=$((s * 2))
      sips -z "$d" "$d" "$SET/src.png" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null 2>&1 || true
    done
    rm -f "$SET/src.png"
    iconutil -c icns "$SET" -o "$APP/Contents/Resources/icon.icns" >/dev/null 2>&1 || true
  fi
  make_notifier || true
}

# A tiny helper app that shows the "finished" notification, so that it carries the book2audio icon
# (a notification sent from the Terminal shows a generic script icon). Optional: without it, the
# notification still appears, with the generic icon.
make_notifier() {
  DIR="$HOME/Library/Application Support/book2audio"
  NOTIFIER="$DIR/book2audio.app"
  [ -f "$APP/Contents/Resources/icon.icns" ] || return 0
  command -v osacompile >/dev/null 2>&1 || return 0
  mkdir -p "$DIR"
  rm -rf "$NOTIFIER"
  osacompile -o "$NOTIFIER" \
    -e 'set f to (POSIX path of (path to home folder)) & "Library/Application Support/book2audio/notification.txt"' \
    -e 'set t to paragraphs of (read (POSIX file f) as «class utf8»)' \
    -e 'display notification (item 2 of t) with title (item 1 of t)' \
    -e 'delay 2' >/dev/null 2>&1 || { rm -rf "$NOTIFIER"; return 0; }
  cp "$APP/Contents/Resources/icon.icns" "$NOTIFIER/Contents/Resources/applet.icns"
  rm -f "$NOTIFIER/Contents/Resources/Assets.car"
  PL="$NOTIFIER/Contents/Info.plist"
  /usr/libexec/PlistBuddy -c "Delete :CFBundleIconName" "$PL" >/dev/null 2>&1 || true
  /usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$PL" >/dev/null 2>&1 || true
  /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier net.alexdarmon.book2audio.notifier" "$PL" >/dev/null 2>&1 \
    || /usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string net.alexdarmon.book2audio.notifier" "$PL" >/dev/null 2>&1 || true
  codesign --force -s - "$NOTIFIER" >/dev/null 2>&1 || true
}

APP_NOTE=""
if [ "$(uname -s)" = "Darwin" ] && [ -z "${BOOK2AUDIO_NO_APP:-}" ]; then
  if make_app; then
    APP_NOTE="Or double-click book2audio in your Applications folder (the one inside your home folder)."
  fi
fi

echo
echo "Done. The voice model downloads the first time you use it."

BIN="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
[ -z "${BOOK2AUDIO_NO_START:-}" ] || exit 0

# Start it without a Terminal window: the app in Applications runs in the background and opens the page.
APP="$HOME/Applications/book2audio.app"
if [ -d "$APP" ] && command -v open >/dev/null 2>&1 && open "$APP" 2>/dev/null; then
  n=0
  while [ "$n" -lt 30 ]; do
    if curl -sf -o /dev/null http://127.0.0.1:8765/status 2>/dev/null; then
      echo "book2audio is open in your browser. You can close this window."
      echo "Next time, double-click book2audio in the Applications folder inside your home folder."
      exit 0
    fi
    n=$((n + 1)); sleep 1
  done
  echo "The app did not answer. Starting book2audio in this window instead."
fi

# No app (BOOK2AUDIO_NO_APP, or not a Mac), or it did not start: run it here, with its full path.
echo "Next time, open a new Terminal window and type:  book2audio"
if [ -x "$BIN/book2audio" ]; then
  echo
  exec "$BIN/book2audio"
fi
