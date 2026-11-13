#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "mlx-audio; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "misaki[en]; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mistral-common[audio]; sys_platform == 'darwin' and platform_machine == 'arm64'",   # needed by Voxtral
#   "beautifulsoup4",
#   "imageio-ffmpeg",          # brings its own ffmpeg, so nothing else has to be installed
#   "soundfile",
#   "numpy",
# ]
# ///
"""book2audio: turn an ebook (EPUB, or AZW3/MOBI via Calibre) into one audio
file per chapter, using a local text-to-speech model on Apple Silicon.

    uv run book2audio.py book.epub --list        # see the chapters
    uv run book2audio.py book.epub --preview     # ~1 minute sample
    uv run book2audio.py book.epub               # whole book
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

DEFAULT_MODEL = "mlx-community/Kokoro-82M-bf16"

# Kokoro: language -> (lang_code, default voice)
KOKORO = {
    "en": ("a", "af_heart"),
    "en-gb": ("b", "bf_emma"),
    "fr": ("f", "ff_siwis"),
    "es": ("e", "ef_dora"),
    "it": ("i", "if_sara"),
    "pt": ("p", "pf_dora"),
    "hi": ("h", "hf_alpha"),
    "ja": ("j", "jf_alpha"),
    "zh": ("z", "zf_xiaobei"),
}
MODEL_ALIASES = {
    "kokoro": DEFAULT_MODEL,                                   # small and fast, best in English
    "voxtral": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",   # 8 GB, natural in French, non-commercial licence
    "voxtral-4bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",   # 2.5 GB, same voices, about 2.5x faster: default for French
}
VOXTRAL = {"fr": "fr_female", "en": "casual_female", "es": "es_female", "de": "de_female",
           "it": "it_female", "pt": "pt_female", "nl": "nl_female", "ar": "ar_male", "hi": "hi_female"}
# Other models (Qwen3-TTS...) take the language as a plain name.
LANG_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "de": "German", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean", "ru": "Russian", "nl": "Dutch", "hi": "Hindi", "ar": "Arabic",
}

ICLOUD = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs"

PAUSE_SENTENCE = 0.15   # seconds of silence between chunks of one paragraph
PAUSE_PARAGRAPH = 0.55  # between paragraphs
PAUSE_TITLE = 1.0       # after the chapter title
MAX_CHARS = 350         # max characters sent to the model at once


# --------------------------------------------------------------------------
# 1. Read the book
# --------------------------------------------------------------------------
@dataclass
class Chapter:
    title: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.paragraphs)


@dataclass
class Book:
    title: str
    author: str
    language: str
    chapters: list[Chapter]


BLOCKS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote",
          "section", "article", "tr", "dt", "dd", "figcaption", "pre"]


def _clean(text: str) -> str:
    text = text.replace("­", "")              # soft hyphens
    text = re.sub(r"[​‌‍﻿]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _html_to_paragraphs(html: bytes) -> tuple[str | None, list[str]]:
    """Return (first heading, paragraphs) of one XHTML document."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup
    # Things nobody wants read aloud: code, footnote markers, menus, tables.
    for tag in body.find_all(["script", "style", "sup", "nav", "table", "svg", "head"]):
        tag.decompose()
    heading = body.find(["h1", "h2", "h3"])
    heading_text = _clean(heading.get_text(" ")) if heading else None
    for br in body.find_all("br"):
        br.replace_with("\n")
    for tag in body.find_all(BLOCKS):
        tag.insert_before("\n\n")
        tag.append("\n\n")
    paragraphs = [_clean(p) for p in re.split(r"\n\s*\n", body.get_text())]
    return heading_text or None, [p for p in paragraphs if p]


def _toc_labels(z: zipfile.ZipFile, manifest: list[dict], base: str) -> dict[str, str]:
    """Map 'path of a file inside the epub' -> 'chapter title from the table of contents'."""
    from bs4 import BeautifulSoup

    labels: dict[str, str] = {}

    def add(toc_path: str, href: str, label: str) -> None:
        href = unquote(href.split("#")[0])
        if not href or not label:
            return
        path = posixpath.normpath(posixpath.join(posixpath.dirname(toc_path), href))
        labels.setdefault(path, _clean(label))

    for item in manifest:  # EPUB 3 navigation document
        if "nav" in (item.get("properties") or "").split():
            path = posixpath.normpath(posixpath.join(base, unquote(item["href"])))
            try:
                soup = BeautifulSoup(z.read(path), "html.parser")
            except KeyError:
                continue
            navs = soup.find_all("nav")
            nav = next((n for n in navs if "toc" in (n.get("epub:type") or "")), navs[0] if navs else None)
            for a in (nav.find_all("a") if nav else []):
                add(path, a.get("href") or "", a.get_text(" "))
    if labels:
        return labels
    for item in manifest:  # EPUB 2 NCX
        if item.get("media-type") == "application/x-dtbncx+xml":
            path = posixpath.normpath(posixpath.join(base, unquote(item["href"])))
            try:
                ncx = ET.fromstring(z.read(path))
            except (KeyError, ET.ParseError):
                continue
            for point in ncx.findall(".//{*}navPoint"):
                text = point.find("{*}navLabel/{*}text")
                content = point.find("{*}content")
                if text is not None and content is not None:
                    add(path, content.get("src") or "", text.text or "")
    return labels


def read_epub(path: Path, min_chars: int = 500) -> Book:
    with zipfile.ZipFile(path) as z:
        if "META-INF/encryption.xml" in z.namelist() and b"EncryptedData" in z.read("META-INF/encryption.xml"):
            print("Warning: this EPUB contains encrypted parts (DRM?). Text may be unreadable.", file=sys.stderr)
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = container.find(".//{*}rootfile").get("full-path")
        opf = ET.fromstring(z.read(opf_path))
        base = posixpath.dirname(opf_path)

        def meta(name: str, default: str) -> str:
            el = opf.find(f".//{{*}}metadata/{{*}}{name}")
            return _clean(el.text) if el is not None and el.text else default

        manifest = [dict(i.attrib) for i in opf.findall(".//{*}manifest/{*}item")]
        by_id = {i["id"]: i for i in manifest if "id" in i}
        labels = _toc_labels(z, manifest, base)

        chapters: list[Chapter] = []
        for ref in opf.findall(".//{*}spine/{*}itemref"):
            item = by_id.get(ref.get("idref"))
            if not item or "html" not in item.get("media-type", ""):
                continue
            if "nav" in (item.get("properties") or "").split():
                continue
            doc = posixpath.normpath(posixpath.join(base, unquote(item["href"])))
            try:
                heading, paragraphs = _html_to_paragraphs(z.read(doc))
            except KeyError:
                continue
            if not paragraphs:
                continue
            label = labels.get(doc)
            # A file missing from the table of contents is the continuation
            # of the previous chapter (publishers often split long chapters).
            if labels and not label and chapters:
                chapters[-1].paragraphs += paragraphs
                continue
            title = label or heading or f"Section {len(chapters) + 1}"
            if _clean(paragraphs[0]).lower() == title.lower():
                paragraphs = paragraphs[1:]   # the title is announced separately
            chapters.append(Chapter(title, paragraphs))

    chapters = [c for c in chapters if c.chars >= min_chars]
    return Book(meta("title", path.stem), meta("creator", ""), meta("language", "en").lower(), chapters)


def to_epub(path: Path, workdir: Path) -> Path:
    """EPUB is used as is. Kindle formats are converted with Calibre."""
    if path.suffix.lower() == ".epub":
        return path
    tool = shutil.which("ebook-convert") or "/Applications/calibre.app/Contents/MacOS/ebook-convert"
    if not Path(tool).exists():
        sys.exit(f"{path.suffix} files need Calibre to be converted first.\n"
                 "Install it with:  brew install --cask calibre")
    out = workdir / (path.stem + ".epub")
    print(f"Converting {path.name} to EPUB with Calibre...")
    result = subprocess.run([tool, str(path), str(out)], capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        hint = " The file is DRM-protected: Calibre cannot open it." if "DRM" in result.stdout + result.stderr else ""
        sys.exit(f"Calibre could not convert this file.{hint}\n{result.stderr[-500:]}")
    return out


# --------------------------------------------------------------------------
# 2. Cut the text into pieces the model reads well
# --------------------------------------------------------------------------
def split_paragraph(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Cut on sentence ends first, then commas, so the voice never stops mid-phrase."""
    sentences = re.findall(r".+?(?:[.!?…]+[\"'”’»)\]]*(?=\s|$)|$)", text, flags=re.S) or [text]
    pieces: list[str] = []
    for s in (s.strip() for s in sentences):
        if not s:
            continue
        if len(s) <= max_chars:
            pieces.append(s)
            continue
        part = ""
        for clause in re.split(r"(?<=[,;:—–])\s+", s):
            while len(clause) > max_chars:            # no punctuation at all: cut on a space
                cut = clause.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                if part:
                    pieces.append(part)
                    part = ""
                pieces.append(clause[:cut].strip())
                clause = clause[cut:].strip()
            if part and len(part) + 1 + len(clause) > max_chars:
                pieces.append(part)
                part = clause
            else:
                part = f"{part} {clause}".strip()
        if part:
            pieces.append(part)
    chunks: list[str] = []                           # pack short sentences together
    for p in pieces:
        if chunks and len(chunks[-1]) + 1 + len(p) <= max_chars:
            chunks[-1] += " " + p
        else:
            chunks.append(p)
    return chunks


# --------------------------------------------------------------------------
# 3. Speak
# --------------------------------------------------------------------------
class Stopped(Exception):
    pass


class Report:
    """Where messages go. The command line prints them, the web page shows them."""

    def log(self, message: str) -> None:
        print(message)

    def progress(self, i: int, n: int, minutes: float, tracker: "Tracker | None" = None) -> None:
        extra = f" | {tracker.percent}% of the job, {tracker.left_text()}" if tracker else ""
        print(f"\r   {i}/{n} passages, {minutes:.1f} min of audio{extra}   ", end="\n" if i == n else "", flush=True)


    def plan(self, chapters: list[dict]) -> None:
        """Called once per book with its chapters: {n, title, chars, done}."""

    def chapter(self, n: int, finished: bool = False) -> None:
        """Called when a chapter starts, and again when it is finished."""

    def compare(self, results: list[dict], running: str) -> None:
        """Called as a model comparison moves on."""


def fmt_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    if minutes < 1:
        return "1 min"
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60:02d} min"


class Tracker:
    """Progress of a whole job, counted in characters of text: a few additions per passage, no cost."""

    def __init__(self, total_chars: int):
        self.total = max(total_chars, 1)
        self.done = 0          # characters finished, including those found on disk
        self.read = 0          # characters read aloud in this run
        self.t0 = 0.0

    def start_clock(self) -> None:
        self.t0 = self.t0 or time.time()

    def skip(self, chars: int) -> None:
        self.done += chars

    def add(self, chars: int) -> None:
        self.done += chars
        self.read += chars

    @property
    def percent(self) -> int:
        return min(100, int(100 * self.done / self.total))

    def seconds_left(self) -> float | None:
        if not self.t0 or self.read < 400:       # too early to say
            return None
        return max(0.0, (self.total - self.done) * (time.time() - self.t0) / self.read)

    def left_text(self) -> str:
        left = self.seconds_left()
        if self.done >= self.total:
            return "finishing"
        if left is None:
            return "estimating the time left..."
        return f"about {fmt_duration(left)} left (done around {time.strftime('%H:%M', time.localtime(time.time() + left))})"


REPORT = Report()


def load_local_first(model_id: str, load_model):
    """Models are kept on disk after the first download (~/.cache/huggingface).
    When a copy is there, load it without contacting Hugging Face at all."""
    root = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub")
    snapshots = root / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not (Path(model_id).exists() or any(snapshots.glob("*/*"))):
        REPORT.log("First use of this model: downloading it once, it is then kept on this Mac.")
        return load_model(model_id)
    try:
        import huggingface_hub.constants as hub
    except ImportError:
        return load_model(model_id)
    previous = hub.HF_HUB_OFFLINE
    hub.HF_HUB_OFFLINE = True
    try:
        return load_model(model_id)
    except Exception:
        hub.HF_HUB_OFFLINE = previous        # local copy incomplete: let it download what is missing
        return load_model(model_id)
    finally:
        hub.HF_HUB_OFFLINE = previous


class Synth:
    """Thin wrapper around mlx-audio. The model is loaded once and reused."""

    def __init__(self, model_id: str, language: str, voice: str | None, speed: float, lang_code: str | None):
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError:
            sys.exit("The voice models need a Mac with Apple Silicon (M1 or later): mlx-audio is not available here.")
        short = language.split("-")[0]
        model_id = MODEL_ALIASES.get(model_id.lower(), model_id)
        self.kwargs: dict = {}
        if "kokoro" in model_id.lower():
            code, default_voice = KOKORO.get(language) or KOKORO.get(short) or KOKORO["en"]
            if short not in KOKORO:
                REPORT.log(f"Warning: Kokoro has no '{language}' voice, reading with an English voice.")
            self.kwargs = {"voice": voice or default_voice, "lang_code": lang_code or code, "speed": speed}
        elif "voxtral" in model_id.lower():
            # Voxtral picks the language from the voice: fr_female, fr_male, casual_female...
            default_voice = VOXTRAL.get(short, "casual_female")
            self.kwargs = {"voice": voice or default_voice}
            if lang_code:
                self.kwargs["lang_code"] = lang_code
        else:
            self.kwargs = {"lang_code": lang_code or LANG_NAMES.get(short, "English")}
            if voice:
                self.kwargs["voice"] = voice
            if speed != 1.0:
                self.kwargs["speed"] = speed
        REPORT.log(f"Loading {model_id} ({', '.join(f'{k}={v}' for k, v in self.kwargs.items())})")
        self.key = f"{model_id}|{sorted(self.kwargs.items())}"
        self.model = load_local_first(model_id, load_model)
        self.sample_rate = int(getattr(self.model, "sample_rate", 0) or 24000)

    def say(self, text: str):
        import numpy as np

        parts = []
        for result in self.model.generate(text=text, **self.kwargs):
            parts.append(np.array(result.audio, dtype=np.float32).reshape(-1))
            self.sample_rate = int(getattr(result, "sample_rate", 0) or self.sample_rate)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def build_script(chapter: Chapter, read_title: bool, max_chars: int = MAX_CHARS) -> list[tuple[str, float]]:
    """The chapter as a list of (passage to read, silence after it)."""
    script: list[tuple[str, float]] = []
    if read_title:
        script.append((chapter.title.rstrip(".") + ".", PAUSE_TITLE))
    for paragraph in chapter.paragraphs:
        chunks = split_paragraph(paragraph, max_chars)
        for i, chunk in enumerate(chunks):
            if re.search(r"\w", chunk):              # skip "***" style separators
                script.append((chunk, PAUSE_PARAGRAPH if i == len(chunks) - 1 else PAUSE_SENTENCE))
    return script


def script_chars(script: list[tuple[str, float]]) -> int:
    return sum(len(text) for text, _ in script)


def render_chapter(synth, script: list[tuple[str, float]], pcm: Path, tracker: Tracker | None = None, resume: bool = True):
    """Read one chapter into a raw audio file, passage by passage.

    After every passage the position is saved next to the audio, so an
    interrupted chapter continues where it stopped instead of starting over.
    Returns (chapter seconds, seconds read in this run, sample rate)."""
    import numpy as np

    state_file = pcm.with_suffix(".json")
    key = f"{getattr(synth, 'key', '')}|{len(script)}|{script_chars(script)}"
    start, size, rate = 0, 0, 0
    if resume and pcm.exists() and state_file.exists():
        try:
            saved = json.loads(state_file.read_text())
            if saved["key"] == key and pcm.stat().st_size >= saved["bytes"]:   # same text, same voice
                start, size, rate = saved["done"], saved["bytes"], saved["rate"]
        except (ValueError, KeyError, OSError):
            pass
    if start:
        REPORT.log(f"   continuing at passage {start + 1} of {len(script)}")
        if tracker:
            tracker.skip(script_chars(script[:start]))
    new_samples = 0
    with open(pcm, "r+b" if start else "wb") as out:
        out.truncate(size)                           # drop a passage that was cut mid-write
        out.seek(size)
        for i, (text, pause) in enumerate(script[start:], start + 1):
            if tracker:
                tracker.start_clock()
            audio = synth.say(text)
            rate = synth.sample_rate
            data = np.concatenate([np.clip(audio, -1.0, 1.0), np.zeros(int(pause * rate), dtype=np.float32)])
            out.write((data * 32767).astype("<i2").tobytes())
            out.flush()
            size += 2 * len(data)
            new_samples += len(data)
            tmp = state_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"key": key, "done": i, "bytes": size, "rate": rate}))
            tmp.replace(state_file)
            if tracker:
                tracker.add(len(text))
            REPORT.progress(i, len(script), size / 2 / rate / 60, tracker)
    if not rate:
        return 0.0, 0.0, 0
    return size / 2 / rate, new_samples / rate, rate


# --------------------------------------------------------------------------
# 4. Encode and package
# --------------------------------------------------------------------------
_ffmpeg: list = []


def ffmpeg_exe() -> str | None:
    """The ffmpeg already on this Mac if there is one, otherwise the copy that comes with the app."""
    if not _ffmpeg:
        found = shutil.which("ffmpeg")
        if not found:
            try:
                import imageio_ffmpeg
                found = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                found = None
        _ffmpeg.append(found)
    return _ffmpeg[0]


def audio_seconds(path: Path) -> float:
    """Length of an audio file, read from ffmpeg's own description of it."""
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    found = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not found:
        raise RuntimeError(f"Could not read the length of {path.name}")
    h, m, sec = found.groups()
    return int(h) * 3600 + int(m) * 60 + float(sec)


def encode(pcm: Path, rate: int, target: Path, tags: dict[str, str]) -> None:
    """Raw audio -> final file. The raw file and its saved position are removed once the file is complete."""
    tmp = target.with_name(".tmp-" + target.name)
    if target.suffix == ".wav":
        import numpy as np
        import soundfile as sf

        sf.write(str(tmp), np.fromfile(pcm, dtype="<i2"), rate, subtype="PCM_16", format="WAV")
    else:
        codec = ["-c:a", "aac", "-b:a", "64k"] if target.suffix == ".m4a" else ["-c:a", "libmp3lame", "-q:a", "5"]
        meta = [x for k, v in tags.items() if v for x in ("-metadata", f"{k}={v}")]
        fmt = "mp4" if target.suffix == ".m4a" else "mp3"
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", str(pcm),
                        *codec, *meta, "-f", fmt, str(tmp)], check=True)
    tmp.replace(target)                              # a half-written file is never left behind
    pcm.unlink()
    pcm.with_suffix(".json").unlink(missing_ok=True)


@contextmanager
def keep_awake():
    """Stop the Mac from going to sleep while a book is being read (macOS caffeinate)."""
    proc = None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        proc = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    try:
        yield
    finally:
        if proc:
            proc.terminate()


def build_m4b(files: list[tuple[str, Path]], target: Path, book: Book) -> None:
    """Join the chapter files into one audiobook file with chapter markers."""
    def esc(s: str) -> str:
        return re.sub(r"([=;#\\\n])", r"\\\1", s)

    meta = [";FFMETADATA1", f"title={esc(book.title)}", f"artist={esc(book.author)}", f"album={esc(book.title)}", "genre=Audiobook"]
    start = 0
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "list.txt"
        lines = []
        for title, f in files:
            end = start + int(audio_seconds(f) * 1000)
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={esc(title)}"]
            start = end
            quoted = str(f.resolve()).replace("'", "'\\''")
            lines.append(f"file '{quoted}'")
        listing.write_text("\n".join(lines), encoding="utf-8")
        meta_file = Path(tmp) / "meta.txt"
        meta_file.write_text("\n".join(meta), encoding="utf-8")
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-i", str(meta_file), "-map_metadata", "1", "-map", "0:a", "-c", "copy", "-f", "mp4", str(target)], check=True)


# --------------------------------------------------------------------------
# 5. Command line
# --------------------------------------------------------------------------
def parse_selection(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(1, total + 1))
    chosen: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        lo, hi = int(a), int(b or a)
        chosen.update(range(lo, hi + 1))
    return sorted(n for n in chosen if 1 <= n <= total)


def safe_name(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", text)).strip()[:80] or "untitled"


def pick_format(wanted: str) -> str:
    if wanted != "wav" and not ffmpeg_exe():
        REPORT.log("ffmpeg not found: writing WAV files, no audiobook file (install it with: brew install ffmpeg).")
        return "wav"
    return wanted


# Pocket TTS (Kyutai) runs on the CPU and needs PyTorch. To keep its libraries from
# ever clashing with the main ones, it runs in its own small process with its own
# environment, created by uv on first use. Text goes in, raw audio comes back.
POCKET = {"fr": ("french_24l", "estelle"), "en": ("english", "alba"), "de": ("german_24l", "juergen"),
          "es": ("spanish_24l", "lola"), "it": ("italian_24l", "giovanni"), "pt": ("portuguese_24l", "rafael")}
POCKET_WORKER = r"""
import json, os, sys
proto = os.fdopen(os.dup(1), "wb")      # keep the real output for audio...
os.dup2(2, 1)                           # ...and send anything the libraries print to the log
from pocket_tts import TTSModel
model = TTSModel.load_model(language=sys.argv[1])
voice = model.get_state_for_audio_prompt(sys.argv[2])
proto.write((json.dumps({"rate": int(model.sample_rate)}) + "\n").encode()); proto.flush()
for line in sys.stdin:
    audio = model.generate_audio(voice, json.loads(line)["text"]).detach().cpu().numpy().astype("<f4").reshape(-1)
    proto.write((json.dumps({"samples": int(audio.size)}) + "\n").encode()); proto.write(audio.tobytes()); proto.flush()
"""


class PocketSynth:
    def __init__(self, language: str, voice: str | None):
        uv = shutil.which("uv")
        if not uv:
            sys.exit("Pocket TTS needs uv to install itself (https://docs.astral.sh/uv/).")
        short = language.split("-")[0]
        name, default_voice = POCKET.get(short) or POCKET["en"]
        voice = voice or default_voice
        self.key = f"pocket|{name}|{voice}"
        self._dir = tempfile.mkdtemp(prefix="book2audio-pocket-")
        script = Path(self._dir) / "worker.py"
        script.write_text(POCKET_WORKER)
        REPORT.log(f"Loading Pocket TTS ({name}, voice={voice}). The first use installs it and downloads the model.")
        self.proc = subprocess.Popen([uv, "run", "--no-project", "--python", "3.12", "--with", "pocket-tts",
                                      "python", "-u", str(script), name, voice], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.sample_rate = int(self._header()["rate"])

    def _header(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Pocket TTS stopped. The reason is in the Terminal window, just above.")
        return json.loads(line)

    def say(self, text: str):
        import numpy as np

        self.proc.stdin.write((json.dumps({"text": text}) + "\n").encode())
        self.proc.stdin.flush()
        size = 4 * int(self._header()["samples"])
        data = b""
        while len(data) < size:
            part = self.proc.stdout.read(size - len(data))
            if not part:
                raise RuntimeError("Pocket TTS stopped in the middle of a passage.")
            data += part
        return np.frombuffer(data, dtype="<f4").copy()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        shutil.rmtree(self._dir, ignore_errors=True)


_synth_cache: dict = {}


def default_model(language: str) -> str:
    # French: the 4-bit Voxtral measured 2.7x real time on a MacBook Pro M6, against 1.05x for the
    # full model, with a voice judged just as good by ear.
    return "voxtral-4bit" if language.startswith("fr") else "kokoro"


def make_synth(args, language: str):
    """The model stays loaded between two jobs that use the same voice. Only one is kept in memory."""
    model = args.model or default_model(language)
    key = (model, language, args.voice, args.speed, args.lang_code)
    if key not in _synth_cache:
        for old in _synth_cache.values():
            if hasattr(old, "close"):
                old.close()
        _synth_cache.clear()
        import gc
        gc.collect()                                 # give the previous model's memory back first
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        _synth_cache[key] = (PocketSynth(language, args.voice) if model.lower() == "pocket"
                             else Synth(model, language, args.voice, args.speed, args.lang_code))
    return _synth_cache[key]


# How fast each model reads, in "times real time". The starting values were measured on a
# MacBook Pro M6; each real run on this machine replaces them, so estimates become its own.
DEFAULT_SPEEDS = {"voxtral": 1.05, "voxtral-4bit": 2.7, "pocket": 5.4}
SPEEDS_FILE = Path.home() / "Audiobooks" / ".speeds.json"


def load_speeds() -> dict:
    try:
        return {**DEFAULT_SPEEDS, **json.loads(SPEEDS_FILE.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULT_SPEEDS)


def save_speed(model: str, speed: float) -> None:
    try:
        known = json.loads(SPEEDS_FILE.read_text())
    except (OSError, ValueError):
        known = {}
    known[model] = round(speed, 2)
    try:
        SPEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SPEEDS_FILE.write_text(json.dumps(known))
    except OSError:
        pass


FRONT_MATTER = re.compile(r"^(copyright|pr[ée]sentation|du m[êe]me auteur|table( des mati[èe]res)?|sommaire|d[ée]dicace|"
                          r"remerciements|mentions l[ée]gales|couverture|identit[ée]|title page|cover|contents|table of contents|"
                          r"also by.*|about the author|dedication|acknowledg(e)?ments)$", re.I)


def is_front_matter(chapter: Chapter) -> bool:
    """Short pages nobody wants read aloud: copyright, table of contents, other titles..."""
    return chapter.chars < 4500 and bool(FRONT_MATTER.match(chapter.title.strip()))


def default_out(book: Book) -> Path:
    return Path.home() / "Audiobooks" / safe_name(book.title)


def chapter_target(out_dir: Path, n: int, chapter: Chapter, fmt: str) -> Path:
    return out_dir / f"{n:02d} - {safe_name(chapter.title)}.{fmt}"


def run_preview(book: Book, chapter_number: int, language: str, args, out_dir: Path) -> Path:
    """About one minute of one chapter, to judge the voice."""
    fmt = pick_format(args.format)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = book.chapters[chapter_number - 1]
    sample, size = Chapter(first.title), 0
    for p in first.paragraphs:
        sample.paragraphs.append(p)
        size += len(p)
        if size > 900:
            break
    target = out_dir / f"preview.{fmt}"
    pcm = out_dir / ".tmp-preview.pcm"
    script = build_script(sample, not args.no_titles)
    synth = make_synth(args, language)
    _, _, rate = render_chapter(synth, script, pcm, Tracker(script_chars(script)), resume=False)
    encode(pcm, rate, target, {"title": "Preview"})
    REPORT.log(f"Preview written to {target}")
    return target


MODEL_LABELS = {"voxtral": "Voxtral", "voxtral-4bit": "Voxtral 4-bit", "kokoro": "Kokoro", "pocket": "Pocket TTS"}


def compare_list(language: str) -> list[str]:
    return ["voxtral", "voxtral-4bit", "pocket"] if language.startswith("fr") else ["kokoro", "voxtral", "pocket"]


def run_compare(book: Book, chapter_number: int, language: str, args, out_dir: Path, models: list[str] | None = None) -> list[dict]:
    """Read the same one-minute passage with several models, and time each one on this machine."""
    fmt = pick_format(args.format)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = book.chapters[chapter_number - 1]
    sample, size = Chapter(first.title), 0
    for p in first.paragraphs:
        sample.paragraphs.append(p)
        size += len(p)
        if size > 900:
            break
    script = build_script(sample, not args.no_titles)
    results: list[dict] = []
    for model in models or compare_list(language):
        row = {"model": model, "label": MODEL_LABELS.get(model, model), "file": "", "error": ""}
        results.append(row)
        REPORT.compare(results, model)
        REPORT.log(f"=== {row['label']} ===")
        try:
            opts = argparse.Namespace(**{**vars(args), "model": model, "voice": None})
            lang = "en" if model == "kokoro" and language.startswith("en") else language
            started = time.time()
            synth = make_synth(opts, lang)
            row["load_s"] = round(time.time() - started, 1)
            pcm = out_dir / f".tmp-compare-{model}.pcm"
            started = time.time()
            audio_s, _, rate = render_chapter(synth, script, pcm, Tracker(script_chars(script)), resume=False)
            row["compute_s"] = round(time.time() - started, 1)
            row["audio_s"] = round(audio_s, 1)
            row["speed"] = round(audio_s / max(row["compute_s"], 0.1), 2)
            save_speed(model, row["speed"])
            row["chars_s"] = round(script_chars(script) / max(row["compute_s"], 0.1), 1)
            target = out_dir / f"compare-{model}.{fmt}"
            encode(pcm, rate, target, {"title": f"Compare {row['label']}"})
            row["file"] = target.name
            REPORT.log(f"{row['label']}: {row['audio_s']} s of audio in {row['compute_s']} s = {row['speed']}x real time "
                       f"({row['chars_s']} characters per second), model loaded in {row['load_s']} s")
        except Stopped:
            raise
        except KeyboardInterrupt:
            raise
        except BaseException as e:                   # one model failing must not hide the others
            row["error"] = (str(e) if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}")[:300]
            REPORT.log(f"{row['label']} failed: {row['error']}")
        REPORT.compare(results, "")
    REPORT.log(f"Listen to the compare-*.{fmt} files in {out_dir}")
    return results


def run_book(book: Book, selection: list[int], language: str, args, out_dir: Path) -> None:
    fmt = pick_format(args.format)
    out_dir.mkdir(parents=True, exist_ok=True)
    read_titles = not args.no_titles
    todo = [n for n in selection if not chapter_target(out_dir, n, book.chapters[n - 1], fmt).exists()]
    for n in selection:
        if n not in todo:
            REPORT.log(f"[{n}/{len(book.chapters)}] {book.chapters[n - 1].title}: already done, skipping")
    scripts = {n: build_script(book.chapters[n - 1], read_titles) for n in todo}
    REPORT.plan([{"n": n, "title": book.chapters[n - 1].title, "chars": book.chapters[n - 1].chars, "done": n not in todo}
                 for n in selection])
    tracker = Tracker(sum(script_chars(s) for s in scripts.values()))
    started = time.time()
    audio_seconds = 0.0
    with keep_awake():
        synth = make_synth(args, language) if todo else None
        for n in todo:
            chapter = book.chapters[n - 1]
            REPORT.log(f"[{n}/{len(book.chapters)}] {chapter.title}")
            REPORT.chapter(n)
            pcm = out_dir / f".tmp-{n:02d}.pcm"
            _, seconds, rate = render_chapter(synth, scripts[n], pcm, tracker)
            audio_seconds += seconds
            encode(pcm, rate, chapter_target(out_dir, n, chapter, fmt),
                   {"title": chapter.title, "album": book.title, "artist": book.author, "track": str(n), "genre": "Audiobook"})
            REPORT.chapter(n, finished=True)

    elapsed = time.time() - started
    if audio_seconds:
        REPORT.log(f"Done: {fmt_duration(audio_seconds)} of audio in {fmt_duration(elapsed)} ({audio_seconds / max(elapsed, 1):.1f}x real time).")
        if audio_seconds > 120:                      # long enough to be a fair measure of this machine
            save_speed(args.model or default_model(language), audio_seconds / max(elapsed, 1))
    # The audiobook holds every chapter finished so far, from this run and earlier ones.
    done = [(c.title, chapter_target(out_dir, n, c, fmt)) for n, c in enumerate(book.chapters, 1)
            if chapter_target(out_dir, n, c, fmt).exists()]
    if not args.no_m4b and fmt == "m4a" and done:
        target = out_dir / f"{safe_name(book.title)}.m4b"
        build_m4b(done, target, book)
        REPORT.log(f"Audiobook written to {target} ({len(done)} of {len(book.chapters)} chapters)")
        if args.iphone:
            if ICLOUD.exists():
                dest = ICLOUD / "Audiobooks"
                dest.mkdir(exist_ok=True)
                shutil.copy2(target, dest / target.name)
                REPORT.log(f"Copied to iCloud Drive/Audiobooks. On the iPhone: Files app > iCloud Drive > Audiobooks > {target.name}")
            else:
                REPORT.log("iCloud Drive is not enabled on this Mac: AirDrop the .m4b file to your iPhone instead.")
    elif args.iphone:
        REPORT.log("The iPhone copy needs the m4a format (with ffmpeg) to build the audiobook file.")
    REPORT.log(f"Files are in {out_dir}")


def open_book(path: Path, min_chars: int) -> Book:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            book = read_epub(to_epub(path, Path(tmp)), min_chars)
    except (zipfile.BadZipFile, KeyError, ET.ParseError, AttributeError):
        sys.exit(f"{path.name} is not a readable EPUB file.")
    if not book.chapters:
        sys.exit("No readable text found in this book (DRM-protected, or made of scanned images?).")
    return book


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Turn an ebook into one audio file per chapter, with a local voice model.")
    ap.add_argument("book", type=Path, nargs="?", help="EPUB file (AZW3 / MOBI / AZW work if Calibre is installed and the file has no DRM)")
    ap.add_argument("--ui", action="store_true", help="open the simple web page instead of using the command line")
    ap.add_argument("-o", "--out", type=Path, help="output folder (default: ~/Audiobooks/<book title>)")
    ap.add_argument("--list", action="store_true", help="show the chapters and stop")
    ap.add_argument("--chapters", help='which chapters to read, e.g. "1-3,7" (numbers from --list)')
    ap.add_argument("--preview", action="store_true", help="read only about one minute, to judge the voice")
    ap.add_argument("--compare", action="store_true", help="read the same minute with each model and time them on this machine")
    ap.add_argument("--lang", help="book language, e.g. en, en-gb, fr (default: read from the book)")
    ap.add_argument("--voice", help="voice name (default: a good voice for the language)")
    ap.add_argument("--speed", type=float, default=1.0, help="reading speed, 1.0 = normal")
    ap.add_argument("--model", help="kokoro, voxtral, voxtral-4bit, pocket, or any mlx-audio TTS model id (default: voxtral-4bit for French books, kokoro otherwise)")
    ap.add_argument("--lang-code", help="advanced: language value passed as is to the model")
    ap.add_argument("--format", choices=["m4a", "mp3", "wav"], default="m4a")
    ap.add_argument("--no-m4b", action="store_true", help="do not build the single audiobook file (.m4b with chapter markers)")
    ap.add_argument("--iphone", action="store_true", help="copy the audiobook to iCloud Drive/Audiobooks, to open it on the iPhone")
    ap.add_argument("--min-chars", type=int, default=500, help="skip sections shorter than this (cover, copyright...)")
    ap.add_argument("--no-titles", action="store_true", help="do not read chapter titles aloud")
    args = ap.parse_args(argv)

    if args.ui or args.book is None:
        return serve_ui(args)
    if not args.book.exists():
        sys.exit(f"File not found: {args.book}")
    book = open_book(args.book, args.min_chars)

    selection = parse_selection(args.chapters, len(book.chapters))
    language = (args.lang or book.language or "en").lower()
    total_chars = sum(book.chapters[n - 1].chars for n in selection)
    print(f"{book.title}" + (f", {book.author}" if book.author else "") + f"  [{language}]")
    for n, c in enumerate(book.chapters, 1):
        mark = " " if n in selection else "-"
        print(f" {mark}{n:3d}. {c.title[:70]:70s} ~{c.chars / 900:4.0f} min")   # ~900 characters per minute read aloud
    print(f"Selected: {len(selection)} chapters, about {total_chars / 900 / 60:.1f} hours of audio.")
    if args.list:
        return 0
    out_dir = args.out or default_out(book)
    if args.compare:
        run_compare(book, selection[0], language, args, out_dir, [args.model] if args.model else None)
    elif args.preview:
        run_preview(book, selection[0], language, args, out_dir)
    else:
        run_book(book, selection, language, args, out_dir)
    return 0


# --------------------------------------------------------------------------
# 6. The simple web page (uv run book2audio.py --ui)
# --------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>book2audio</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect x='.5' y='.5' width='63' height='63' rx='13' fill='%23111113' stroke='%2326262B'/%3E%3Cpath d='M13 30V34M22 26V38M42 26V38M51 30V34' fill='none' stroke='%23F7F7F7' stroke-width='6' stroke-linecap='square'/%3E%3Crect x='28' y='20' width='8' height='24' fill='%238C52EE'/%3E%3C/svg%3E">
<script>try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}</script>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark;
    --bg:#0a0a0c; --panel:#111113; --raise:#17171a; --line:#232327; --ink:#ededef; --soft:#8b8b93; --faint:#5c5c66;
    --accent:#8C52EE; --accent-ink:#b392f5; --tile:#111113; --tile-line:#26262B; --glyph:#F7F7F7; --danger:#f2777a; }
  :root[data-theme="light"] { color-scheme: light;
    --bg:#ffffff; --panel:#fafafa; --raise:#f4f4f5; --line:#E3E3E3; --ink:#0A0A0A; --soft:#62626b; --faint:#9a9aa3;
    --accent:#8C52EE; --accent-ink:#7a3fe0; --tile:#F7F7F7; --tile-line:#E3E3E3; --glyph:#0A0A0A; --danger:#c62f33; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 Inter, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
         -webkit-font-smoothing:antialiased; font-feature-settings:"cv11","ss01"; }
  .mono, .m, #log, .pos, #total { font-family:"Geist Mono", "SF Mono", ui-monospace, Menlo, monospace; font-variant-numeric:tabular-nums; }
  header { border-bottom:1px solid var(--line); }
  header > div { max-width:760px; margin:0 auto; padding:14px 24px; display:flex; align-items:center; gap:10px; }
  header svg { display:block; }
  .word { font-weight:600; letter-spacing:-0.02em; font-size:15px; }
  .tag { color:var(--soft); font-size:13px; margin-left:6px; }
  #theme { margin-left:auto; }
  #steps { list-style:none; counter-reset:s; display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; padding:0; margin:0 0 16px; }
  #steps li { counter-increment:s; border:1px solid var(--line); border-radius:8px; padding:14px 16px; color:var(--soft); font-size:13px; }
  #steps li::before { content:counter(s); display:block; font-family:"Geist Mono", "SF Mono", ui-monospace, Menlo, monospace; font-size:12px; color:var(--accent-ink); margin-bottom:6px; }
  #steps b { display:block; color:var(--ink); font-weight:600; }
  .tagk { color:var(--faint); font-size:11px; margin-left:8px; border:1px solid var(--line); border-radius:4px; padding:1px 5px; }
  @media (max-width:560px) { #steps { grid-template-columns:1fr; } .tag { display:none; } }
  main { max-width:760px; margin:0 auto; padding:32px 24px 96px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:20px; margin-bottom:16px; }
  .label { font-size:12px; color:var(--soft); letter-spacing:.02em; margin:0 0 12px; display:flex; align-items:center; gap:8px; }
  .label::before { content:""; width:3px; height:12px; background:var(--accent); }
  h2 { font-size:18px; line-height:1.3; font-weight:600; letter-spacing:-0.02em; margin:0; }
  .sub { color:var(--soft); font-size:13px; margin:2px 0 0; }
  #drop { border-style:dashed; text-align:center; padding:40px 20px; cursor:pointer; color:var(--soft); transition:border-color .15s, color .15s, background .15s; }
  #drop strong { color:var(--ink); font-weight:500; display:block; margin-bottom:2px; }
  #drop:hover, #drop.over { border-color:var(--accent); color:var(--ink); background:var(--raise); }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:end; margin-top:16px; }
  label { font-size:12px; color:var(--soft); display:block; }
  label.check { display:flex; align-items:center; gap:8px; font-size:13px; color:var(--ink); height:32px; }
  input[type=checkbox] { accent-color:var(--accent); width:14px; height:14px; margin:0; }
  select { font:inherit; font-size:13px; height:32px; padding:0 28px 0 10px; border:1px solid var(--line); border-radius:6px; background:var(--raise); color:var(--ink); margin-top:4px; }
  button { font:inherit; font-size:13px; font-weight:500; height:32px; padding:0 12px; border-radius:6px; border:1px solid var(--line); background:var(--raise); color:var(--ink); cursor:pointer; transition:background .15s, border-color .15s; }
  button:hover:not(:disabled) { border-color:var(--faint); }
  button.main { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.main:hover:not(:disabled) { background:#7a3fe0; border-color:#7a3fe0; }
  button.ghost { background:transparent; }
  button:disabled { opacity:.4; cursor:default; }
  button:focus-visible, select:focus-visible, #drop:focus-visible, a:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  #chapters { max-height:300px; overflow:auto; margin-top:12px; border:1px solid var(--line); border-radius:6px; background:var(--bg); }
  .ch, .q { display:flex; gap:10px; padding:8px 12px; border-bottom:1px solid var(--line); align-items:center; font-size:13px; }
  .ch { color:var(--ink); cursor:pointer; } .ch:last-child, .q:last-child { border-bottom:0; } .ch:hover { background:var(--raise); }
  .ch span.t, .q span.t { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .m { color:var(--soft); font-size:12px; white-space:nowrap; }
  .ch.done span.t { color:var(--soft); } .ch.done span.t::after { content:"done"; color:var(--accent-ink); font-size:11px; margin-left:8px; border:1px solid var(--line); border-radius:4px; padding:1px 5px; }
  #qlist { border:1px solid var(--line); border-radius:6px; background:var(--bg); }
  .q .pos { color:var(--faint); width:1.6em; font-size:12px; }
  .q.running span.t { font-weight:600; } .q.running .pos { color:var(--accent-ink); } .q.done span.t { color:var(--soft); } .q.error .m { color:var(--danger); }
  .links { font-size:12px; margin-top:16px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  .links a, .q a { color:var(--accent-ink); cursor:pointer; font-size:12px; } .links a:hover, .q a:hover { text-decoration:underline; }
  #total { color:var(--soft); margin-left:auto; font-size:12px; }
  #cmprows { border:1px solid var(--line); border-radius:6px; background:var(--bg); }
  .c { display:grid; grid-template-columns:130px 1fr; gap:4px 14px; padding:12px; border-bottom:1px solid var(--line); align-items:center; font-size:13px; }
  .c:last-child { border-bottom:0; } .c b { font-weight:600; } .c audio { margin:0; height:32px; } .c .m { white-space:normal; } .c .bad { color:var(--danger); }
  #work { margin-top:16px; border-top:1px solid var(--line); padding-top:16px; }
  #state { font-weight:600; font-size:13px; margin-bottom:14px; }
  .cap { display:flex; justify-content:space-between; gap:12px; font-size:13px; } .cap span:first-child { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .cap.soft { font-size:12px; color:var(--soft); margin-bottom:14px; } #chpct { font-size:13px; }
  #bar { height:10px; background:var(--line); border-radius:3px; overflow:hidden; margin:8px 0 10px; } #bar div { height:100%; width:0; background:var(--accent); transition:width .4s; }
  #segs { display:flex; gap:2px; height:6px; margin-bottom:8px; }
  #segs div { flex:1 1 0; min-width:3px; background:var(--line); border-radius:1px; overflow:hidden; }
  #segs i { display:block; height:100%; width:0; background:var(--accent); transition:width .4s; }
  #segs div.now { outline:1px solid var(--accent-ink); outline-offset:1px; }
  #log { font-size:12px; line-height:1.6; color:var(--soft); white-space:pre-wrap; max-height:140px; overflow:auto; margin:0; }
  #err { color:var(--danger); font-size:13px; min-height:0; margin:0 0 12px; } #err:empty { display:none; }
  audio { width:100%; margin-top:14px; height:36px; } .hide { display:none !important; }
  @media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style></head><body>
<header><div><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="28" height="28" role="img" aria-label="book2audio"><rect x="0.5" y="0.5" width="63" height="63" rx="13" fill="var(--tile)" stroke="var(--tile-line)"/><path d="M13 30 V34 M22 26 V38 M42 26 V38 M51 30 V34" fill="none" stroke="var(--glyph)" stroke-width="6" stroke-linecap="square"/><rect x="28" y="20" width="8" height="24" fill="#8C52EE"/></svg><span class="word">book2audio</span><span class="tag">Ebooks in, audiobooks out. All on this Mac.</span>
<button id="theme" class="ghost" aria-label="Switch between dark and light theme">Light</button>
<button id="quit" class="ghost" title="Stop book2audio. Everything read so far is kept.">Quit</button></div></header>
<div id="gone" class="hide"><div class="card" style="max-width:520px;margin:80px auto;text-align:center"><h2>book2audio has quit</h2><p class="sub">Everything read so far is kept, and the queue will be there next time. You can close this tab.</p></div></div>
<main>

<div id="queue" class="card hide">
  <p class="label">Queue</p>
  <div id="qlist"></div>
  <div class="row">
    <button id="qstart" class="main">Start the queue</button>
    <button id="stop" class="hide">Stop</button>
    <span id="qnote" class="m"></span>
  </div>
  <div id="work" class="hide">
    <div id="state"></div>
    <div class="cap"><span id="chname"></span><span id="chpct" class="mono"></span></div>
    <div id="bar" role="progressbar" aria-label="Current chapter"><div></div></div>
    <div id="segs" aria-label="Chapters of this book"></div>
    <div class="cap soft"><span id="segnote"></span><span id="detail" class="mono"></span></div>
    <div class="links" style="margin-top:0"><a id="more">Show details</a></div>
    <pre id="log" class="hide"></pre>
  </div>
</div>

<div id="drop" class="card" tabindex="0" role="button"><strong>Drop an EPUB here, or click to choose one</strong>AZW3 and MOBI work if Calibre is installed</div>
<ol id="steps"><li><b>Drop a book</b>EPUB without DRM</li><li><b>Hear the voice</b>A 1-minute preview</li><li><b>Queue it and leave</b>It reads chapter by chapter</li></ol>
<input id="file" type="file" accept=".epub,.azw3,.azw,.mobi" class="hide">
<p id="err"></p>

<div id="book" class="card hide">
  <p class="label">Book</p>
  <h2 id="title"></h2><div id="meta" class="sub mono"></div>
  <div class="row">
    <div><label for="model">Voice model</label><select id="model"><option value="voxtral-4bit">Natural (Voxtral 4-bit, recommended)</option><option value="voxtral">Natural, full size (Voxtral, 2.5x slower)</option><option value="pocket">Light, on the CPU (Pocket TTS)</option><option value="kokoro">Fast (Kokoro)</option></select></div>
    <div><label for="voice">Voice</label><select id="voice"></select></div>
    <label class="check"><input type="checkbox" id="iphone" checked> Copy to iCloud Drive for the iPhone</label>
  </div>
  <div class="links"><a id="all">Select all chapters</a><a id="none">Select none</a><a id="todo">Only chapters not done</a><span id="total"></span></div>
  <div id="chapters"></div>
  <div class="row">
    <button id="preview">1-minute preview</button>
    <button id="compare">Compare models</button>
    <button id="add" class="main">Add to the queue</button>
    <button id="folder" class="ghost">Open folder</button>
  </div>
  <audio id="player" controls class="hide"></audio>
  <div id="cmp" class="hide"><p class="label" style="margin-top:20px">Same passage, each model, timed on this Mac</p><div id="cmprows"></div></div>
</div>
</main>
<script>
const $ = id => document.getElementById(id);
const VOX = { fr:["fr_female","fr_male"], en:["casual_female","casual_male","cheerful_female","neutral_female","neutral_male"],
              es:["es_female","es_male"], de:["de_female","de_male"], it:["it_female","it_male"], pt:["pt_female","pt_male"], nl:["nl_female","nl_male"] };
const VOICES = { voxtral: VOX, "voxtral-4bit": VOX,
                 pocket: { fr:["estelle"], en:["alba","marius","cosette","jane","george"], de:["juergen"], es:["lola"], it:["giovanni"], pt:["rafael"] },
                 kokoro: { fr:["ff_siwis"], en:["af_heart","af_bella","am_michael","am_adam","bf_emma","bm_george"], es:["ef_dora","em_alex"], it:["if_sara","im_nicola"], pt:["pf_dora"] } };
const STATUS = { waiting:"waiting", running:"reading now", done:"finished", error:"failed" };
let book = null, lastPreview = 0, timer = null, speeds = {}, quitting = false;
const api = (path, opts) => fetch(path, opts).then(async r => { const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j; });
const post = (path, data) => api(path, { method:"POST", body:JSON.stringify(data || {}) });
const fail = e => { $("err").textContent = e.message || e; };

function fillVoices() {
  const lang = book.language.slice(0, 2), list = (VOICES[$("model").value][lang]) || VOICES[$("model").value].en;
  $("voice").innerHTML = list.map(v => `<option>${v}</option>`).join("");
}
function countSelected() {
  const boxes = [...document.querySelectorAll("#chapters input:checked")];
  const min = boxes.reduce((s, b) => s + book.chapters[b.value - 1].minutes, 0);
  const hours = min / 60, speed = speeds[$("model").value];
  const dur = h => h < 1 ? `${Math.max(1, Math.round(h * 60))} min` : `${h.toFixed(1)} h`;
  $("total").textContent = `${boxes.length} chapters · ${dur(hours)} of audio` + (speed && boxes.length ? ` · about ${dur(hours / speed)} to create` : "");
  return boxes.map(b => +b.value);
}
function showBook(b) {
  book = b; $("err").textContent = "";
  $("title").textContent = b.title; $("meta").textContent = [b.author, b.language, b.out].filter(Boolean).join(" · ");
  $("model").value = b.language.startsWith("fr") ? "voxtral-4bit" : "kokoro"; fillVoices();
  $("chapters").innerHTML = "";
  b.chapters.forEach(c => {
    const row = document.createElement("label"); row.className = "ch" + (c.done ? " done" : "");
    row.innerHTML = `<input type="checkbox" value="${c.n}" ${c.done || c.skip ? "" : "checked"}><span class="t"></span><span class="m">${c.minutes} min</span>`;
    row.querySelector(".t").textContent = c.title;
    if (c.skip && !c.done) { const k = document.createElement("span"); k.className = "tagk"; k.textContent = "front matter"; row.querySelector(".t").append(k); }
    $("chapters").append(row);
  });
  $("book").classList.remove("hide"); $("player").classList.add("hide"); $("steps").classList.add("hide"); countSelected();
}
function upload(file) {
  if (!file) return; $("drop").innerHTML = "<strong>Reading the book...</strong>"; $("drop").append(file.name);
  api("/open", { method:"POST", headers:{ "X-Name": encodeURIComponent(file.name) }, body:file })
    .then(showBook).catch(fail).finally(() => { $("drop").innerHTML = "<strong>Drop another book here, or click to choose one</strong>It will be added behind the others"; });
}
function settings() {
  const chapters = countSelected(); if (!chapters.length) { fail("Select at least one chapter."); return null; }
  $("err").textContent = "";
  return { chapters, model:$("model").value, voice:$("voice").value, iphone:$("iphone").checked };
}
function showQueue(s) {
  $("queue").classList.toggle("hide", !s.queue.length && !s.lines.length);
  $("qlist").innerHTML = "";
  s.queue.forEach((q, i) => {
    const row = document.createElement("div"); row.className = "q " + q.status;
    row.innerHTML = `<span class="pos">${i + 1}</span><span class="t"></span><span class="m"></span><a class="open">open</a>` + (q.status === "running" ? "" : `<a class="rm">remove</a>`);
    row.querySelector(".t").textContent = q.title;
    row.querySelector(".m").textContent = `${q.done} of ${q.chapters} chapters · ${q.status === "error" && q.note ? q.note : STATUS[q.status]}`;
    row.querySelector(".open").onclick = () => post("/folder", { id:q.id }).catch(fail);
    const a = row.querySelector(".rm"); if (a) a.onclick = () => post("/queue/remove", { id:q.id }).then(poll).catch(fail);
    $("qlist").append(row);
  });
  const todo = s.queue.filter(q => q.status === "waiting" || q.status === "error").length;
  const started = s.queue.some(q => q.done > 0 && q.status !== "done");
  $("qstart").textContent = started ? "Resume the queue" : "Start the queue";
  $("qstart").classList.toggle("hide", s.running); $("qstart").disabled = !todo;
  $("stop").classList.toggle("hide", !s.running);
  $("qnote").textContent = s.running ? "" : (todo ? `${todo} book${todo > 1 ? "s" : ""} to read` : (s.queue.length ? "Nothing left to read" : ""));
}
function showProgress(s) {
  const segs = s.running ? s.segments : [], now = segs.find(g => g.n === s.now);
  const pct = s.running && s.total ? Math.floor(100 * s.i / s.total) : 0;
  $("chname").textContent = now ? now.title : (s.running && s.total ? "Preview" : "");
  $("chpct").textContent = s.running && s.total ? pct + "%" : "";
  $("bar").firstElementChild.style.width = pct + "%";
  $("detail").textContent = s.running && s.total ? `passage ${s.i} of ${s.total}` : "";
  const box = $("segs"); box.classList.toggle("hide", !segs.length);
  if (box.children.length !== segs.length) box.innerHTML = segs.map(() => "<div><i></i></div>").join("");
  segs.forEach((g, k) => { const d = box.children[k];
    d.style.flexGrow = g.w; d.className = g.n === s.now ? "now" : ""; d.title = `${g.title}: ${g.p}%`; d.firstChild.style.width = g.p + "%"; });
  const done = segs.filter(g => g.p === 100).length;
  $("segnote").textContent = segs.length ? `${done} of ${segs.length} chapters done` : "";
}
let cmpSeen = "";
function showCompare(s) {
  const rows = s.compare || []; $("cmp").classList.toggle("hide", !rows.length);
  const sig = JSON.stringify(rows) + s.comparing; if (sig === cmpSeen) return; cmpSeen = sig;   // do not restart players that are playing
  $("cmprows").innerHTML = "";
  rows.forEach(r => {
    const d = document.createElement("div"); d.className = "c";
    const waiting = !r.file && !r.error;
    d.innerHTML = `<b></b><span class="m"></span>` + (r.file ? `<span></span><audio controls preload="none" src="/audio?f=${encodeURIComponent(r.file)}&t=${Date.now()}"></audio>` : "");
    d.querySelector("b").textContent = r.label;
    const m = d.querySelector(".m");
    if (r.error) { m.textContent = "failed: " + r.error; m.classList.add("bad"); }
    else if (waiting) m.textContent = s.comparing === r.model ? "reading..." : "waiting";
    else m.textContent = `${r.speed}x real time · ${r.audio_s} s of audio in ${r.compute_s} s · ${r.chars_s} characters/s · loaded in ${r.load_s} s`;
    $("cmprows").append(d);
  });
}
function poll() {
  clearTimeout(timer);
  api("/status").then(s => {
    speeds = s.speeds || {}; if (book) countSelected();
    $("steps").classList.toggle("hide", !!book || s.queue.length > 0);
    showQueue(s);
    $("work").classList.toggle("hide", !s.lines.length);
    $("state").textContent = s.running ? (s.current ? s.current + ": " : "") + (s.total ? `${s.percent}% done, ${s.left}` : "getting ready...") : (s.error ? "Stopped on an error" : "Idle");
    showProgress(s); showCompare(s);
    $("log").textContent = s.lines.join("\n") + (s.error ? "\n" + s.error : "");
    if (s.error) { $("log").classList.remove("hide"); $("more").textContent = "Hide details"; }   // an error is never hidden
    $("log").scrollTop = 1e9;
    $("preview").disabled = $("compare").disabled = s.running;
    if (s.preview && s.preview_at !== lastPreview) { lastPreview = s.preview_at; $("player").src = "/audio?f=" + encodeURIComponent(s.preview) + "&t=" + s.preview_at; $("player").classList.remove("hide"); $("player").play().catch(() => {}); }
    if (book && s.book_done) s.book_done.forEach(n => { const row = $("chapters").children[n - 1]; if (row) row.classList.add("done"); });
    if (s.running) timer = setTimeout(poll, 1000);
  }).catch(() => { if (quitting) { document.querySelector("main").classList.add("hide"); $("gone").classList.remove("hide"); } else timer = setTimeout(poll, 3000); });
}
$("drop").onclick = () => $("file").click();
$("drop").onkeydown = e => { if (e.key === "Enter" || e.key === " ") $("file").click(); };
$("file").onchange = e => { upload(e.target.files[0]); e.target.value = ""; };   // so the same book can be chosen again
["dragover","dragenter"].forEach(n => $("drop").addEventListener(n, e => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave","drop"].forEach(n => $("drop").addEventListener(n, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
$("drop").addEventListener("drop", e => upload(e.dataTransfer.files[0]));
$("model").onchange = () => { fillVoices(); countSelected(); }; $("chapters").onchange = countSelected;
$("more").onclick = () => { const hidden = $("log").classList.toggle("hide"); $("more").textContent = hidden ? "Show details" : "Hide details"; $("log").scrollTop = 1e9; };
$("quit").onclick = () => { quitting = true; post("/quit").catch(() => {}).finally(() => setTimeout(poll, 1500)); };
const setAll = fn => { document.querySelectorAll("#chapters .ch").forEach(r => r.querySelector("input").checked = fn(r)); countSelected(); };
$("all").onclick = () => setAll(r => !r.querySelector(".tagk")); $("none").onclick = () => setAll(() => false); $("todo").onclick = () => setAll(r => !r.classList.contains("done"));
$("preview").onclick = () => { const d = settings(); if (d) post("/preview", d).then(poll).catch(fail); };
$("compare").onclick = () => { const d = settings(); if (d) post("/compare", d).then(poll).catch(fail); };
$("add").onclick = () => { const d = settings(); if (d) post("/queue/add", d).then(() => { $("book").classList.add("hide"); poll(); window.scrollTo(0, 0); }).catch(fail); };
$("qstart").onclick = () => post("/queue/start").then(poll).catch(fail);
$("stop").onclick = () => post("/stop").then(poll).catch(fail);
$("folder").onclick = () => post("/folder").catch(fail);
const themeLabel = () => { $("theme").textContent = document.documentElement.dataset.theme === "light" ? "Dark" : "Light"; };
$("theme").onclick = () => { const t = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = t; try { localStorage.setItem("theme", t); } catch (e) {} themeLabel(); };
themeLabel();
poll();
</script></body></html>
"""

QUEUE_FILE = Path.home() / "Audiobooks" / ".queue.json"


class WebReport(Report):
    def __init__(self):
        self.lines: list[str] = []
        self.i = self.total = 0
        self.running = False
        self.stop = False
        self.error = ""
        self.preview = ""
        self.preview_at = 0
        self.percent = 0
        self.left = ""
        self.current = ""
        self.results: list[dict] = []      # model comparison, shown under the book
        self.comparing = ""
        self.segments: list[dict] = []     # one per chapter of the book being read
        self.now = 0                       # number of the chapter being read

    def log(self, message: str) -> None:
        print(message)
        self.lines = (self.lines + [message])[-200:]

    def plan(self, chapters: list[dict]) -> None:
        self.segments = [{"n": c["n"], "title": c["title"], "w": max(c["chars"], 1), "p": 100 if c["done"] else 0} for c in chapters]
        self.now, self.i, self.total = 0, 0, 0

    def compare(self, results: list[dict], running: str) -> None:
        self.results, self.comparing = [dict(r) for r in results], running
        self.i = self.total = 0
        if running:
            self.current = f"Comparing models, {MODEL_LABELS.get(running, running)}"

    def chapter(self, n: int, finished: bool = False) -> None:
        self.now, self.i, self.total = n, 0, 0
        for seg in self.segments:
            if seg["n"] == n:
                seg["p"] = 100 if finished else 0

    def progress(self, i: int, n: int, minutes: float, tracker: Tracker | None = None) -> None:
        self.i, self.total = i, n
        for seg in self.segments:
            if seg["n"] == self.now:
                seg["p"] = min(99, int(100 * i / n))     # 100 only once the file is written
        if tracker:
            self.percent, self.left = tracker.percent, tracker.left_text()
        if self.stop:
            raise Stopped()


def serve_ui(args, port: int = 8765, open_browser: bool = True) -> int:
    global REPORT
    state = REPORT = WebReport()
    jobs: queue.Queue = queue.Queue()
    session: dict = {"book": None, "out": None, "epub": None}
    uploads = Path(tempfile.mkdtemp(prefix="book2audio-"))
    lock = threading.Lock()

    # The queue is a small file on disk, rewritten at every change: whatever
    # happens (Stop, crash, reboot), the next launch finds it as it was.
    try:
        items: list[dict] = json.loads(QUEUE_FILE.read_text())
    except (OSError, ValueError):
        items = []
    for item in items:
        if item.get("status") == "running":          # interrupted last time
            item["status"] = "waiting"

    def save_queue() -> None:
        QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = QUEUE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, indent=1))
        tmp.replace(QUEUE_FILE)

    def files_done(out: Path, chapters: list[int]) -> list[int]:
        return [n for n in chapters if out.exists() and next(iter(out.glob(f"{n:02d} - *.m4a")), None)]

    def options(job: dict):
        return argparse.Namespace(**{**vars(args), "model": job.get("model") or args.model,
                                     "voice": job.get("voice") or args.voice, "iphone": bool(job.get("iphone"))})

    def language_for(book: Book, opts) -> str:
        if "kokoro" in (opts.model or "") and (opts.voice or "").startswith("b"):
            return "en-gb"
        return (args.lang or book.language or "en").lower()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the terminal quiet
            pass

        def reply(self, data, status: int = 200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/status":
                book = session["book"]
                with lock:
                    view = [{"id": q["id"], "title": q["title"], "status": q["status"], "note": q.get("note", ""),
                             "chapters": len(q["chapters"]), "done": len(files_done(Path(q["out"]), q["chapters"]))} for q in items]
                self.reply({"running": state.running, "lines": state.lines[-12:], "i": state.i, "total": state.total,
                            "error": state.error, "percent": state.percent, "left": state.left, "current": state.current,
                            "preview": state.preview, "preview_at": state.preview_at, "queue": view,
                            "compare": state.results, "comparing": state.comparing,
                            "segments": state.segments if state.running else [], "now": state.now,
                            "speeds": load_speeds(),
                            "book_done": files_done(session["out"], list(range(1, len(book.chapters) + 1))) if book else []})
            elif url.path == "/audio" and session["out"]:
                name = Path(parse_qs(url.query).get("f", [""])[0]).name
                path = session["out"] / name
                if not name or not path.is_file():
                    return self.reply({"error": "not found"}, 404)
                data = path.read_bytes()
                start, end = 0, len(data) - 1
                rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
                if rng and (rng.group(1) or rng.group(2)):       # Safari only plays audio served in ranges
                    if rng.group(1):
                        start = int(rng.group(1))
                        end = min(int(rng.group(2)), end) if rng.group(2) else end
                    else:
                        start = max(0, len(data) - int(rng.group(2)))
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Type", "audio/mp4" if path.suffix in (".m4a", ".m4b") else "audio/wav")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                self.end_headers()
                self.wfile.write(data[start:end + 1])
            else:
                self.reply({"error": "not found"}, 404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.reply({"error": "forbidden"}, 403)   # only this page may drive the app
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                if self.path == "/open":
                    name = Path(unquote(self.headers.get("X-Name", "book.epub"))).name
                    path = uploads / name
                    path.write_bytes(body)
                    book = open_book(path, args.min_chars)
                    session.update(book=book, out=args.out or default_out(book), epub=path)
                    state.preview, state.results = "", []
                    everything = list(range(1, len(book.chapters) + 1))
                    done = files_done(session["out"], everything)
                    self.reply({"title": book.title, "author": book.author, "language": (args.lang or book.language or "en").lower(),
                                "out": str(session["out"]).replace(str(Path.home()), "~"),
                                "chapters": [{"n": n, "title": c.title, "minutes": round(c.chars / 900), "done": n in done,
                                              "skip": is_front_matter(c)}
                                             for n, c in enumerate(book.chapters, 1)]})
                elif self.path == "/compare":
                    if state.running:
                        return self.reply({"error": "Already working. Stop it first."}, 409)
                    if not session["book"]:
                        return self.reply({"error": "Choose a book first."}, 400)
                    state.running, state.stop, state.error, state.lines, state.current = True, False, "", [], "Comparing models"
                    state.percent, state.left, state.segments, state.now, state.results = 0, "", [], 0, []
                    jobs.put({"kind": "compare", **json.loads(body)})
                    self.reply({"ok": True})
                elif self.path == "/preview":
                    if state.running:
                        return self.reply({"error": "Already working. Stop it first."}, 409)
                    if not session["book"]:
                        return self.reply({"error": "Choose a book first."}, 400)
                    state.running, state.stop, state.error, state.lines, state.current = True, False, "", [], "Preview"
                    state.percent, state.left, state.segments, state.now = 0, "", [], 0
                    jobs.put({"kind": "preview", **json.loads(body)})
                    self.reply({"ok": True})
                elif self.path == "/queue/add":
                    book, out = session["book"], session["out"]
                    if not book:
                        return self.reply({"error": "Choose a book first."}, 400)
                    job = json.loads(body)
                    chapters = sorted(n for n in job.get("chapters", []) if 1 <= n <= len(book.chapters))
                    out.mkdir(parents=True, exist_ok=True)
                    source = out / (".source" + session["epub"].suffix.lower())   # kept so the queue survives a restart
                    shutil.copy2(session["epub"], source)
                    with lock:
                        if any(q["out"] == str(out) and q["status"] == "running" for q in items):
                            return self.reply({"error": "This book is being read right now."}, 409)
                        items[:] = [q for q in items if not (q["out"] == str(out) and q["status"] != "done")]
                        items.append({"id": str(int(time.time() * 1000)), "title": book.title, "epub": str(source), "out": str(out),
                                      "chapters": chapters, "model": job.get("model"), "voice": job.get("voice"),
                                      "iphone": bool(job.get("iphone")), "status": "waiting", "note": ""})
                        save_queue()
                    self.reply({"ok": True})
                elif self.path == "/queue/remove":
                    wanted = json.loads(body).get("id")
                    with lock:
                        items[:] = [q for q in items if q["id"] != wanted or q["status"] == "running"]
                        save_queue()
                    self.reply({"ok": True})
                elif self.path == "/queue/start":
                    if state.running:
                        return self.reply({"error": "Already working."}, 409)
                    with lock:
                        for q in items:
                            if q["status"] == "error":       # an explicit start is also a retry
                                q["status"], q["note"] = "waiting", ""
                        save_queue()
                    state.running, state.stop, state.error, state.lines, state.current = True, False, "", [], ""
                    state.percent, state.left = 0, ""
                    jobs.put({"kind": "queue"})
                    self.reply({"ok": True})
                elif self.path == "/stop":
                    state.stop = True
                    self.reply({"ok": True})
                elif self.path == "/folder":
                    wanted = (json.loads(body) if body else {}).get("id")
                    with lock:
                        folder = next((Path(q["out"]) for q in items if q["id"] == wanted), None) if wanted else session["out"]
                    if folder:
                        folder.mkdir(parents=True, exist_ok=True)
                        subprocess.run(["open", str(folder)], check=False)
                    self.reply({"ok": True})
                elif self.path == "/quit":
                    state.stop = True
                    jobs.put(None)
                    self.reply({"ok": True})
                else:
                    self.reply({"error": "not found"}, 404)
            except SystemExit as e:                  # friendly errors raised while reading the book
                self.reply({"error": str(e)}, 400)
            except Exception as e:
                self.reply({"error": f"{type(e).__name__}: {e}"}, 500)

    def run_queue() -> None:
        """Read the waiting books one after the other, in the order of the list."""
        while not state.stop:
            with lock:
                item = next((q for q in items if q["status"] == "waiting"), None)
                if not item:
                    break
                item["status"] = "running"
                position = sum(1 for q in items if q["status"] in ("done", "running", "error"))
                save_queue()
            state.current = f"Book {position} of {len(items)}, {item['title']}"
            state.percent, state.left, state.segments, state.now = 0, "", [], 0
            state.log(f"=== {item['title']} ===")
            status, note = "done", ""
            try:
                book = open_book(Path(item["epub"]), args.min_chars)
                opts = options(item)
                run_book(book, item["chapters"], language_for(book, opts), opts, Path(item["out"]))
            except (Stopped, KeyboardInterrupt):
                status = "waiting"                   # its position is kept: it goes on from there next time
                raise
            except BaseException as e:               # one bad book must not block the ones behind it
                status, note = "error", (str(e) if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}")[:200]
                state.log(f"Could not finish this book: {note}")
            finally:
                with lock:
                    item["status"], item["note"] = status, note
                    save_queue()
        if not state.stop:
            state.log("The queue is finished.")

    url = f"http://127.0.0.1:{port}/"
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        try:                                         # already running: just show it
            import urllib.request
            json.loads(urllib.request.urlopen(url + "status", timeout=3).read())["queue"]
        except Exception:
            sys.exit(f"Port {port} is used by another program, so book2audio cannot start.")
        print("book2audio is already running: opening its page.")
        if open_browser:
            webbrowser.open(url)
        return 0
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"book2audio is running at {url}  (close this window or press Ctrl+C to quit)")
    if open_browser:
        webbrowser.open(url)

    # The voice model runs here, in the main thread: the web server only hands over jobs.
    while True:
        job = jobs.get()
        if job is None:
            break
        try:
            if job["kind"] == "compare":
                book, opts = session["book"], options(job)
                first = min(n for n in job.get("chapters", [1]) if 1 <= n <= len(book.chapters))
                run_compare(book, first, (args.lang or book.language or "en").lower(), opts, session["out"])
            elif job["kind"] == "preview":
                book, opts = session["book"], options(job)
                first = min(n for n in job.get("chapters", [1]) if 1 <= n <= len(book.chapters))
                target = run_preview(book, first, language_for(book, opts), opts, session["out"])
                state.preview, state.preview_at = target.name, int(time.time())
            else:
                run_queue()
        except Stopped:
            state.log("Stopped. Everything read so far is kept: resume the queue to continue.")
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            state.error = f"{type(e).__name__}: {e}" if not isinstance(e, SystemExit) else str(e)
            import traceback
            traceback.print_exc()
        finally:
            state.running = False
            state.i = state.total = 0
    server.shutdown()
    return 0


def cli() -> None:
    """Entry point of the `book2audio` command."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where it left off.")
        sys.exit(130)


if __name__ == "__main__":
    cli()
