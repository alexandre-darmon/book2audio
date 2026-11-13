#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "mlx-audio",
#   "misaki[en]",
#   "mistral-common[audio]",   # needed by Voxtral
#   "beautifulsoup4",
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
    "voxtral": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",   # 8 GB, more natural in French, non-commercial licence
    "voxtral-4bit": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",   # 2.5 GB compressed version, to compare speed and quality
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
            sys.exit("mlx-audio is not installed (it needs an Apple Silicon Mac).\n"
                     "Run this script with:  uv run book2audio.py ...")
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
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", str(pcm),
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
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)],
                                   capture_output=True, text=True, check=True)
            end = start + int(float(probe.stdout.strip()) * 1000)
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}", f"title={esc(title)}"]
            start = end
            quoted = str(f.resolve()).replace("'", "'\\''")
            lines.append(f"file '{quoted}'")
        listing.write_text("\n".join(lines), encoding="utf-8")
        meta_file = Path(tmp) / "meta.txt"
        meta_file.write_text("\n".join(meta), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
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
    if wanted != "wav" and not shutil.which("ffmpeg"):
        REPORT.log("ffmpeg not found: writing WAV files, no audiobook file (install it with: brew install ffmpeg).")
        return "wav"
    return wanted


_synth_cache: dict = {}


def make_synth(args, language: str):
    """The model stays loaded between two jobs that use the same voice."""
    model = args.model or ("voxtral" if language.startswith("fr") else "kokoro")
    key = (model, language, args.voice, args.speed, args.lang_code)
    if key not in _synth_cache:
        _synth_cache.clear()
        _synth_cache[key] = Synth(model, language, args.voice, args.speed, args.lang_code)
    return _synth_cache[key]


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


def run_book(book: Book, selection: list[int], language: str, args, out_dir: Path) -> None:
    fmt = pick_format(args.format)
    out_dir.mkdir(parents=True, exist_ok=True)
    read_titles = not args.no_titles
    todo = [n for n in selection if not chapter_target(out_dir, n, book.chapters[n - 1], fmt).exists()]
    for n in selection:
        if n not in todo:
            REPORT.log(f"[{n}/{len(book.chapters)}] {book.chapters[n - 1].title}: already done, skipping")
    scripts = {n: build_script(book.chapters[n - 1], read_titles) for n in todo}
    tracker = Tracker(sum(script_chars(s) for s in scripts.values()))
    started = time.time()
    audio_seconds = 0.0
    with keep_awake():
        synth = make_synth(args, language) if todo else None
        for n in todo:
            chapter = book.chapters[n - 1]
            REPORT.log(f"[{n}/{len(book.chapters)}] {chapter.title}")
            pcm = out_dir / f".tmp-{n:02d}.pcm"
            _, seconds, rate = render_chapter(synth, scripts[n], pcm, tracker)
            audio_seconds += seconds
            encode(pcm, rate, chapter_target(out_dir, n, chapter, fmt),
                   {"title": chapter.title, "album": book.title, "artist": book.author, "track": str(n), "genre": "Audiobook"})

    elapsed = time.time() - started
    if audio_seconds:
        REPORT.log(f"Done: {fmt_duration(audio_seconds)} of audio in {fmt_duration(elapsed)} ({audio_seconds / max(elapsed, 1):.1f}x real time).")
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
    ap.add_argument("--lang", help="book language, e.g. en, en-gb, fr (default: read from the book)")
    ap.add_argument("--voice", help="voice name (default: a good voice for the language)")
    ap.add_argument("--speed", type=float, default=1.0, help="reading speed, 1.0 = normal")
    ap.add_argument("--model", help="kokoro, voxtral, voxtral-4bit, or any mlx-audio TTS model id (default: voxtral for French books, kokoro otherwise)")
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
    if args.preview:
        run_preview(book, selection[0], language, args, out_dir)
    else:
        run_book(book, selection, language, args, out_dir)
    return 0


# --------------------------------------------------------------------------
# 6. The simple web page (uv run book2audio.py --ui)
# --------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>book2audio</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --ink:#1c1b19; --soft:#6b675f; --line:#e2ddd3; --paper:#faf8f4; --card:#fff; --accent:#b4532a; }
  @media (prefers-color-scheme: dark) { :root { --ink:#ece8df; --soft:#a09a8e; --line:#3a3731; --paper:#191816; --card:#22201d; --accent:#e08a5c; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--paper); color:var(--ink); font:16px/1.5 -apple-system, "Helvetica Neue", sans-serif; }
  main { max-width:720px; margin:0 auto; padding:40px 20px 80px; }
  h1 { font:600 28px/1.2 "Iowan Old Style", Georgia, serif; margin:0 0 4px; }
  .sub { color:var(--soft); margin:0 0 28px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:20px; margin-bottom:16px; }
  #drop { border:2px dashed var(--line); text-align:center; padding:36px 20px; cursor:pointer; color:var(--soft); }
  #drop.over { border-color:var(--accent); color:var(--ink); }
  h2 { font:600 20px/1.3 "Iowan Old Style", Georgia, serif; margin:0; }
  .row { display:flex; gap:12px; flex-wrap:wrap; align-items:end; margin-top:14px; }
  label { font-size:13px; color:var(--soft); display:block; }
  select { font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--ink); margin-top:2px; }
  button { font:inherit; padding:8px 16px; border-radius:6px; border:1px solid var(--line); background:var(--card); color:var(--ink); cursor:pointer; }
  button.main { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  button:disabled { opacity:.45; cursor:default; }
  button:focus-visible, select:focus-visible, #drop:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  #chapters { max-height:300px; overflow:auto; margin-top:14px; border-top:1px solid var(--line); }
  .ch, .q { display:flex; gap:10px; padding:7px 2px; border-bottom:1px solid var(--line); align-items:center; }
  .ch span.t, .q span.t { flex:1; } .m { color:var(--soft); font-variant-numeric:tabular-nums; font-size:14px; white-space:nowrap; }
  .ch.done span.t::after { content:"  done"; color:var(--accent); font-size:13px; }
  .q .pos { color:var(--soft); width:1.4em; text-align:right; font-variant-numeric:tabular-nums; }
  .q.running span.t { font-weight:600; } .q.done span.t { color:var(--soft); } .q.error .m { color:#c0392b; }
  .links { font-size:14px; margin-top:8px; } .links a, .q a { color:var(--accent); cursor:pointer; margin-right:14px; font-size:14px; }
  #bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin:12px 0 8px; } #bar div { height:100%; width:0; background:var(--accent); transition:width .4s; }
  #log { font:13px/1.5 ui-monospace, Menlo, monospace; color:var(--soft); white-space:pre-wrap; max-height:150px; overflow:auto; margin:0; }
  #err { color:#c0392b; } audio { width:100%; margin-top:12px; } .hide { display:none; }
</style></head><body><main>
<h1>book2audio</h1>
<p class="sub">Ebooks in, audiobooks out. Everything runs on this Mac.</p>

<div id="queue" class="card hide">
  <h2>Queue</h2>
  <div id="qlist"></div>
  <div class="row">
    <button id="qstart" class="main">Start the queue</button>
    <button id="stop" class="hide">Stop</button>
    <span id="qnote" class="m"></span>
  </div>
  <div id="work" class="hide" style="margin-top:14px"><strong id="state"></strong> <span id="detail" class="m"></span><div id="bar"><div></div></div><pre id="log"></pre></div>
</div>

<div id="drop" class="card" tabindex="0" role="button">Drop an EPUB here, or click to choose one<br><small>AZW3 and MOBI work if Calibre is installed</small></div>
<input id="file" type="file" accept=".epub,.azw3,.azw,.mobi" class="hide">
<p id="err"></p>

<div id="book" class="card hide">
  <h2 id="title"></h2><div id="meta" class="sub" style="margin:0"></div>
  <div class="row">
    <div><label for="model">Voice model</label><select id="model"><option value="voxtral">Natural (Voxtral)</option><option value="voxtral-4bit">Natural, lighter (Voxtral 4-bit)</option><option value="kokoro">Fast (Kokoro)</option></select></div>
    <div><label for="voice">Voice</label><select id="voice"></select></div>
    <div><label><input type="checkbox" id="iphone" checked> Copy to iCloud Drive for the iPhone</label></div>
  </div>
  <div class="links"><a id="all">Select all</a><a id="none">Select none</a><a id="todo">Only chapters not done</a><span id="total"></span></div>
  <div id="chapters"></div>
  <div class="row">
    <button id="preview">1-minute preview</button>
    <button id="add" class="main">Add to the queue</button>
    <button id="folder">Open folder</button>
  </div>
  <audio id="player" controls class="hide"></audio>
</div>
</main>
<script>
const $ = id => document.getElementById(id);
const VOX = { fr:["fr_female","fr_male"], en:["casual_female","casual_male","cheerful_female","neutral_female","neutral_male"],
              es:["es_female","es_male"], de:["de_female","de_male"], it:["it_female","it_male"], pt:["pt_female","pt_male"], nl:["nl_female","nl_male"] };
const VOICES = { voxtral: VOX, "voxtral-4bit": VOX,
                 kokoro: { fr:["ff_siwis"], en:["af_heart","af_bella","am_michael","am_adam","bf_emma","bm_george"], es:["ef_dora","em_alex"], it:["if_sara","im_nicola"], pt:["pf_dora"] } };
const STATUS = { waiting:"waiting", running:"reading now", done:"finished", error:"failed" };
let book = null, lastPreview = 0, timer = null;
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
  $("total").textContent = `${boxes.length} chapters, about ${(min / 60).toFixed(1)} h of audio`;
  return boxes.map(b => +b.value);
}
function showBook(b) {
  book = b; $("err").textContent = "";
  $("title").textContent = b.title; $("meta").textContent = [b.author, b.language, b.out].filter(Boolean).join(" · ");
  $("model").value = b.language.startsWith("fr") ? "voxtral" : "kokoro"; fillVoices();
  $("chapters").innerHTML = "";
  b.chapters.forEach(c => {
    const row = document.createElement("label"); row.className = "ch" + (c.done ? " done" : "");
    row.innerHTML = `<input type="checkbox" value="${c.n}" ${c.done ? "" : "checked"}><span class="t"></span><span class="m">${c.minutes} min</span>`;
    row.querySelector(".t").textContent = c.title; $("chapters").append(row);
  });
  $("book").classList.remove("hide"); $("player").classList.add("hide"); countSelected();
}
function upload(file) {
  if (!file) return; $("drop").textContent = "Reading " + file.name + "...";
  api("/open", { method:"POST", headers:{ "X-Name": encodeURIComponent(file.name) }, body:file })
    .then(showBook).catch(fail).finally(() => { $("drop").innerHTML = "Drop another book here, or click to choose one"; });
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
    row.innerHTML = `<span class="pos">${i + 1}</span><span class="t"></span><span class="m"></span>` + (q.status === "running" ? "" : `<a>remove</a>`);
    row.querySelector(".t").textContent = q.title;
    row.querySelector(".m").textContent = `${q.done} of ${q.chapters} chapters · ${q.status === "error" && q.note ? q.note : STATUS[q.status]}`;
    const a = row.querySelector("a"); if (a) a.onclick = () => post("/queue/remove", { id:q.id }).then(poll).catch(fail);
    $("qlist").append(row);
  });
  const todo = s.queue.filter(q => q.status === "waiting" || q.status === "error").length;
  const started = s.queue.some(q => q.done > 0 && q.status !== "done");
  $("qstart").textContent = started ? "Resume the queue" : "Start the queue";
  $("qstart").classList.toggle("hide", s.running); $("qstart").disabled = !todo;
  $("stop").classList.toggle("hide", !s.running);
  $("qnote").textContent = s.running ? "" : (todo ? `${todo} book${todo > 1 ? "s" : ""} to read` : (s.queue.length ? "Nothing left to read" : ""));
}
function poll() {
  clearTimeout(timer);
  api("/status").then(s => {
    showQueue(s);
    $("work").classList.toggle("hide", !s.lines.length);
    $("state").textContent = s.running ? (s.current ? s.current + ": " : "") + (s.total ? `${s.percent}% done, ${s.left}` : "getting ready...") : (s.error ? "Stopped on an error" : "Idle");
    $("detail").textContent = s.running && s.total ? `passage ${s.i} of ${s.total} in the current chapter` : "";
    $("bar").firstElementChild.style.width = s.running ? s.percent + "%" : "0";
    $("log").textContent = s.lines.join("\n") + (s.error ? "\n" + s.error : ""); $("log").scrollTop = 1e9;
    $("preview").disabled = s.running;
    if (s.preview && s.preview_at !== lastPreview) { lastPreview = s.preview_at; $("player").src = "/audio?f=" + encodeURIComponent(s.preview) + "&t=" + s.preview_at; $("player").classList.remove("hide"); $("player").play().catch(() => {}); }
    if (book && s.book_done) s.book_done.forEach(n => { const row = $("chapters").children[n - 1]; if (row) row.classList.add("done"); });
    if (s.running) timer = setTimeout(poll, 1000);
  }).catch(() => { timer = setTimeout(poll, 3000); });
}
$("drop").onclick = () => $("file").click();
$("drop").onkeydown = e => { if (e.key === "Enter" || e.key === " ") $("file").click(); };
$("file").onchange = e => upload(e.target.files[0]);
["dragover","dragenter"].forEach(n => $("drop").addEventListener(n, e => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave","drop"].forEach(n => $("drop").addEventListener(n, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
$("drop").addEventListener("drop", e => upload(e.dataTransfer.files[0]));
$("model").onchange = fillVoices; $("chapters").onchange = countSelected;
const setAll = fn => { document.querySelectorAll("#chapters .ch").forEach(r => r.querySelector("input").checked = fn(r)); countSelected(); };
$("all").onclick = () => setAll(() => true); $("none").onclick = () => setAll(() => false); $("todo").onclick = () => setAll(r => !r.classList.contains("done"));
$("preview").onclick = () => { const d = settings(); if (d) post("/preview", d).then(poll).catch(fail); };
$("add").onclick = () => { const d = settings(); if (d) post("/queue/add", d).then(() => { $("book").classList.add("hide"); poll(); window.scrollTo(0, 0); }).catch(fail); };
$("qstart").onclick = () => post("/queue/start").then(poll).catch(fail);
$("stop").onclick = () => post("/stop").then(poll).catch(fail);
$("folder").onclick = () => post("/folder").catch(fail);
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

    def log(self, message: str) -> None:
        print(message)
        self.lines = (self.lines + [message])[-200:]
        self.i = self.total = 0

    def progress(self, i: int, n: int, minutes: float, tracker: Tracker | None = None) -> None:
        self.i, self.total = i, n
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
                    state.preview = ""
                    everything = list(range(1, len(book.chapters) + 1))
                    done = files_done(session["out"], everything)
                    self.reply({"title": book.title, "author": book.author, "language": (args.lang or book.language or "en").lower(),
                                "out": str(session["out"]).replace(str(Path.home()), "~"),
                                "chapters": [{"n": n, "title": c.title, "minutes": round(c.chars / 900), "done": n in done}
                                             for n, c in enumerate(book.chapters, 1)]})
                elif self.path == "/preview":
                    if state.running:
                        return self.reply({"error": "Already working. Stop it first."}, 409)
                    if not session["book"]:
                        return self.reply({"error": "Choose a book first."}, 400)
                    state.running, state.stop, state.error, state.lines, state.current = True, False, "", [], "Preview"
                    state.percent, state.left = 0, ""
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
                    if session["out"]:
                        session["out"].mkdir(parents=True, exist_ok=True)
                        subprocess.run(["open", str(session["out"])], check=False)
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
            state.percent, state.left = 0, ""
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

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"book2audio is running at {url}  (close this window or press Ctrl+C to quit)")
    if open_browser:
        webbrowser.open(url)

    # The voice model runs here, in the main thread: the web server only hands over jobs.
    while True:
        job = jobs.get()
        if job is None:
            break
        try:
            if job["kind"] == "preview":
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


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where it left off.")
        sys.exit(130)
