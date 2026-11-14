#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "mlx-audio; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "misaki[en]; sys_platform == 'darwin' and platform_machine == 'arm64'",
#   "mistral-common[audio]; sys_platform == 'darwin' and platform_machine == 'arm64'",   # needed by Voxtral
#   "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl ; sys_platform == 'darwin' and platform_machine == 'arm64'",   # Kokoro's English needs it, and cannot fetch it by itself
#   "mobi",                    # reads Kindle files (AZW3, MOBI) without Calibre
#   "beautifulsoup4",
#   "imageio-ffmpeg",          # brings its own ffmpeg, so nothing else has to be installed
#   "soundfile",
#   "numpy",
# ]
# ///
"""book2audio: turn an ebook (EPUB, AZW3 or MOBI without DRM) into one audio
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
    "clone": "mlx-community/chatterbox-fp16",                  # reads with a voice recorded by the user (Chatterbox, MIT)
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
PAUSE_SCENE = 1.4       # at a scene break ("* * *")
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
    cover: bytes = b""          # the cover image found inside the ebook, if any
    cover_ext: str = ""


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
    for tag in body.find_all(["script", "style", "nav", "table", "svg", "head"]):
        tag.decompose()
    for tag in body.find_all("sup"):
        mark = tag.get_text().strip()
        if tag.find("a") or not mark or re.fullmatch(r"[\d*†‡\[\]().,\s]+", mark):
            tag.decompose()                          # a footnote mark
        else:
            tag.unwrap()                             # part of a word: XIX<sup>e</sup>, 1<sup>er</sup>, M<sup>me</sup>
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

        # The cover is already in the book: no need to look for it anywhere else.
        cover, cover_ext = b"", ""
        named = opf.find(".//{*}metadata/{*}meta[@name='cover']")
        images = [i for i in manifest if i.get("media-type", "") in ("image/jpeg", "image/png")]
        pick = next((i for i in images if "cover-image" in (i.get("properties") or "").split()), None) \
            or (by_id.get(named.get("content")) if named is not None else None) \
            or next((i for i in images if "cover" in (i.get("id", "") + i.get("href", "")).lower()), None)
        if pick and pick.get("media-type") in ("image/jpeg", "image/png"):
            try:
                cover = z.read(posixpath.normpath(posixpath.join(base, unquote(pick["href"]))))
                cover_ext = ".png" if pick["media-type"] == "image/png" else ".jpg"
            except KeyError:
                cover = b""
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
    return Book(meta("title", path.stem), meta("creator", ""), meta("language", "en").lower(), chapters, cover, cover_ext)


def html_to_epub(html: Path, out: Path, fallback_title: str) -> None:
    """Old MOBI files unpack to one long HTML page, with a list of chapter positions beside it.
    Cut the page at its chapters and page breaks, and wrap the pieces as an EPUB with its table of contents."""
    raw = html.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    opf = next(iter(html.parent.glob("*.opf")), None)
    meta = opf.read_text(encoding="utf-8", errors="replace") if opf else ""
    ncx = next(iter(html.parent.glob("*.ncx")), None)
    toc = ncx.read_text(encoding="utf-8", errors="replace") if ncx else ""

    def field(name: str, default: str = "") -> str:
        found = re.search(rf"<dc:{name}[^>]*>(.*?)</dc:{name}>", meta, re.S | re.I)
        return re.sub(r"\s+", " ", found.group(1)).strip() if found else default

    def position(anchor: str) -> int:
        """Where the block holding this anchor starts, or -1."""
        found = re.search(rf'<a\s[^>]*id="{re.escape(anchor)}"', text)
        if not found:
            return -1
        start = found.start()
        window = max(0, start - 600)
        opened = None
        for opened in re.finditer(r"<(?:p|div|h[1-6]|blockquote)\b[^>]*>", text[window:start]):
            pass
        if opened:                                   # the anchor sits at the start of a block: cut before that block
            before = text[window + opened.start():start]
            if not re.search(r"</(?:p|div|h[1-6]|blockquote)>", before) and not re.search(r"\w", re.sub(r"<[^>]+>", " ", before)):
                return window + opened.start()
        return start

    body = re.search(r"<body[^>]*>", text, re.I)
    begin = body.end() if body else 0
    end = text.lower().rfind("</body>")
    end = end if end > begin else len(text)
    titles: dict[int, str] = {}                      # cut position -> chapter title
    for label, anchor in re.findall(r"<navLabel>\s*<text>(.*?)</text>\s*</navLabel>\s*<content src=\"[^\"#]*#([^\"]+)\"", toc, re.S):
        at = position(anchor)
        if at >= begin:
            titles.setdefault(at, re.sub(r"\s+", " ", label).strip())
    contents = re.search(r'<reference[^>]*type="toc"[^>]*href="[^"#]*#([^"]+)"', meta)
    contents_at = position(contents.group(1)) if contents else -1
    cuts = sorted({begin, *titles, *(m.end() for m in re.finditer(r"<mbp:pagebreak\s*/?>", text, re.I) if begin <= m.end() < end)})
    pieces = []                                      # (title or "", html)
    for start, stop in zip(cuts, cuts[1:] + [end]):
        part = re.sub(r"<mbp:pagebreak\s*/?>", "", text[start:stop], flags=re.I)
        if start <= contents_at < stop and contents_at >= 0:
            continue                                 # the book's own table of contents page: not to be read aloud
        if re.search(r"\w", re.sub(r"<[^>]+>", " ", part)):
            pieces.append((titles.get(start, ""), part))
    names = [f"part{i:04d}.xhtml" for i in range(1, len(pieces) + 1)]
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;")
    cover = re.search(r'<meta name="cover" content="([^"]+)"', meta)
    cover = re.search(rf'<item[^>]*id="{re.escape(cover.group(1))}"[^>]*href="([^"]+)"', meta) if cover else None
    cover_file = html.parent / cover.group(1) if cover else None
    cover_item = ""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="book.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
        if cover_file and cover_file.is_file() and cover_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
            kind = "image/png" if cover_file.suffix.lower() == ".png" else "image/jpeg"
            z.write(cover_file, "cover" + cover_file.suffix.lower())
            cover_item = f'<item id="cover" href="cover{cover_file.suffix.lower()}" media-type="{kind}" properties="cover-image"/>'
        z.writestr("book.opf", '<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   f'<dc:title>{esc(field("title", fallback_title))}</dc:title><dc:creator>{esc(field("creator"))}</dc:creator>'
                   f'<dc:language>{esc(field("language", "en"))}</dc:language></metadata><manifest>'
                   '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>' + cover_item
                   + "".join(f'<item id="p{i}" href="{n}" media-type="application/xhtml+xml"/>' for i, n in enumerate(names))
                   + '</manifest><spine toc="ncx">' + "".join(f'<itemref idref="p{i}"/>' for i in range(len(names))) + "</spine></package>")
        z.writestr("toc.ncx", '<?xml version="1.0" encoding="utf-8"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><navMap>'
                   + "".join(f'<navPoint id="n{i}" playOrder="{i + 1}"><navLabel><text>{esc(title)}</text></navLabel><content src="{name}"/></navPoint>'
                             for i, (name, (title, _)) in enumerate(zip(names, pieces)) if title) + "</navMap></ncx>")
        for name, (_, part) in zip(names, pieces):
            # a large first letter followed by a space ("L orsque") would be read as a separate word
            part = re.sub(r"(<font[^>]*>\s*\w)(?:\xa0|&nbsp;|&#160;|\s)+(</font>)(?=\w)", r"\1\2", part)
            z.writestr(name, f"<html><head><title></title></head><body>{part}</body></html>")


def kindle_to_epub(path: Path, out: Path) -> None:
    """AZW3 and MOBI without Calibre, with the KindleUnpack code of the 'mobi' package."""
    try:
        import mobi
        from loguru import logger
        logger.disable("mobi")                       # it is talkative
    except ImportError:
        sys.exit(f"{path.suffix} files need Calibre to be converted first.\nInstall it with:  brew install --cask calibre")
    folder = ""
    try:
        folder, found = mobi.extract(str(path))
        found = Path(found)
        if found.suffix == ".epub":
            shutil.copy2(found, out)
        elif found.suffix == ".html":
            html_to_epub(found, out, path.stem)
        else:
            sys.exit(f"{path.name} holds pages as images or a PDF, not text that can be read aloud.")
    except SystemExit:
        raise
    except Exception as e:
        drm = re.search(r"drm|encrypt", str(e), re.I)
        sys.exit(f"{path.name} is DRM-protected and cannot be read." if drm else
                 f"{path.name} could not be opened ({type(e).__name__}: {str(e)[:120]}). Converting it to EPUB with Calibre may work.")
    finally:
        if folder:
            shutil.rmtree(folder, ignore_errors=True)


def to_epub(path: Path, workdir: Path) -> Path:
    """EPUB is used as is. Kindle formats are converted: with Calibre when it is installed, without it otherwise."""
    if path.suffix.lower() == ".epub":
        return path
    out = workdir / (path.stem + ".epub")
    tool = shutil.which("ebook-convert") or "/Applications/calibre.app/Contents/MacOS/ebook-convert"
    if not Path(tool).exists() or os.environ.get("BOOK2AUDIO_NO_CALIBRE"):
        print(f"Reading {path.name}...")
        kindle_to_epub(path, out)
        return out
    print(f"Converting {path.name} to EPUB with Calibre...")
    result = subprocess.run([tool, str(path), str(out)], capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        hint = " The file is DRM-protected: Calibre cannot open it." if "DRM" in result.stdout + result.stderr else ""
        sys.exit(f"Calibre could not convert this file.{hint}\n{result.stderr[-500:]}")
    return out


# --------------------------------------------------------------------------
# 2. Cut the text into pieces the model reads well
# --------------------------------------------------------------------------
NOT_A_SENTENCE_END = re.compile(r"(?:^|[\s(«“\"'-])(?:[A-ZÀ-Ý]|MM|Mr|Mrs|Ms|Dr|Pr|St|Ste|Mgr|Jr|Sr|vs|cf|vol|chap|art|fig|env|av|apr)\.$")


# ---- Reading rules: turn what is written for the eye into what should be said ----
_UNITS = ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze",
          "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf"]
_TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante", 80: "quatre-vingt"}


def fr_number(n: int) -> str:
    """A whole number from 0 to 9999 in French words."""
    if n < 20:
        return _UNITS[n]
    if n < 100:
        ten, unit = (60, n - 60) if 70 <= n < 80 else (80, n - 80) if n >= 80 else (n // 10 * 10, n % 10)
        if unit == 0:
            return "quatre-vingts" if ten == 80 else _TENS[ten]
        if unit in (1, 11) and ten != 80:
            return f"{_TENS[ten]} et {_UNITS[unit]}"
        return f"{_TENS[ten]}-{_UNITS[unit]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "cent" if hundreds == 1 else f"{_UNITS[hundreds]} cent"
        return head + ("s" if hundreds > 1 and rest == 0 else "") + (f" {fr_number(rest)}" if rest else "")
    thousands, rest = divmod(n, 1000)
    head = "mille" if thousands == 1 else f"{fr_number(thousands)} mille"
    return head + (f" {fr_number(rest)}" if rest else "")


def fr_ordinal(n: int, feminine: bool = False) -> str:
    if n == 1:
        return "première" if feminine else "premier"
    word = fr_number(n)
    if word.endswith("s") and n in (80,) or word.endswith("cents"):
        word = word[:-1]
    if word.endswith("e"):
        word = word[:-1]
    elif word.endswith("cinq"):
        word += "u"
    elif word.endswith("neuf"):
        word = word[:-1] + "v"
    return word + "ième"


def roman(text: str) -> int | None:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not text or not re.fullmatch(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})", text):
        return None
    total = 0
    for a, b in zip(text, text[1:] + " "):
        total += -values[a] if b != " " and values[a] < values[b] else values[a]
    return total


_FR_WORDS = [(r"\bMM\.", "Messieurs"), (r"\bMmes\b\.?", "Mesdames"), (r"\bMme\b\.?", "Madame"), (r"\bMlles\b\.?", "Mesdemoiselles"),
             (r"\bMlle\b\.?", "Mademoiselle"), (r"\bM\.(?=\s+[A-ZÀ-Ý])", "Monsieur"), (r"\bDr\b\.?(?=\s+[A-ZÀ-Ý])", "Docteur"),
             (r"\bPr\b\.?(?=\s+[A-ZÀ-Ý])", "Professeur"), (r"\bMgr\b\.?", "Monseigneur"), (r"\bSte\b\.?(?=[\s-]+[A-ZÀ-Ý])", "Sainte"),
             (r"\bSt\b\.?(?=[\s-]+[A-ZÀ-Ý])", "Saint"), (r"\b[Nn]°\s*", "numéro "), (r"\bcf\.", "voir"), (r"\bp\.\s*(?=\d)", "page "),
             (r"\bav\. J\.-C\.", "avant Jésus-Christ"), (r"\bapr\. J\.-C\.", "après Jésus-Christ"), (r"\bc\.-à-d\.", "c'est-à-dire")]
_EN_WORDS = [(r"\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|vs)\.", r"\1"), (r"\be\.g\.,?", "for example,"), (r"\bi\.e\.,?", "that is,")]
_PARTS = r"chapitre|livre|partie|tome|acte|scène|volume|chant|chapter|book|part|act|scene"


def prepare_text(text: str, language: str = "fr") -> str:
    """Rewrite nothing: only spell out what a narrator would say differently from what is printed."""
    french, english = language.startswith("fr"), language.startswith("en")
    text = re.sub(r"(?<=[\w»”\"'.,;:!?])\s?\[\d+\]", "", text)          # [12] footnote calls
    text = re.sub(r"(?<=[^\W\d])[¹²³⁴⁵⁶⁷⁸⁹⁰]+|(?<=\w)\*+", "", text)       # word¹, word*
    text = text.replace("...", "…")
    if len(text) < 120 and re.search(r"[A-ZÀ-Ý]{2}", text) and not re.search(r"[a-zà-ÿ]", text):
        # A line in capitals is a heading, not an acronym to spell out. Roman numerals keep their capitals.
        text = " ".join(w if roman(w.strip(".,:;")) and len(w.strip(".,:;")) > 1 or w in ("I", "V", "X") else w.lower()
                        for w in text.split())
        text = text[:1].upper() + text[1:]

    def part_number(m):
        n = roman(m.group(2))
        if n is None:
            return m.group(0)
        if not french:
            return f"{m.group(1)} {n}"
        return f"{m.group(1)} {'premier' if n == 1 else fr_number(n)}"
    if french:
        text = re.sub(rf"\b({_PARTS})\s+([IVXL]+)\b", part_number, text, flags=re.I)
    elif english:                                    # never the pronoun: "the book I read"
        text = re.sub(r"\b(Chapter|Book|Part|Act|Scene|Volume)\s+([IVXL]{2,}|V|X)\b", part_number, text)

    if french:
        def century(m):
            n = roman(m.group(1))
            return f"{fr_ordinal(n)} siècle" if n else m.group(0)
        text = re.sub(r"\b([IVX]+)\s?(?:e|ème|ᵉ|er)?\s+siècle", century, text)
        text = re.sub(r"\bIer\b", "premier", text)
        text = re.sub(r"\bIre\b", "première", text)

        def regnal(m):
            n = roman(m.group(2))
            return f"{m.group(1)} {fr_number(n)}" if n else m.group(0)
        text = re.sub(r"\b([A-ZÀ-Ý][a-zà-ÿ]+(?:-[A-ZÀ-Ý][a-zà-ÿ]+)?)\s+((?=[IVX]{2,})(?:X{0,3})(?:IX|IV|V?I{0,3}))\b", regnal, text)
        text = re.sub(r"\b(\d{1,2})\s?h\s?(\d{2})\b", r"\1 heures \2", text)          # 7 h 12
        text = re.sub(r"\b(\d{1,2})\s?h\b(?!')", r"\1 heures", text)                    # 7 h
        text = re.sub(r"\b1 heures\b", "1 heure", text)
        text = re.sub(r"\b1(?:er|ᵉʳ)\b", "premier", text)
        text = re.sub(r"\b1(?:re|ère)\b", "première", text)
        text = re.sub(r"\b(\d{1,3})(?:e|ème|ᵉ)\b", lambda m: fr_ordinal(int(m.group(1))), text)
        text = re.sub(r"\betc\.(?=\s+[A-ZÀ-Ý«—–]|\s*$)", "et cetera.", text)
        text = re.sub(r"\betc\.", "et cetera", text)
        for pattern, word in _FR_WORDS:
            text = re.sub(pattern, word, text)
    elif english:
        text = re.sub(r"\betc\.(?=\s+[A-Z“\"]|\s*$)", "et cetera.", text)
        text = re.sub(r"\betc\.", "et cetera", text)
        for pattern, word in _EN_WORDS:
            text = re.sub(pattern, word, text)

    text = re.sub(r"^\s*[—–-]\s+", "", text)                               # a line of dialogue: the dash is not said
    text = re.sub(r"\s+[—–]\s+|\s+-\s+", ", ", text)                       # a dash inside a sentence is a breath
    text = re.sub(r",\s*([,.;:!?…])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def split_paragraph(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Cut on sentence ends first, then commas, so the voice never stops mid-phrase."""
    sentences = re.findall(r".+?(?:[.!?…]+[\"'”’»)\]]*(?=\s|$)|$)", text, flags=re.S) or [text]
    merged: list[str] = []                           # "J. R. R. Tolkien", "M. Dupont": not sentence ends
    for part in sentences:
        if merged and NOT_A_SENTENCE_END.search(merged[-1].rstrip()):
            merged[-1] += part
        else:
            merged.append(part)
    sentences = merged
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

    def voices(self, done: int, total: int, label: str, stem: str, error: str = "") -> None:
        """Called before each voice sample, and when one could not be made."""
        if error:
            print(f"   {label}: {error}")


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
        self.model_id = MODEL_ALIASES.get(model_id.lower(), model_id)
        self.speed, self.lang_code = speed, lang_code
        self.kwargs: dict = {}
        self.reference: Path | None = None           # a recording of the voice to imitate (Chatterbox)
        self._conds = None
        if "kokoro" in self.model_id.lower() and language.split("-")[0] not in KOKORO:
            REPORT.log(f"Warning: Kokoro has no '{language}' voice, reading with an English voice.")
        self.set_voice(voice, language)
        try:
            from importlib.metadata import version
            library = f", mlx-audio {version('mlx-audio')}"
        except Exception:
            library = ""
        REPORT.log(f"Loading {self.model_id} ({', '.join(f'{k}={v}' for k, v in self.kwargs.items())}{library})")
        self.model = load_local_first(self.model_id, load_model)
        self.sample_rate = int(getattr(self.model, "sample_rate", 0) or 24000)

    def set_voice(self, voice: str | None, language: str) -> None:
        """Change the voice. The model stays loaded: only what is passed to it changes."""
        short, name = language.split("-")[0], self.model_id.lower()
        if "kokoro" in name:
            code, default_voice = KOKORO.get(language) or KOKORO.get(short) or KOKORO["en"]
            voice = voice or default_voice
            if re.fullmatch(r"[a-z][fm]_\w+", voice):      # a Kokoro voice carries its language: bf_emma is British
                code = voice[0]
            self.kwargs = {"voice": voice, "lang_code": self.lang_code or code, "speed": self.speed}
        elif "voxtral" in name:
            # Voxtral picks the language from the voice: fr_female, fr_male, casual_female...
            self.kwargs = {"voice": voice or VOXTRAL.get(short, "casual_female")}
            if self.lang_code:
                self.kwargs["lang_code"] = self.lang_code
        elif "chatterbox" in name:
            # Chatterbox has no voice of its own: it imitates the one in a short recording.
            path = my_voice_path(voice or "")
            if not path:
                raise RuntimeError("This model reads with a voice you give it: record or choose one first.")
            if path != self.reference:
                self.reference, self._conds = path, None
            self.kwargs = {"lang_code": self.lang_code or short, "verbose": False}
            self.key = f"{self.model_id}|{short}|{path.stem}|{path.stat().st_size}"
            return
        else:
            self.kwargs = {"lang_code": self.lang_code or LANG_NAMES.get(short, "English")}
            if voice:
                self.kwargs["voice"] = voice
            if self.speed != 1.0:
                self.kwargs["speed"] = self.speed
        self.reference, self._conds = None, None
        self.key = f"{self.model_id}|{sorted(self.kwargs.items())}"

    def say(self, text: str):
        import numpy as np

        parts = []
        extra = {}
        if self.reference is not None:
            if hasattr(self.model, "prepare_conditionals"):
                if self._conds is None:              # the recording is analysed once, not for every passage
                    self._conds = self.model.prepare_conditionals(str(self.reference), 24000)
                extra = {"conds": self._conds}
            else:                                    # another version of the library: hand it the recording each time
                extra = {"ref_audio": str(self.reference)}
        for result in self.model.generate(text=text, **self.kwargs, **extra):
            parts.append(np.array(result.audio, dtype=np.float32).reshape(-1))
            self.sample_rate = int(getattr(result, "sample_rate", 0) or self.sample_rate)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def build_script(chapter: Chapter, read_title: bool, language: str = "fr", plain: bool = False,
                 max_chars: int = MAX_CHARS) -> list[tuple[str, float]]:
    """The chapter as a list of (passage to read, silence after it)."""
    say = (lambda t: t) if plain else (lambda t: prepare_text(t, language))
    script: list[tuple[str, float]] = []
    if read_title:
        script.append((say(chapter.title).rstrip(".") + ".", PAUSE_TITLE))
    for paragraph in chapter.paragraphs:
        if not re.search(r"\w", paragraph):          # "* * *": a scene break is a longer silence, not words
            if script:
                script[-1] = (script[-1][0], max(script[-1][1], PAUSE_SCENE))
            continue
        chunks = [c for c in split_paragraph(say(paragraph), max_chars) if re.search(r"\w", c)]
        for i, chunk in enumerate(chunks):
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


def build_m4b(files: list[tuple[str, Path]], target: Path, book: Book, cover: Path | None = None) -> None:
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
        base = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(meta_file)]
        if cover and cover.exists():                 # the cover shows up in audiobook players
            with_cover = subprocess.run([*base, "-i", str(cover), "-map_metadata", "1", "-map", "0:a", "-map", "2:v", "-c", "copy",
                                         "-disposition:v:0", "attached_pic", "-f", "mp4", str(target)], capture_output=True)
            if with_cover.returncode == 0:
                return
        subprocess.run([*base, "-map_metadata", "1", "-map", "0:a", "-c", "copy", "-f", "mp4", str(target)], check=True)


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
name, voice = sys.argv[2], model.get_state_for_audio_prompt(sys.argv[2])
proto.write((json.dumps({"rate": int(model.sample_rate)}) + "\n").encode()); proto.flush()
for line in sys.stdin:
    job = json.loads(line)
    try:
        if job.get("voice") and job["voice"] != name:      # another voice of the same model: no reload
            voice = model.get_state_for_audio_prompt(job["voice"]); name = job["voice"]
        audio = model.generate_audio(voice, job["text"]).detach().cpu().numpy().astype("<f4").reshape(-1)
    except Exception as e:                                 # say what went wrong and stay ready for the next passage
        proto.write((json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n").encode()); proto.flush()
        continue
    proto.write((json.dumps({"samples": int(audio.size)}) + "\n").encode()); proto.write(audio.tobytes()); proto.flush()
"""


# Kyutai TTS 1.6B (French and English) through Kyutai's own MLX code. It needs an older
# version of MLX than the other models, one more reason to give it its own environment.
KYUTAI_VOICES = {"fr": "cml-tts/fr/10087_11650_000028-0002_enhanced.wav",
                 "en": "expresso/ex03-ex01_happy_001_channel1_334s.wav"}
KYUTAI_WORKER = r"""
import json, os, sys
proto = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
import mlx.core as mx
import numpy as np
import sentencepiece
from moshi_mlx import models
from moshi_mlx.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
from moshi_mlx.utils.loaders import hf_get

mx.random.seed(299792458)
raw_config = hf_get("config.json", DEFAULT_DSM_TTS_REPO)
with open(hf_get(raw_config), "r") as f:
    raw_config = json.load(f)
mimi_weights = hf_get(raw_config["mimi_name"], DEFAULT_DSM_TTS_REPO)
moshi_weights = hf_get(raw_config.get("moshi_name", "model.safetensors"), DEFAULT_DSM_TTS_REPO)
tokenizer = hf_get(raw_config["tokenizer_name"], DEFAULT_DSM_TTS_REPO)
lm_config = models.LmConfig.from_config_dict(raw_config)
lm_config.transformer.max_seq_len = lm_config.transformer.context
model = models.Lm(lm_config)
model.set_dtype(mx.bfloat16)
model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)
text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))
audio_tokenizer = models.mimi.Mimi(models.mimi_202407(lm_config.generated_codebooks))
audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)
tts = TTSModel(model, audio_tokenizer, text_tokenizer, voice_repo=DEFAULT_DSM_TTS_VOICE_REPO, temp=0.6, cfg_coef=1,
               max_padding=8, initial_padding=2, final_padding=2, padding_bonus=0, raw_config=raw_config)
cfg = None
no_text = no_prefix = True
if tts.valid_cfg_conditionings:
    cfg, tts.cfg_coef, no_text, no_prefix = tts.cfg_coef, 1.0, False, False
name = sys.argv[2]
attributes = [tts.make_condition_attributes([tts.get_voice_path(name)] if tts.multi_speaker else [], cfg)]
proto.write((json.dumps({"rate": int(tts.mimi.sample_rate)}) + "\n").encode()); proto.flush()
for line in sys.stdin:
    job = json.loads(line)
    if job.get("voice") and job["voice"] != name and tts.multi_speaker:
        try:
            attributes = [tts.make_condition_attributes([tts.get_voice_path(job["voice"])], cfg)]; name = job["voice"]
        except Exception as e:
            proto.write((json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n").encode()); proto.flush()
            continue
    chunks = []
    def on_frame(frame):
        if (frame == -1).any():
            return
        pcm = tts.mimi.decode_step(frame[:, :, None])
        chunks.append(np.array(mx.clip(pcm[0, 0], -1, 1)))
    tts.mimi.reset_state()
    tts.generate([tts.prepare_script([job["text"]])], attributes,
                 cfg_is_no_prefix=no_prefix, cfg_is_no_text=no_text, on_frame=on_frame)
    audio = (np.concatenate(chunks, -1) if chunks else np.zeros(0)).astype("<f4").reshape(-1)
    proto.write((json.dumps({"samples": int(audio.size)}) + "\n").encode()); proto.write(audio.tobytes()); proto.flush()
"""


class WorkerSynth:
    """A model that lives in its own process and environment. Text goes in, raw audio comes back."""

    def __init__(self, label: str, key: str, worker: str, packages: list[str], argv: list[str]):
        uv = shutil.which("uv")
        if not uv:
            sys.exit(f"{label} needs uv to install itself (https://docs.astral.sh/uv/).")
        self.label, self._key, self.default_voice = label, key, argv[-1]
        self.set_voice(None)
        self._dir = tempfile.mkdtemp(prefix="book2audio-worker-")
        script = Path(self._dir) / "worker.py"
        script.write_text(worker)
        REPORT.log(f"Loading {label} ({', '.join(argv)}). The first use installs it and downloads the model.")
        with_args = [x for name in packages for x in ("--with", name)]
        self.proc = subprocess.Popen([uv, "run", "--no-project", "--python", "3.12", *with_args,
                                      "python", "-u", str(script), *argv], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.sample_rate = int(self._header()["rate"])

    def set_voice(self, voice: str | None, language: str = "") -> None:
        self.voice = voice or self.default_voice
        self.key = self._key.format(voice=self.voice)

    def _header(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.label} stopped. The reason is in the Terminal window, just above.")
        header = json.loads(line)
        if header.get("error"):
            raise RuntimeError(header["error"])
        return header

    def say(self, text: str):
        import numpy as np

        self.proc.stdin.write((json.dumps({"text": text, "voice": self.voice}) + "\n").encode())
        self.proc.stdin.flush()
        size = 4 * int(self._header()["samples"])
        data = b""
        while len(data) < size:
            part = self.proc.stdout.read(size - len(data))
            if not part:
                raise RuntimeError(f"{self.label} stopped in the middle of a passage.")
            data += part
        return np.frombuffer(data, dtype="<f4").copy()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        shutil.rmtree(self._dir, ignore_errors=True)


def worker_synth(model: str, language: str, voice: str | None) -> WorkerSynth:
    short = language.split("-")[0]
    if model == "pocket":
        name, default_voice = POCKET.get(short) or POCKET["en"]
        voice = voice or default_voice
        return WorkerSynth("Pocket TTS", "pocket|" + name + "|{voice}", POCKET_WORKER, ["pocket-tts"], [name, voice])
    voice = voice or KYUTAI_VOICES.get(short) or KYUTAI_VOICES["en"]
    return WorkerSynth("Kyutai TTS", "kyutai|{voice}", KYUTAI_WORKER, ["moshi_mlx==0.2.12", "numpy", "sentencepiece"],
                       [short, voice])


WORKER_MODELS = ("pocket", "kyutai")

_synth_cache: dict = {}


def default_model(language: str) -> str:
    """The best model this machine can run for this language (see recommend)."""
    return recommend(language)["recommended"]


def make_synth(args, language: str):
    """The model stays loaded between two jobs, whatever the voice. Only one is kept in memory."""
    model = args.model or default_model(language)
    key = (model, language.split("-")[0], args.speed, args.lang_code)    # the voice is not part of it: it can change
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
        _synth_cache[key] = (worker_synth(model.lower(), language, args.voice) if model.lower() in WORKER_MODELS
                             else Synth(model, language, args.voice, args.speed, args.lang_code))
    _synth_cache[key].set_voice(args.voice, language)
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
    sample = sample_of(book, chapter_number)
    target = out_dir / f"preview.{fmt}"
    pcm = out_dir / ".tmp-preview.pcm"
    script = build_script(sample, not args.no_titles, language, args.plain)
    synth = make_synth(args, language)
    _, _, rate = render_chapter(synth, script, pcm, Tracker(script_chars(script)), resume=False)
    encode(pcm, rate, target, {"title": "Preview"})
    REPORT.log(f"Preview written to {target}")
    return target


# The voices offered for each model and language: (id for the model, name shown, short note).
def _kokoro(*ids: str) -> list[tuple[str, str, str]]:
    accent = {"a": ", American", "b": ", British"}
    return [(v, v[3:].capitalize(), ("Woman" if v[1] == "f" else "Man") + accent.get(v[0], "")) for v in ids]


def _voxtral(*ids: str) -> list[tuple[str, str, str]]:
    return [(v, "Woman" if v.endswith("female") else "Man", "" if re.match(r"[a-z]{2}_", v) else v.split("_")[0].capitalize())
            for v in ids]


# Pocket TTS ships 27 voices. Six are offered per language: the one recorded in that language, then five
# of the original English voices, which keep a foreign accent outside English (heard in French).
POCKET_ENGLISH = {"alba": "Woman", "cosette": "Woman", "fantine": "Woman", "marius": "Man", "javert": "Man", "jean": "Man"}
POCKET_NATIVE = {"estelle": "Woman", "lola": "Woman", "juergen": "Man", "giovanni": "Man", "rafael": "Man"}


def _pocket(code: str, first: str) -> list[tuple[str, str, str]]:
    if code == "en":
        return [(v, v.capitalize(), who + (", default" if v == first else "")) for v, who in POCKET_ENGLISH.items()]
    others = [(v, v.capitalize(), f"{who}, foreign accent") for v, who in list(POCKET_ENGLISH.items())[:6] if v != "jean"]
    return [(first, first.capitalize(), f"{POCKET_NATIVE[first]}, {LANG_NAMES_UI[code]}, default")] + others


LANG_NAMES_UI = {"fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese"}
_VOX = {**{code: _voxtral(f"{code}_female", f"{code}_male") for code in ("fr", "es", "de", "it", "pt", "nl", "hi")},
        "en": _voxtral("neutral_female", "neutral_male", "casual_female", "casual_male", "cheerful_female"),
        "ar": _voxtral("ar_male")}
VOICE_CATALOG = {
    "voxtral-4bit": _VOX,
    "voxtral": _VOX,
    "kyutai": {code: [(voice, "Kyutai voice", "Default")] for code, voice in KYUTAI_VOICES.items()},
    "pocket": {code: _pocket(code, default) for code, (_, default) in POCKET.items()},
    "kokoro": {
        "fr": _kokoro("ff_siwis"),
        "en": _kokoro("af_heart", "af_bella", "af_nicole", "af_sarah", "af_aoede", "af_kore", "af_nova", "af_sky",
                      "am_michael", "am_adam", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam", "am_onyx",
                      "bf_emma", "bf_isabella", "bf_alice", "bf_lily", "bm_george", "bm_fable", "bm_lewis", "bm_daniel"),
        "es": _kokoro("ef_dora", "em_alex"),
        "it": _kokoro("if_sara", "im_nicola"),
        "pt": _kokoro("pf_dora", "pm_alex"),
        "hi": _kokoro("hf_alpha", "hf_beta", "hm_omega", "hm_psi"),
        "ja": _kokoro("jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"),
        "zh": _kokoro("zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"),
    },
}

# What every voice says in its sample: the same two sentences, so that only the voice differs.
SAMPLE_TEXT = {
    "fr": ("Bonjour, je suis l'une des voix de Book to Audio.",
           "Je lis vos livres numériques à voix haute, chapitre après chapitre, pour que vous puissiez les écouter où vous voulez."),
    "en": ("Hello, I am one of the voices of Book to Audio.",
           "I read your ebooks aloud, chapter by chapter, so you can listen to them wherever you like."),
    "es": ("Hola, soy una de las voces de Book to Audio.",
           "Leo tus libros electrónicos en voz alta, capítulo a capítulo, para que puedas escucharlos donde quieras."),
    "de": ("Hallo, ich bin eine der Stimmen von Book to Audio.",
           "Ich lese Ihre E-Books laut vor, Kapitel für Kapitel, damit Sie sie hören können, wo immer Sie möchten."),
    "it": ("Ciao, sono una delle voci di Book to Audio.",
           "Leggo i tuoi ebook ad alta voce, capitolo dopo capitolo, così puoi ascoltarli dove vuoi."),
    "pt": ("Olá, sou uma das vozes do Book to Audio.",
           "Leio os seus livros digitais em voz alta, capítulo a capítulo, para que os possa ouvir onde quiser."),
    "nl": ("Hallo, ik ben een van de stemmen van Book to Audio.",
           "Ik lees je e-books hardop voor, hoofdstuk voor hoofdstuk, zodat je ernaar kunt luisteren waar je maar wilt."),
    "hi": ("नमस्ते, यह Book to Audio की आवाज़ों में से एक है।",
           "यह ऐप आपकी ई-किताबें अध्याय दर अध्याय पढ़कर सुनाता है, ताकि आप उन्हें कहीं भी सुन सकें।"),
    "ar": ("مرحباً، أنا أحد أصوات Book to Audio.",
           "أقرأ كتبك الإلكترونية بصوت عالٍ، فصلاً بعد فصل، لتستمع إليها أينما شئت."),
    "ja": ("こんにちは、Book to Audio の声のひとつです。",
           "電子書籍を一章ずつ読み上げますので、お好きな場所でお聴きいただけます。"),
    "zh": ("你好，我是 Book to Audio 的声音之一。",
           "我会一章一章地朗读你的电子书，让你随时随地都能收听。"),
}
VOICE_DIR = Path.home() / "Audiobooks" / ".voices"       # samples are made once on this Mac, then kept


MY_VOICES = VOICE_DIR / "mine"                           # recordings given by the user, never sent anywhere


def my_voices() -> list[dict]:
    rows = []
    for path in sorted(MY_VOICES.glob("*.wav")) if MY_VOICES.is_dir() else []:
        try:
            name = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["name"]
        except Exception:
            name = path.stem
        rows.append({"id": path.stem, "name": name})
    return rows


def my_voice_path(voice_id: str) -> Path | None:
    path = MY_VOICES / f"{voice_id}.wav"
    return path if re.fullmatch(r"[a-z0-9-]+", voice_id or "") and path.is_file() else None


def save_my_voice(name: str, data: bytes, ext: str) -> dict:
    """Keep a recording as a voice: mono, 24 kHz, 30 seconds at most. The model listens to the first 10."""
    import unicodedata
    name = " ".join(name.split())[:40] or "My voice"
    base = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()).strip("-") or "voice"
    slug, n = base, 2
    while (MY_VOICES / f"{slug}.wav").exists():
        slug, n = f"{base}-{n}", n + 1
    MY_VOICES.mkdir(parents=True, exist_ok=True)
    source = MY_VOICES / f".tmp-{slug}.{re.sub(r'[^a-z0-9]', '', ext.lower()) or 'bin'}"
    target = MY_VOICES / f"{slug}.wav"
    source.write_bytes(data)
    try:
        done = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "24000",
                               "-t", "30", str(target)], capture_output=True, text=True)
        if done.returncode != 0 or not target.exists():
            raise ValueError("This file is not a recording that can be read.")
        seconds = audio_seconds(target)
        if seconds < 5:
            raise ValueError(f"This recording lasts {seconds:.0f} seconds. 10 to 20 seconds of speech work best.")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        source.unlink(missing_ok=True)
    target.with_suffix(".json").write_text(json.dumps({"name": name, "seconds": round(seconds, 1)}, ensure_ascii=False), encoding="utf-8")
    return {"id": slug, "name": name}


def delete_my_voice(voice_id: str) -> None:
    path = my_voice_path(voice_id)
    if not path:
        return
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    for sample in VOICE_DIR.glob(f"clone--*--{voice_id}.*"):
        sample.unlink(missing_ok=True)


def voice_stem(model: str, language: str, voice: str) -> str:
    """The file name of a voice sample, without its extension."""
    name = re.sub(r"[^A-Za-z0-9]+", "-", voice.rsplit("/", 1)[-1].removesuffix(".wav")).strip("-")
    return f"{model}--{language.split('-')[0]}--{name}"


def voice_samples() -> dict[str, Path]:
    """The samples already made, by stem."""
    if not VOICE_DIR.is_dir():
        return {}
    return {p.stem: p for p in VOICE_DIR.iterdir() if p.suffix in (".m4a", ".wav") and not p.name.startswith(".")}


# Samples already made for the main models are published with the project: the page plays them
# straight away, without loading any model. Each one is fetched the first time it is played.
SAMPLES_URL = os.environ.get("BOOK2AUDIO_SAMPLES_URL", "https://raw.githubusercontent.com/alexandre-darmon/book2audio/main/voices/")
_published: dict[str, str] = {}                          # stem -> file name


def load_published() -> None:
    """The list of published samples. Without a network there are simply none."""
    import urllib.request
    try:
        names = json.loads(urllib.request.urlopen(SAMPLES_URL + "index.json", timeout=6).read())
        _published.update({Path(n).stem: Path(n).name for n in names if re.fullmatch(r"[\w.-]+\.(m4a|wav)", str(n))})
    except Exception:
        pass


def fetch_published(stem: str) -> Path | None:
    import urllib.request
    name = _published.get(stem)
    if not name:
        return None
    try:
        data = urllib.request.urlopen(SAMPLES_URL + name, timeout=15).read()
    except Exception:
        _published.pop(stem, None)                   # not reachable: the page will offer to create it on this Mac
        return None
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = VOICE_DIR / (".tmp-" + name)
    tmp.write_bytes(data)
    tmp.replace(VOICE_DIR / name)
    return VOICE_DIR / name


def run_all_voice_samples(args) -> None:
    """Every voice of every model that fits this Mac, in French and English (or --lang, --model).
    With -o, the samples are also copied to that folder with their list: this is what gets published."""
    languages = [args.lang.split("-")[0].lower()] if args.lang else ["fr", "en"]
    models = [args.model] if args.model else [m for m in MODELS if m != "kyutai" and model_fit(m) != "no"]
    for model in models:
        for language in languages:
            rows = VOICE_CATALOG.get(model, {}).get(language, [])
            todo = [v for v, _, _ in rows if voice_stem(model, language, v) not in voice_samples()]
            if todo:
                REPORT.log(f"{MODEL_LABELS.get(model, model)}, {LANGUAGES.get(language, language)}: {len(todo)} sample(s) to make")
                run_voice_samples(model, language, todo, args)
    done = voice_samples()
    REPORT.log(f"{len(done)} samples in {VOICE_DIR}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for path in done.values():
            shutil.copy2(path, args.out / path.name)
        (args.out / "index.json").write_text(json.dumps(sorted(p.name for p in done.values()), indent=1) + "\n")
        REPORT.log(f"Copied to {args.out}, with index.json")


def run_voice_samples(model: str, language: str, voices: list[str], args) -> None:
    """One short sample per voice. The model is loaded once for all of them."""
    short = language.split("-")[0]
    first, second = SAMPLE_TEXT.get(short) or SAMPLE_TEXT["en"]
    script = [(first, PAUSE_PARAGRAPH), (second, 0.0)]
    fmt = "wav" if pick_format("m4a") == "wav" else "m4a"
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    names = {v: name for v, name, _ in VOICE_CATALOG.get(model, {}).get(short, [])}
    names.update({v["id"]: v["name"] for v in my_voices()} if model == "clone" else {})
    label = MODEL_LABELS.get(model, model)
    for k, voice in enumerate(voices):
        stem = voice_stem(model, short, voice)
        REPORT.voices(k, len(voices), label, stem)
        try:
            synth = make_synth(argparse.Namespace(**{**vars(args), "model": model, "voice": voice}), short)
            pcm = VOICE_DIR / f".tmp-{stem}.pcm"
            _, _, rate = render_chapter(synth, script, pcm, None, resume=False)
            encode(pcm, rate, VOICE_DIR / f"{stem}.{fmt}", {"title": f"{names.get(voice, voice)} ({label})", "artist": "book2audio"})
        except (Stopped, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:                       # one voice that fails must not stop the others
            REPORT.voices(k, len(voices), f"{label}, {names.get(voice, voice)}", stem, f"{type(e).__name__}: {e}"[:160])
            worker = next(iter(_synth_cache.values()), None)
            if isinstance(worker, WorkerSynth):
                try:
                    worker.proc.wait(timeout=0.5)
                    worker.close()
                    _synth_cache.clear()             # its process died: a fresh one starts for the next voice
                except subprocess.TimeoutExpired:
                    pass                             # still alive: only this voice failed
    REPORT.voices(len(voices), len(voices), label, "")


# What each model needs while it reads: memory in GB (model plus working room), and whether it uses the
# Apple Silicon GPU. Memory figures are estimates, refined as people report their own machines.
MODELS = {
    "voxtral-4bit": {"label": "Voxtral 4-bit", "need": 3.5, "gpu": True, "langs": "fr en es de it pt nl ar hi"},
    "voxtral": {"label": "Voxtral", "need": 9.0, "gpu": True, "langs": "fr en es de it pt nl ar hi"},
    "kyutai": {"label": "Kyutai TTS", "need": 5.0, "gpu": True, "langs": "fr en"},
    "pocket": {"label": "Pocket TTS", "need": 1.5, "gpu": False, "langs": "fr en de es it pt"},
    "kokoro": {"label": "Kokoro", "need": 1.5, "gpu": True, "langs": "en fr es it pt hi ja zh"},
    # Never picked automatically: it needs a recording first. Its memory figure is an estimate.
    "clone": {"label": "Your own voice", "need": 4.0, "gpu": True, "langs": "fr en es de it pt nl hi ar ja zh"},
}
MODEL_LABELS = {k: v["label"] for k, v in MODELS.items()}
# Voxtral 4-bit comes first wherever it speaks the language: it is the model confirmed on a real Mac.
PREFERENCE = {"fr": ["voxtral-4bit", "voxtral", "pocket", "kokoro"], "en": ["voxtral-4bit", "kokoro", "pocket", "voxtral"]}
LANGUAGES = {"fr": "French", "en": "English", "es": "Spanish", "de": "German", "it": "Italian", "pt": "Portuguese",
             "nl": "Dutch", "hi": "Hindi", "ar": "Arabic", "ja": "Japanese", "zh": "Chinese"}
_machine: dict = {}


def machine_info() -> dict:
    """The chip and the memory of this computer."""
    if _machine:
        return _machine
    import platform

    chip, memory = platform.processor() or platform.machine(), 0.0
    try:
        if sys.platform == "darwin":
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or chip
            memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()) / 2**30
        else:
            memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (OSError, ValueError):
        pass
    _machine.update(chip=chip, memory_gb=round(memory), cores=os.cpu_count() or 1,
                    apple_silicon=sys.platform == "darwin" and platform.machine() == "arm64")
    return _machine


def model_fit(model: str, machine: dict | None = None) -> str:
    """'fits', 'tight' or 'no'. The rule keeps 20% of the memory free, plus 4 GB for macOS and other apps."""
    machine = machine or machine_info()
    spec = MODELS.get(model)
    if not spec or not machine.get("memory_gb"):
        return "fits"                                # unknown model or machine: do not block anything
    if spec["gpu"] and not machine.get("apple_silicon"):
        return "no"
    room = machine["memory_gb"] * 0.8
    return "fits" if spec["need"] <= room - 4 else "tight" if spec["need"] <= room - 2 else "no"


def recommend(language: str, machine: dict | None = None) -> dict:
    """Which models suit this machine for this language, and the one to pick."""
    machine = machine or machine_info()
    short = language.split("-")[0]
    speeds = load_speeds()
    rows = []
    for model, spec in MODELS.items():
        fit = model_fit(model, machine)
        rows.append({"model": model, "label": spec["label"], "fit": fit, "speaks": short in spec["langs"].split(),
                     "where": "GPU" if spec["gpu"] else "CPU", "need_gb": spec["need"], "speed": speeds.get(model)})
    order = PREFERENCE.get(short) or [m for m, v in MODELS.items() if short in v["langs"].split()] or PREFERENCE["en"]
    order = [m for m in order if m != "clone"]
    usable = {r["model"]: r for r in rows if r["speaks"]}
    best = next((m for m in order if m in usable and usable[m]["fit"] == "fits"), None) \
        or next((m for m in order if m in usable and usable[m]["fit"] == "tight"), None) or "pocket"
    return {"chip": machine["chip"], "memory_gb": machine["memory_gb"], "models": rows, "recommended": best}


def print_check(language: str) -> None:
    r = recommend(language)
    print(f"This computer: {r['chip']}, {r['memory_gb']} GB of memory")
    print(f"For books in '{language}':")
    words = {"fits": "fits", "tight": "tight, close other apps", "no": "does not fit"}
    for row in r["models"]:
        if row["speaks"]:
            speed = f"about {row['speed']}x real time" if row["speed"] else "speed not measured yet"
            star = "  <- recommended" if row["model"] == r["recommended"] else ""
            print(f"  {row['label']:14s} {row['where']}  needs ~{row['need_gb']} GB  {words[row['fit']]:24s} {speed}{star}")


def compare_list(language: str) -> list[str]:
    short = language.split("-")[0]
    wanted = ["voxtral", "voxtral-4bit", "kyutai", "pocket"] if short == "fr" else ["kokoro", "voxtral-4bit", "kyutai", "pocket"]
    return [m for m in wanted if short in MODELS[m]["langs"].split() and model_fit(m) != "no"]


def sample_of(book: Book, chapter_number: int) -> Chapter:
    """About one minute from the start of a chapter."""
    first = book.chapters[chapter_number - 1]
    sample, size = Chapter(first.title), 0
    for p in first.paragraphs:
        sample.paragraphs.append(p)
        size += len(p)
        if size > 900:
            break
    return sample


# A short text written for this purpose, free to use, that touches what a narrator must get right:
# titles, abbreviations, a century, a clock time, dialogue, a question, an exclamation.
DEMO = {
    "fr": Chapter("Chapitre III", [
        "Le train de 7 h 12 entra en gare avec onze minutes de retard. M. Lambert replia son journal, consulta sa montre, "
        "puis regarda Mme Roche, qui dormait encore contre la vitre.",
        "— Nous arrivons, dit-il doucement.",
        "Elle ouvrit les yeux — lentement, comme on remonte d'un puits — et sourit.",
        "— Déjà ? J'ai rêvé que nous étions revenus au XIXe siècle. Il y avait des fiacres, des becs de gaz, etc. "
        "Vous portiez un chapeau ridicule !",
        "— Je n'en doute pas.",
        "Dehors, la pluie avait cessé. Sur le quai n° 4, un enfant courait après un ballon rouge, et personne, "
        "absolument personne, ne semblait pressé.",
    ]),
    "en": Chapter("Chapter III", [
        "The 7:12 train pulled in eleven minutes late. Mr. Lambert folded his newspaper, checked his watch, "
        "and looked at Mrs. Roche, still asleep against the window.",
        "“We're here,” he said quietly.",
        "She opened her eyes - slowly, like someone climbing out of a well - and smiled.",
        "“Already? I dreamed we had gone back to the nineteenth century. There were cabs, gas lamps, etc. "
        "You were wearing a ridiculous hat!”",
        "“I don't doubt it.”",
        "Outside, the rain had stopped. On platform 4, a child was chasing a red ball, and nobody, "
        "absolutely nobody, seemed to be in a hurry.",
    ]),
}


def best_for(model: str) -> str:
    """The Macs a model suits, from the same memory rule as the recommendation."""
    sizes = [8, 16, 24, 32]
    fits = {gb: model_fit(model, {"memory_gb": gb, "apple_silicon": True}) for gb in sizes}
    ok = next((gb for gb in sizes if fits[gb] == "fits"), None)
    tight = next((gb for gb in sizes if fits[gb] == "tight"), None)
    if ok is None:
        return "more than 32 GB"
    text = f"{ok} GB and up"
    return f"{tight} GB if other apps are closed, {text}" if tight and tight < ok else text


def write_benchmark(rows: list[dict], language: str, out_dir: Path, prefix: str) -> Path:
    machine = machine_info()
    lines = [f"# Benchmark ({language})", "",
             f"Measured on: {machine['chip']}, {machine['memory_gb']} GB of memory, on {time.strftime('%Y-%m-%d')}.",
             "Same text for every model, about one minute of audio. Speed includes the model's warm-up.", "",
             "| Model | Listen | Speed here | Characters per second | Runs on | Memory needed | Suits a Mac with |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        spec = MODELS.get(row["model"], {})
        if row.get("file"):
            lines.append(f"| {row['label']} | [{row['file']}]({row['file']}) | {row['speed']}x real time | {row['chars_s']} | "
                         f"{'GPU' if spec.get('gpu') else 'CPU'} | about {spec.get('need', '?')} GB | {best_for(row['model'])} |")
        else:
            lines.append(f"| {row['label']} | failed here | | | | | |")
    lines += ["", "How to read it: 2x real time means one hour of audio takes half an hour to create.",
              "Memory figures are estimates. Quality is for your ears: listen to the files.", ""]
    target = out_dir / f"BENCHMARK-{prefix.split('-')[-1]}.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def run_demo(args) -> None:
    """Read the built-in sample (or --demo-text) with every model that fits, for the README."""
    language = (args.lang or "fr").lower()
    short = language.split("-")[0]
    if args.demo_text:
        blocks = [_clean(b) for b in re.split(r"\n\s*\n", Path(args.demo_text).read_text(encoding="utf-8")) if _clean(b)]
        sample = Chapter(blocks[0], blocks[1:]) if len(blocks) > 1 else Chapter("", blocks)
    else:
        sample = DEMO.get(short) or DEMO["en"]
    in_repo = (Path.cwd() / "book2audio.py").exists()
    out_dir = args.out or (Path.cwd() / "demo" if in_repo else Path.home() / "Audiobooks" / "demo")
    if not sample.title:
        args.no_titles = True
    rows = run_compare(sample, language, args, out_dir, [args.model] if args.model else None, prefix=f"demo-{short}")
    for row in rows:                                 # a small video of the waveform, the only audio GitHub plays inline
        if row["file"].endswith(".m4a"):
            audio = out_dir / row["file"]
            video = audio.with_suffix(".mp4")
            done = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(audio), "-filter_complex",
                                   "color=c=0x0a0a0c:s=1280x320:r=25[bg];[0:a]showwaves=s=1280x320:mode=cline:rate=25:colors=0x8C52EE[w];"
                                   "[bg][w]overlay=shortest=1,format=yuv420p[v]", "-map", "[v]", "-map", "0:a",
                                   "-c:v", "libx264", "-preset", "fast", "-c:a", "copy", "-movflags", "+faststart", str(video)],
                                  capture_output=True)
            if done.returncode != 0:
                video.unlink(missing_ok=True)
    report = write_benchmark(rows, language, out_dir, f"demo-{short}")
    print("\n" + report.read_text(encoding="utf-8"))
    print(f"Files and {report.name} are in {out_dir}")


def run_compare(sample: Chapter, language: str, args, out_dir: Path, models: list[str] | None = None,
                prefix: str = "compare") -> list[dict]:
    """Read the same one-minute passage with several models, and time each one on this machine."""
    fmt = pick_format(args.format)
    out_dir.mkdir(parents=True, exist_ok=True)
    script = build_script(sample, not args.no_titles, language, args.plain)
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
            pcm = out_dir / f".tmp-{prefix}-{model}.pcm"
            started = time.time()
            audio_s, _, rate = render_chapter(synth, script, pcm, Tracker(script_chars(script)), resume=False)
            row["compute_s"] = round(time.time() - started, 1)
            row["audio_s"] = round(audio_s, 1)
            row["speed"] = round(audio_s / max(row["compute_s"], 0.1), 2)
            save_speed(model, row["speed"])
            row["chars_s"] = round(script_chars(script) / max(row["compute_s"], 0.1), 1)
            target = out_dir / f"{prefix}-{model}.{fmt}"
            encode(pcm, rate, target, {"title": f"{row['label']}"})
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
    REPORT.log(f"Listen to the {prefix}-*.{fmt} files in {out_dir}")
    return results


def run_book(book: Book, selection: list[int], language: str, args, out_dir: Path) -> None:
    fmt = pick_format(args.format)
    out_dir.mkdir(parents=True, exist_ok=True)
    read_titles = not args.no_titles
    todo = [n for n in selection if not chapter_target(out_dir, n, book.chapters[n - 1], fmt).exists()]
    for n in selection:
        if n not in todo:
            REPORT.log(f"[{n}/{len(book.chapters)}] {book.chapters[n - 1].title}: already done, skipping")
    scripts = {n: build_script(book.chapters[n - 1], read_titles, language, args.plain) for n in todo}
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
        cover = None
        if book.cover:
            cover = out_dir / f"cover{book.cover_ext}"
            cover.write_bytes(book.cover)
        build_m4b(done, target, book, cover)
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
    ap.add_argument("book", type=Path, nargs="?", help="EPUB, AZW3 or MOBI file without DRM")
    ap.add_argument("--ui", action="store_true", help="open the simple web page instead of using the command line")
    ap.add_argument("-o", "--out", type=Path, help="output folder (default: ~/Audiobooks/<book title>)")
    ap.add_argument("--list", action="store_true", help="show the chapters and stop")
    ap.add_argument("--chapters", help='which chapters to read, e.g. "1-3,7" (numbers from --list)')
    ap.add_argument("--preview", action="store_true", help="read only about one minute, to judge the voice")
    ap.add_argument("--compare", action="store_true", help="read the same minute with each model and time them on this machine")
    ap.add_argument("--check", action="store_true", help="show this computer's chip and memory, and the models that suit it")
    ap.add_argument("--demo", action="store_true", help="read a built-in sample text with each model (no book needed)")
    ap.add_argument("--voice-samples", action="store_true",
                    help="make the short sample of every voice (French and English, or --lang; every model that fits, or --model). "
                         "With -o, also copy them to that folder")
    ap.add_argument("--demo-text", help="with --demo: a text file to read instead of the built-in sample")
    ap.add_argument("--lang", help="book language, e.g. en, en-gb, fr (default: read from the book)")
    ap.add_argument("--voice", help="voice name (default: a good voice for the language)")
    ap.add_argument("--speed", type=float, default=1.0, help="reading speed, 1.0 = normal")
    ap.add_argument("--model", help="kokoro, voxtral, voxtral-4bit, pocket, kyutai, or any mlx-audio TTS model id (default: the best one this machine can run)")
    ap.add_argument("--lang-code", help="advanced: language value passed as is to the model")
    ap.add_argument("--format", choices=["m4a", "mp3", "wav"], default="m4a")
    ap.add_argument("--no-m4b", action="store_true", help="do not build the single audiobook file (.m4b with chapter markers)")
    ap.add_argument("--iphone", action="store_true", help="copy the audiobook to iCloud Drive/Audiobooks, to open it on the iPhone")
    ap.add_argument("--min-chars", type=int, default=500, help="skip sections shorter than this (cover, copyright...)")
    ap.add_argument("--no-titles", action="store_true", help="do not read chapter titles aloud")
    ap.add_argument("--plain", action="store_true", help="read the text exactly as printed, without the reading rules (abbreviations, Roman numerals...)")
    args = ap.parse_args(argv)

    if args.check:
        print_check((args.lang or "fr").lower())
        return 0
    if args.demo:
        run_demo(args)
        return 0
    if args.voice_samples:
        run_all_voice_samples(args)
        return 0
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
        run_compare(sample_of(book, selection[0]), language, args, out_dir, [args.model] if args.model else None)
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/png" sizes="64x64" href="/favicon.png">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cstyle%3E.t%7Bfill:%23111113;stroke:%2326262B%7D.g%7Bstroke:%23F7F7F7%7D@media (prefers-color-scheme:light)%7B.t%7Bfill:%23F7F7F7;stroke:%23E3E3E3%7D.g%7Bstroke:%230A0A0A%7D%7D%3C/style%3E%3Crect class='t' x='.5' y='.5' width='63' height='63' rx='13'/%3E%3Cpath class='g' d='M13 30V34M22 26V38M42 26V38M51 30V34' fill='none' stroke-width='6' stroke-linecap='square'/%3E%3Crect x='28' y='20' width='8' height='24' fill='%238C52EE'/%3E%3C/svg%3E">
<style>
  :root { color-scheme: dark;       /* dark or light follows the Mac's own setting */
    --bg:#0a0a0c; --panel:#111113; --raise:#17171a; --line:#232327; --ink:#ededef; --soft:#8b8b93; --faint:#5c5c66;
    --accent:#8C52EE; --accent-ink:#b392f5; --accent-wash:rgba(140,82,238,.14); --tile:#111113; --tile-line:#26262B; --glyph:#F7F7F7;
    --danger:#f2777a; --ok:#58c98b; }
  @media (prefers-color-scheme: light) { :root { color-scheme: light;
    --bg:#ffffff; --panel:#fafafa; --raise:#f1f1f3; --line:#E3E3E3; --ink:#0A0A0A; --soft:#62626b; --faint:#9a9aa3;
    --accent:#8C52EE; --accent-ink:#7a3fe0; --accent-wash:rgba(140,82,238,.10); --tile:#F7F7F7; --tile-line:#E3E3E3; --glyph:#0A0A0A;
    --danger:#c62f33; --ok:#1f8f55; } }
  * { box-sizing:border-box; }
  html { -webkit-text-size-adjust:100%; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Helvetica Neue", sans-serif;
         -webkit-font-smoothing:antialiased; font-variant-numeric:tabular-nums; }
  .hide { display:none !important; }
  .soft { color:var(--soft); }
  a { color:var(--accent-ink); cursor:pointer; text-decoration:none; } a:hover { text-decoration:underline; }
  h2, h3 { margin:0; font-weight:650; letter-spacing:-0.015em; }
  h2 { font-size:20px; line-height:1.25; } h3 { font-size:14px; }

  /* Top bar */
  header { border-bottom:1px solid var(--line); position:sticky; top:0; background:color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter:blur(10px); z-index:5; }
  header > div { max-width:780px; margin:0 auto; padding:12px 24px; display:flex; align-items:center; gap:10px; }
  header svg { display:block; flex:none; }
  .word { font-weight:650; letter-spacing:-0.02em; font-size:15px; }
  #mac { color:var(--soft); font-size:13px; margin-left:6px; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .bar-actions { margin-left:auto; display:flex; gap:2px; flex:none; align-items:center; }
  #power { display:flex; align-items:center; gap:7px; font-size:13px; color:var(--soft); margin-right:8px; }
  #power i { width:8px; height:8px; border-radius:50%; background:var(--ok); box-shadow:0 0 0 3px color-mix(in srgb, var(--ok) 22%, transparent); }
  #power.off i { background:var(--danger); box-shadow:0 0 0 3px color-mix(in srgb, var(--danger) 22%, transparent); }
  button.icon { width:32px; padding:0; display:grid; place-items:center; } button.icon svg { display:block; }

  /* Controls */
  button { font:inherit; font-size:13px; font-weight:500; height:32px; padding:0 12px; border-radius:6px; border:1px solid var(--line); background:var(--raise); color:var(--ink); cursor:pointer;
           transition:background .12s, border-color .12s, color .12s; white-space:nowrap; }
  button:hover:not(:disabled) { border-color:var(--faint); }
  button.main { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  button.main:hover:not(:disabled) { background:#7a3fe0; border-color:#7a3fe0; }
  button.quiet { background:transparent; border-color:transparent; color:var(--soft); padding:0 10px; }
  button.quiet:hover:not(:disabled) { background:var(--raise); color:var(--ink); border-color:transparent; }
  button:disabled { opacity:.4; cursor:default; }
  button:focus-visible, select:focus-visible, a:focus-visible, #drop:focus-visible, .ch:focus-within { outline:2px solid var(--accent); outline-offset:2px; }
  select { font:inherit; font-size:13px; height:32px; padding:0 28px 0 10px; border:1px solid var(--line); border-radius:6px; background:var(--raise); color:var(--ink); max-width:100%; }
  input[type=checkbox] { accent-color:var(--accent); width:14px; height:14px; margin:0; flex:none; }
  label.field { display:grid; grid-template-columns:minmax(0, 1fr); gap:4px; font-size:12px; color:var(--soft); min-width:0; }
  label.field select { width:100%; min-width:0; }
  label.check { display:flex; align-items:center; gap:8px; font-size:13px; cursor:pointer; }

  main { max-width:780px; margin:0 auto; padding:28px 24px 96px; display:grid; grid-template-columns:minmax(0, 1fr); gap:16px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:20px; }
  .panel > h3 { margin-bottom:12px; }

  /* Now reading: the one loud element of the page */
  #now { padding:24px; border-color:color-mix(in srgb, var(--accent) 45%, var(--line)); }
  .now-top { display:flex; align-items:flex-end; justify-content:space-between; gap:20px; }
  .now-top > div:first-child { min-width:0; }
  #nowkind { font-size:13px; color:var(--soft); }
  #nowtitle { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #pct { font-size:44px; line-height:1; font-weight:650; letter-spacing:-0.04em; flex:none; }
  #pct small { font-size:18px; font-weight:500; color:var(--soft); letter-spacing:0; margin-left:2px; }
  #left { margin-top:6px; color:var(--soft); font-size:13px; }
  /* The chapters as a row of bars, like the mark: height is the chapter's length, the fill is how much is read */
  #segs { display:flex; align-items:flex-end; gap:3px; height:72px; margin:22px 0 10px; }
  #segs div { flex:1 1 0; max-width:34px; min-width:3px; background:var(--line); border-radius:2px; position:relative; overflow:hidden; }
  #segs i { position:absolute; left:0; right:0; bottom:0; height:0; background:var(--accent); transition:height .5s; }
  #segs div.now { background:var(--accent-wash); box-shadow:0 0 0 1px var(--accent-ink); }
  .now-line { display:flex; justify-content:space-between; gap:16px; font-size:13px; }
  .now-line span:first-child { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #chbar { height:4px; background:var(--line); border-radius:2px; overflow:hidden; margin-top:8px; } #chbar div { height:100%; width:0; background:var(--accent); transition:width .4s; }
  .now-actions { display:flex; align-items:center; gap:8px; margin-top:18px; }
  #log { font:12px/1.6 "SF Mono", ui-monospace, Menlo, monospace; color:var(--soft); white-space:pre-wrap; max-height:150px; overflow:auto; margin:14px 0 0; padding:12px; background:var(--bg); border:1px solid var(--line); border-radius:6px; }
  #nowerr { color:var(--danger); font-size:13px; margin-top:12px; }

  /* Queue */
  #qlist { border:1px solid var(--line); border-radius:8px; background:var(--bg); overflow:hidden; }
  .q { display:grid; grid-template-columns:20px minmax(0, 1fr) auto; gap:4px 12px; padding:12px 14px; border-bottom:1px solid var(--line); align-items:center; }
  .q:last-child { border-bottom:0; }
  .q .dot { width:8px; height:8px; border-radius:50%; background:var(--faint); justify-self:center; }
  .q.running .dot { background:var(--accent); box-shadow:0 0 0 4px var(--accent-wash); } .q.done .dot { background:var(--ok); } .q.error .dot { background:var(--danger); }
  .q .t { font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .q .links { display:flex; gap:14px; font-size:13px; }
  .q .meta { grid-column:2 / 4; display:flex; align-items:center; gap:12px; font-size:13px; color:var(--soft); }
  .q .mini { flex:1; max-width:160px; height:4px; background:var(--line); border-radius:2px; overflow:hidden; } .q .mini i { display:block; height:100%; background:var(--accent); } .q.done .mini i { background:var(--ok); }
  .q.error .note { color:var(--danger); }
  .q .hint { grid-column:2 / 4; font-size:13px; color:var(--soft); }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:14px; }

  /* Add a book */
  #drop { border:1px dashed var(--faint); border-radius:10px; text-align:center; padding:44px 20px; cursor:pointer; color:var(--soft); transition:border-color .12s, background .12s; }
  #drop strong { display:block; color:var(--ink); font-weight:600; font-size:15px; margin-bottom:2px; }
  #drop:hover, #drop.over { border-color:var(--accent); background:var(--accent-wash); }
  #drop.small { padding:14px 20px; } #drop.small strong { font-size:13px; display:inline; margin:0; } #drop.small span { display:none; }
  #steps { list-style:none; counter-reset:s; display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; padding:0; margin:0; }
  #steps li { counter-increment:s; padding:0 4px; color:var(--soft); font-size:13px; }
  #steps li::before { content:counter(s); display:grid; place-items:center; width:22px; height:22px; border-radius:50%; background:var(--raise); border:1px solid var(--line); color:var(--ink); font-size:12px; font-weight:600; margin-bottom:8px; }
  #steps b { display:block; color:var(--ink); font-weight:600; }
  #err { color:var(--danger); font-size:13px; margin:0; } #err:empty { display:none; }

  /* The book being set up */
  .book-head { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
  .book-head > div { min-width:0; }
  #meta { color:var(--soft); font-size:13px; margin-top:2px; }
  .settings { display:grid; grid-template-columns:minmax(0, 1fr) minmax(0, 1.5fr); gap:12px; margin-top:18px; align-items:end; }
  .vhead { display:flex; gap:4px 14px; align-items:baseline; flex-wrap:wrap; margin-top:14px; font-size:12px; color:var(--soft); }
  .vhead a { margin-left:auto; font-size:13px; }
  #voices { display:grid; grid-template-columns:repeat(auto-fill, minmax(172px, 1fr)); gap:6px; margin-top:6px; max-height:216px; overflow:auto; padding:1px; }
  .v { display:flex; align-items:center; gap:10px; padding:6px 10px 6px 6px; border:1px solid var(--line); border-radius:8px; background:var(--bg); cursor:pointer; min-width:0; }
  .v:hover { border-color:var(--faint); } .v:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
  .v.sel { border-color:var(--accent); background:color-mix(in srgb, var(--accent) 9%, var(--bg)); }
  .v .play { width:28px; height:28px; padding:0; border-radius:50%; flex:none; display:grid; place-items:center; }
  .v:not(.ready) .play { border-style:dashed; color:var(--soft); background:transparent; }
  .v.playing .play { background:var(--accent); border-color:var(--accent); color:#fff; }
  .v.busy .play svg { animation:turn .9s linear infinite; }
  .v .who { display:grid; min-width:0; line-height:1.25; }
  .v .vn { font-size:13px; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .v .vd { font-size:11px; color:var(--soft); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .v .vd:empty { display:none; }
  .v.bad .vd { color:var(--danger); }
  @keyframes turn { to { transform:rotate(360deg); } }
  .v .del { margin-left:auto; height:22px; padding:0 6px; font-size:11px; color:var(--soft); background:transparent; border-color:transparent; }
  .v .del:hover:not(:disabled) { color:var(--danger); border-color:var(--line); }
  #clonebox { margin-top:12px; padding:14px 16px; border:1px solid var(--line); border-radius:8px; background:var(--bg); display:grid; gap:10px; }
  #clonebox.hide { display:none; }
  #clonebox h3 { margin:0; font-size:13px; font-weight:600; } #clonebox p { margin:0; font-size:13px; color:var(--soft); max-width:68ch; }
  #clonebox blockquote { margin:0; padding:10px 14px; border-left:2px solid var(--accent); background:var(--panel); font-size:15px; line-height:1.55; max-width:68ch; }
  .crow { display:flex; align-items:center; gap:12px; flex-wrap:wrap; font-size:13px; color:var(--soft); }
  #crec.on { background:var(--danger); border-color:var(--danger); color:#fff; }
  #cname { font:inherit; font-size:13px; height:32px; padding:0 10px; border:1px solid var(--line); border-radius:6px; background:var(--raise); color:var(--ink); width:min(100%, 260px); }
  #cname:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  #cplay { height:32px; max-width:100%; } #cplay.hide { display:none; }
  .cover { width:52px; height:74px; object-fit:cover; border-radius:4px; border:1px solid var(--line); flex:none; background:var(--raise); }
  .cover.ph { display:grid; place-items:center; background:var(--tile); border-color:var(--tile-line); }   /* no cover in the book: the mark, in the Mac's light or dark */
  .with-cover { display:flex; gap:14px; align-items:center; min-width:0; } .with-cover > div { min-width:0; }
  #summary { margin-top:16px; padding:12px 14px; border-radius:8px; background:var(--raise); font-size:13px; }
  #summary b { font-weight:600; }
  .pick { display:flex; gap:14px; align-items:center; margin-top:18px; font-size:13px; flex-wrap:wrap; }
  .pick h3 { margin-right:auto; }
  #chapters { max-height:308px; overflow:auto; margin-top:10px; border:1px solid var(--line); border-radius:8px; background:var(--bg); }
  .ch { display:grid; grid-template-columns:14px 26px minmax(0, 1fr) auto; gap:10px; padding:8px 12px; border-bottom:1px solid var(--line); align-items:center; font-size:13px; cursor:pointer; }
  .ch:last-child { border-bottom:0; } .ch:hover { background:var(--raise); }
  .ch .n { color:var(--faint); text-align:right; } .ch .t { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; } .ch .m { color:var(--soft); }
  .ch.done .t, .ch.skip .t { color:var(--soft); }
  .tagk { color:var(--soft); font-size:11px; margin-left:8px; border:1px solid var(--line); border-radius:4px; padding:1px 6px; }
  .ch.done .tagk { color:var(--ok); border-color:color-mix(in srgb, var(--ok) 40%, var(--line)); }
  .icloud { display:flex; align-items:center; gap:8px; margin-top:14px; }
  .info { position:relative; display:inline-flex; }
  .i { width:18px; height:18px; padding:0; border-radius:50%; font-size:11px; font-weight:700; font-style:italic; font-family:Georgia, serif; line-height:1; color:var(--soft); background:transparent; border-color:var(--faint); }
  #tip { position:absolute; left:50%; bottom:calc(100% + 10px); transform:translateX(-50%); width:300px; padding:12px 14px; border-radius:8px; background:var(--ink); color:var(--bg);
         font-size:13px; line-height:1.45; opacity:0; visibility:hidden; transition:opacity .12s; z-index:6; pointer-events:none; }
  #tip::after { content:""; position:absolute; left:50%; top:100%; margin-left:-6px; border:6px solid transparent; border-top-color:var(--ink); }
  .info:hover #tip, .info:focus-within #tip { opacity:1; visibility:visible; }
  @media (max-width:640px) { #tip { position:fixed; left:16px; right:16px; bottom:16px; width:auto; transform:none; } #tip::after { display:none; } }
  .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:16px; align-items:center; }
  .actions .main { margin-left:auto; }
  audio { width:100%; height:36px; margin-top:14px; }
  #cmprows { border:1px solid var(--line); border-radius:8px; background:var(--bg); margin-top:10px; }
  .c { display:grid; grid-template-columns:130px minmax(0, 1fr); gap:6px 14px; padding:12px 14px; border-bottom:1px solid var(--line); align-items:center; font-size:13px; }
  .c:last-child { border-bottom:0; } .c b { font-weight:600; } .c audio { margin:0; height:32px; grid-column:2; } .c .bad { color:var(--danger); } .c .m { color:var(--soft); }

  #toast { position:fixed; left:50%; bottom:28px; transform:translate(-50%, 20px); background:var(--ink); color:var(--bg); padding:9px 16px; border-radius:8px; font-size:13px; font-weight:500;
           opacity:0; pointer-events:none; transition:opacity .2s, transform .2s; z-index:10; }
  #toast.on { opacity:1; transform:translate(-50%, 0); }
  .center { max-width:480px; margin:72px auto; text-align:center; } .center p { color:var(--soft); margin:8px 0 0; }

  @media (max-width:640px) {
    header > div, main { padding-left:16px; padding-right:16px; } #mac { display:none; }
    .settings, #steps { grid-template-columns:1fr; } #powertext { display:none; } #pct { font-size:34px; } .panel, #now { padding:16px; } #nowtitle { white-space:normal; }
    .now-line { flex-wrap:wrap; gap:2px 16px; } .now-actions { flex-wrap:wrap; } .actions .main { margin-left:0; flex:1 1 100%; order:-1; }
    .q { grid-template-columns:20px minmax(0, 1fr); } .q .links { grid-column:2; } .q .meta, .q .hint { grid-column:2; }
  }
  @media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style></head><body>
<header><div>
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="28" height="28" role="img" aria-label="book2audio"><rect x="0.5" y="0.5" width="63" height="63" rx="13" fill="var(--tile)" stroke="var(--tile-line)"/><path d="M13 30 V34 M22 26 V38 M42 26 V38 M51 30 V34" fill="none" stroke="var(--glyph)" stroke-width="6" stroke-linecap="square"/><rect x="28" y="20" width="8" height="24" fill="#8C52EE"/></svg>
  <span class="word">book2audio</span><span id="mac"></span>
  <span class="bar-actions">
    <span id="power" class="on"><i></i><span id="powertext">On</span></span>
    <button id="restart" class="quiet icon" aria-label="Restart book2audio" title="Restart, for example after an update. A book being read goes on where it stopped.">
      <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M16 10a6 6 0 1 1-1.9-4.4"/><path d="M16.2 3.2v3.4h-3.4"/></svg></button>
    <button id="quit" class="quiet icon" aria-label="Turn book2audio off" title="Turn book2audio off. Everything read so far is kept.">
      <svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M10 3v7"/><path d="M6 5.3a6 6 0 1 0 8 0"/></svg></button>
  </span>
</div></header>

<div id="gone" class="hide center"><h2>book2audio is off</h2><p>Everything read so far is kept, and the queue will be there next time. To turn it on again, open book2audio from your Applications folder. This page reconnects by itself.</p></div>
<div id="back" class="hide center"><h2>Restarting</h2><p>This page comes back by itself in a few seconds. Nothing read so far is lost.</p></div>

<main>
  <section id="now" class="panel hide" aria-live="polite">
    <div class="now-top">
      <div class="with-cover"><img id="nowcover" class="cover hide" alt=""><span id="nowph" class="cover ph" aria-hidden="true"><svg viewBox="0 0 64 64" width="30" height="30"><path d="M13 30 V34 M22 26 V38 M42 26 V38 M51 30 V34" fill="none" stroke="var(--glyph)" stroke-width="6" stroke-linecap="square"/><rect x="28" y="20" width="8" height="24" fill="#8C52EE"/></svg></span><div><div id="nowkind"></div><h2 id="nowtitle"></h2></div></div>
      <div id="pct"></div>
    </div>
    <div id="left"></div>
    <div id="segs" aria-label="Chapters of this book"></div>
    <div class="now-line"><span id="chname"></span><span id="chpct" class="soft"></span></div>
    <div id="chbar" role="progressbar" aria-label="Current chapter" aria-valuemin="0" aria-valuemax="100"><div></div></div>
    <div id="nowerr" class="hide"></div>
    <div class="now-actions">
      <button id="stop">Pause</button>
      <span id="segnote" class="soft" style="font-size:13px"></span>
      <a id="more" style="margin-left:auto;font-size:13px">Show details</a>
    </div>
    <pre id="log" class="hide"></pre>
  </section>

  <section id="queue" class="panel hide">
    <h3>Queue</h3>
    <div id="qlist"></div>
    <div class="row">
      <button id="qstart" class="main">Start reading</button>
      <span id="qnote" class="soft" style="font-size:13px"></span>
    </div>
  </section>

  <div id="drop" tabindex="0" role="button"><strong>Drop an ebook here, or click to choose one</strong><span>EPUB without DRM. AZW3 and MOBI work if Calibre is installed.</span></div>
  <input id="file" type="file" accept=".epub,.azw3,.azw,.mobi" class="hide">
  <p id="err" role="alert"></p>
  <ol id="steps"><li><b>Drop a book</b>It stays on this Mac.</li><li><b>Hear the voice</b>One minute, before you commit.</li><li><b>Queue it and leave</b>It reads chapter by chapter and can be paused at any time.</li></ol>

  <section id="book" class="panel hide">
    <div class="book-head">
      <div class="with-cover"><img id="bookcover" class="cover hide" alt=""><span id="bookph" class="cover ph" aria-hidden="true"><svg viewBox="0 0 64 64" width="30" height="30"><path d="M13 30 V34 M22 26 V38 M42 26 V38 M51 30 V34" fill="none" stroke="var(--glyph)" stroke-width="6" stroke-linecap="square"/><rect x="28" y="20" width="8" height="24" fill="#8C52EE"/></svg></span><div><h2 id="title"></h2><div id="meta"></div></div></div>
      <button id="closebook" class="quiet" aria-label="Close this book">Close</button>
    </div>
    <div class="settings">
      <label class="field"><span>Language<span id="langnote"></span></span><select id="lang"></select></label>
      <label class="field">Voice model<select id="model"><option value="voxtral-4bit" data-label="Natural (Voxtral 4-bit)"></option><option value="voxtral" data-label="Natural, full size (Voxtral, slower)"></option><option value="kyutai" data-label="Kyutai TTS (experimental)"></option><option value="pocket" data-label="Light, on the CPU (Pocket TTS)"></option><option value="kokoro" data-label="Fast (Kokoro)"></option><option value="clone" data-label="Your own voice (experimental)"></option></select></label>
    </div>
    <div class="vhead"><span id="vlabel">Voice</span><span id="vnote"></span><a id="vall" class="hide"></a></div>
    <div id="voices" role="radiogroup" aria-labelledby="vlabel"></div><input type="hidden" id="voice">
    <div id="clonebox" class="hide">
      <h3>Add a voice</h3>
      <p>Read this aloud in a quiet room, at your normal pace. About 15 seconds are enough. A recording you already have works too.</p>
      <blockquote id="cscript"></blockquote>
      <div class="crow"><button type="button" id="crec">Record</button><span id="ctime"></span><a id="cfile">Choose a recording</a><input type="file" id="cpick" accept="audio/*,.m4a,.wav,.mp3,.aac" hidden></div>
      <audio id="cplay" controls class="hide"></audio>
      <div class="crow"><input type="text" id="cname" placeholder="Name of this voice" maxlength="40" aria-label="Name of this voice"></div>
      <label class="check"><input type="checkbox" id="cok"> This is my own voice, or the person has agreed to this use</label>
      <div class="crow"><button type="button" class="main" id="csave" disabled>Save this voice</button><span id="cnote">The recording stays on this Mac.</span></div>
    </div>
    <div id="summary"></div>
    <div class="pick"><h3>Chapters</h3><a id="all">All chapters</a><a id="todo">Only those not done</a><a id="none">None</a></div>
    <div id="chapters"></div>
    <div class="icloud"><label class="check"><input type="checkbox" id="iphone" checked> Copy the finished audiobook to iCloud Drive</label>
      <span class="info"><button type="button" class="i" aria-label="What does this do?" aria-describedby="tip">i</button>
        <span id="tip" role="tooltip">When the book is finished, its audiobook file (.m4b) is copied to iCloud Drive, in a folder named Audiobooks. From there it appears on your iPhone and your other Apple devices: open the Files app, then iCloud Drive, then Audiobooks. The original stays on this Mac.</span></span></div>
    <div class="actions">
      <button id="preview">Play a 1-minute preview</button>
      <button id="compare">Compare models</button>
      <button id="folder" class="quiet">Open folder</button>
      <button id="add" class="main">Add to the queue</button>
    </div>
    <audio id="player" controls class="hide"></audio>
    <div id="cmp" class="hide"><h3 style="margin-top:20px">Same passage, each model, timed on this Mac</h3><div id="cmprows"></div></div>
  </section>
</main>
<div id="toast" role="status"></div>

<script>
const $ = id => document.getElementById(id);
const STATUS = { waiting:"Waiting", running:"Reading now", done:"Ready", error:"Failed" };
let book = null, lastPreview = 0, timer = null, speeds = {}, quitting = false, restarting = false, offline = false, boot = "", wasRunning = false, toastTimer = null;
const api = (path, opts) => fetch(path, opts).then(async r => { const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j; });
const post = (path, data) => api(path, { method:"POST", body:JSON.stringify(data || {}) });
const fail = e => { $("err").textContent = e.message || e; };
const dur = h => h < 1 ? `${Math.max(1, Math.round(h * 60))} min` : `${h >= 10 ? Math.round(h) : h.toFixed(1)} h`;
function toast(text) { $("toast").textContent = text; $("toast").classList.add("on"); clearTimeout(toastTimer); toastTimer = setTimeout(() => $("toast").classList.remove("on"), 2600); }

const ICON = { play:'<svg viewBox="0 0 12 12" width="11" height="11" aria-hidden="true"><path d="M3 1.5v9l7.5-4.5z" fill="currentColor"/></svg>',
               pause:'<svg viewBox="0 0 12 12" width="11" height="11" aria-hidden="true"><path d="M3.5 2v8M8.5 2v8" stroke="currentColor" stroke-width="2.4" fill="none"/></svg>',
               wait:'<svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true"><path d="M6 1.2a4.8 4.8 0 1 1-4.8 4.8" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linecap="round"/></svg>' };
const vplayer = new Audio();
let samples = new Set(), sampleErrors = {}, sampling = "", wantPlay = "", playing = "", working = false;
let myVoices = [];
const READ_ALOUD = { fr:"Il y a des livres que l'on garde longtemps près de soi. On les ouvre le soir, quand la maison est calme, et l'on retrouve aussitôt les mêmes voix, les mêmes lieux, comme si rien n'avait changé depuis la dernière page. Voilà pourquoi j'aime tant qu'on me lise une histoire.",
  en:"Some books stay close to us for a long time. We open them in the evening, when the house is quiet, and we find the same voices and the same places at once, as if nothing had changed since the last page. That is why I love being read a story." };
const cloning = () => $("model").value === "clone";
const voiceList = () => cloning() ? myVoices.map(v => ({ id:v.id, name:v.name, note:"Your recording", stem:`clone--${$("lang").value}--${v.id}`, mine:true }))
  : ((book && book.catalog.voices[$("model").value]) || {})[$("lang").value] || [];
function fillVoices() {                               // one row per voice: click to choose it, play to hear it
  const list = voiceList(), keep = $("voice").value;
  $("voice").value = list.some(v => v.id === keep) ? keep : (list[0] || {}).id || "";
  vplayer.pause(); $("voices").innerHTML = "";
  $("clonebox").classList.toggle("hide", !cloning()); $("cscript").textContent = READ_ALOUD[$("lang").value] || READ_ALOUD.en;
  list.forEach(v => {
    const row = document.createElement("div"); row.className = "v"; row.tabIndex = 0; row.setAttribute("role", "radio"); row.dataset.id = v.id;
    row.innerHTML = '<button type="button" class="play"></button><span class="who"><span class="vn"></span><span class="vd"></span></span>';
    row.querySelector(".vn").textContent = v.name;
    row.onclick = () => { $("voice").value = v.id; paintVoices(); };
    row.onkeydown = e => { if (e.target === row && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); row.click(); } };
    row.querySelector(".play").onclick = e => { e.stopPropagation(); playVoice(v); };
    if (v.mine) { const del = document.createElement("button"); del.type = "button"; del.className = "del"; del.textContent = "Delete";
      del.onclick = e => { e.stopPropagation();
        if (del.dataset.sure) api("/myvoice/delete", { method:"POST", body:JSON.stringify({ id:v.id }) }).then(r => { myVoices = r.my_voices; fillVoices(); toast(`Deleted: ${v.name}`); }).catch(fail);
        else { del.dataset.sure = 1; del.textContent = "Delete for good?"; setTimeout(() => { delete del.dataset.sure; del.textContent = "Delete"; }, 4000); } };
      row.append(del); }
    $("voices").append(row);
  });
  paintVoices();
}
function paintVoices() {                              // only classes and labels change, so a sample that is playing keeps playing
  const list = voiceList(), missing = list.filter(v => !samples.has(v.stem));
  list.forEach((v, k) => { const row = $("voices").children[k]; if (!row) return;
    const ready = samples.has(v.stem), busy = sampling === v.stem, on = playing === v.stem && !vplayer.paused, bad = sampleErrors[v.stem];
    row.className = "v" + (v.id === $("voice").value ? " sel" : "") + (ready ? " ready" : "") + (busy ? " busy" : "") + (on ? " playing" : "") + (bad && !ready ? " bad" : "");
    row.setAttribute("aria-checked", v.id === $("voice").value);
    row.querySelector(".vd").textContent = bad && !ready ? "Sample failed" : v.note; row.title = bad && !ready ? bad : "";
    const b = row.querySelector(".play"), state = busy ? "wait" : on ? "pause" : "play";
    if (b.dataset.state !== state) { b.dataset.state = state; b.innerHTML = ICON[state]; }
    b.disabled = !ready && working;
    b.setAttribute("aria-label", busy ? `Creating the sample of ${v.name}` : on ? `Stop ${v.name}` : ready ? `Play ${v.name}` : `Create and play the sample of ${v.name}`); });
  const speed = speeds[$("model").value], secs = missing.length * 12 / (speed || 1) + 40;
  $("vnote").textContent = !list.length ? (cloning() ? "No voice yet: add one below." : "") : working && sampling ? "Creating samples..." : !missing.length ? "Press play to hear a voice."
    : samples.size && missing.length < list.length ? "Press play to hear a voice. A dashed button creates its sample first."
    : "Press play to hear a voice. Its sample is created the first time, then kept on this Mac.";
  $("vall").classList.toggle("hide", working || missing.length < 2);
  $("vall").textContent = `Create ${missing.length === list.length ? "all" : "the missing"} ${missing.length} samples` + (speed ? `, about ${Math.max(1, Math.round(secs / 60))} min` : "");
}
function startSample(v) { playing = v.stem; vplayer.src = "/voice?s=" + encodeURIComponent(v.stem); return vplayer.play(); }
vplayer.addEventListener("error", () => { playing = ""; poll(); toast("This sample could not be fetched. Press play again to create it on this Mac."); });
function playVoice(v) {
  if (playing === v.stem && !vplayer.paused) { vplayer.pause(); return; }
  if (samples.has(v.stem)) { startSample(v).catch(fail); return; }
  wantPlay = v.stem; makeSamples([v.id]);
}
function makeSamples(ids) { $("err").textContent = ""; post("/voices", { model:$("model").value, lang:$("lang").value, voices:ids }).then(poll).catch(e => { wantPlay = ""; fail(e); }); }
["play", "pause", "ended"].forEach(n => vplayer.addEventListener(n, paintVoices));
// ---- a voice of your own: record it in the page, or pick a file
let take = null, recorder = null, recTimer = null;
const canSave = () => { $("csave").disabled = !(take && $("cok").checked); };
function setTake(blob, ext) { take = { blob, ext }; $("cplay").src = URL.createObjectURL(blob); $("cplay").classList.remove("hide"); canSave(); }
function stopRecording() { if (recorder && recorder.state !== "inactive") recorder.stop(); }
$("crec").onclick = () => {
  if (recorder && recorder.state === "recording") return stopRecording();
  if (!navigator.mediaDevices || !window.MediaRecorder) return fail("This browser cannot record. Choose a recording instead.");
  navigator.mediaDevices.getUserMedia({ audio:true }).then(stream => {
    const chunks = [], started = Date.now(); recorder = new MediaRecorder(stream);
    recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => { clearInterval(recTimer); stream.getTracks().forEach(t => t.stop()); $("crec").classList.remove("on"); $("crec").textContent = "Record again";
      const type = recorder.mimeType || "audio/webm"; setTake(new Blob(chunks, { type }), type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm"); };
    recorder.start(); $("crec").classList.add("on"); $("crec").textContent = "Stop"; $("err").textContent = "";
    recTimer = setInterval(() => { const sec = Math.floor((Date.now() - started) / 1000); $("ctime").textContent = `${sec} s` + (sec < 10 ? "" : ", that is enough"); if (sec >= 30) stopRecording(); }, 250);
  }).catch(() => fail("The microphone is not available. Allow it for this page, or choose a recording instead."));
};
$("cfile").onclick = () => $("cpick").click();
$("cpick").onchange = e => { const f = e.target.files[0]; e.target.value = ""; if (!f) return;
  setTake(f, (f.name.split(".").pop() || "wav").toLowerCase()); $("ctime").textContent = f.name; if (!$("cname").value) $("cname").value = f.name.replace(/\.[^.]+$/, "").slice(0, 40); };
$("cok").onchange = canSave;
$("csave").onclick = () => { if (!take) return; $("csave").disabled = true; $("err").textContent = "";
  api("/myvoice", { method:"POST", headers:{ "X-Name":encodeURIComponent($("cname").value || "My voice"), "X-Ext":take.ext, "X-Consent":"yes" }, body:take.blob })
    .then(r => { myVoices = r.my_voices; take = null; $("cplay").classList.add("hide"); $("cplay").removeAttribute("src"); $("cname").value = ""; $("ctime").textContent = ""; $("cok").checked = false; $("crec").textContent = "Record";
      fillVoices(); $("voice").value = r.voice.id; paintVoices(); toast(`Voice saved: ${r.voice.name}. Press play to hear it read.`); })
    .catch(fail).finally(canSave); };
function showVoices(s) {
  if (JSON.stringify(s.my_voices || []) !== JSON.stringify(myVoices)) { myVoices = s.my_voices || []; if (book && cloning()) fillVoices(); }
  samples = new Set(s.samples || []); sampleErrors = s.sample_errors || {}; sampling = s.sampling || ""; working = s.running;
  const v = wantPlay && voiceList().find(x => x.stem === wantPlay);
  if (v && samples.has(v.stem)) { wantPlay = ""; startSample(v).catch(() => toast(`The sample of ${v.name} is ready: press play.`)); }
  if (!s.running) wantPlay = "";
  paintVoices();
}
function fillModels() {                                // only the models that speak this language and fit this Mac can be picked
  const c = book.catalog, lang = $("lang").value, best = c.recommended[lang];
  [...$("model").options].forEach(o => { const fit = c.fits[o.value] || "fits", speaks = (c.speaks[o.value] || []).includes(lang);
    o.disabled = !speaks || fit === "no";
    o.textContent = o.dataset.label + (!speaks ? `, no ${c.languages[lang]} voice` : o.value === best ? ", recommended" : fit === "no" ? ", too big for this Mac" : fit === "tight" ? ", tight on this Mac" : ""); });
  $("model").value = best; fillVoices();
}
function countSelected() {
  const boxes = [...document.querySelectorAll("#chapters input:checked")];
  const hours = boxes.reduce((s, b) => s + book.chapters[b.value - 1].minutes, 0) / 60, speed = speeds[$("model").value];
  $("summary").innerHTML = "";
  const b = document.createElement("b"); b.textContent = `${boxes.length} chapter${boxes.length === 1 ? "" : "s"}, ${dur(hours)} of audio.`;
  $("summary").append(b, boxes.length ? (speed ? ` About ${dur(hours / speed)} to create on this Mac with this model.` : " Its speed on this Mac is not known yet: run a comparison to measure it.") : " Tick at least one chapter.");
  $("add").disabled = !boxes.length;
  return boxes.map(b => +b.value);
}
function showBook(b) {
  book = b; $("err").textContent = "";
  $("title").textContent = b.title; $("meta").textContent = [b.author, b.out].filter(Boolean).join(", saved in ");
  $("lang").innerHTML = "";
  const short = b.language.slice(0, 2);
  Object.entries(b.catalog.languages).forEach(([code, name]) => { const o = document.createElement("option"); o.value = code; o.textContent = name; $("lang").append(o); });
  $("langnote").textContent = b.catalog.languages[short] ? `, ${b.catalog.languages[short]} according to the book` : "";
  $("lang").value = b.catalog.languages[short] ? short : "en"; fillModels();
  $("bookcover").classList.toggle("hide", !b.cover); $("bookph").classList.toggle("hide", !!b.cover); if (b.cover) $("bookcover").src = "/cover?t=" + Date.now();
  $("chapters").innerHTML = "";
  b.chapters.forEach(c => {
    const row = document.createElement("label"); row.className = "ch" + (c.done ? " done" : c.skip ? " skip" : "");
    row.innerHTML = `<input type="checkbox" value="${c.n}" ${c.done || c.skip ? "" : "checked"}><span class="n">${c.n}</span><span class="t"></span><span class="m">${c.minutes} min</span>`;
    row.querySelector(".t").textContent = c.title;
    if (c.done || c.skip) { const k = document.createElement("span"); k.className = "tagk"; k.textContent = c.done ? "done" : "front matter"; row.querySelector(".t").append(k); }
    $("chapters").append(row);
  });
  $("book").classList.remove("hide"); $("player").classList.add("hide"); $("cmp").classList.add("hide");
  layout(); countSelected(); $("book").scrollIntoView({ behavior:"smooth", block:"start" });
}
function closeBook() { vplayer.pause(); book = null; $("book").classList.add("hide"); layout(); }
function upload(file) {
  if (!file) return; $("drop").querySelector("strong").textContent = "Reading " + file.name + "...";
  api("/open", { method:"POST", headers:{ "X-Name": encodeURIComponent(file.name) }, body:file }).then(showBook).catch(fail).finally(layout);
}
function settings() {
  const chapters = countSelected(); if (!chapters.length) { fail("Tick at least one chapter."); return null; }
  if (cloning() && !$("voice").value) { fail("Add a voice first: this model reads with a voice you give it."); return null; }
  $("err").textContent = "";
  return { chapters, lang:$("lang").value, model:$("model").value, voice:$("voice").value, iphone:$("iphone").checked };
}
let hasQueue = false;
function layout() {                                   // the drop zone shrinks once there is something else to look at
  const busy = !!book || hasQueue;
  $("drop").classList.toggle("small", busy); $("steps").classList.toggle("hide", busy);
  $("drop").querySelector("strong").textContent = busy ? "Add another ebook" : "Drop an ebook here, or click to choose one";
}
function showQueue(s) {
  hasQueue = s.queue.length > 0; $("queue").classList.toggle("hide", !hasQueue);
  $("qlist").innerHTML = "";
  s.queue.forEach(q => {
    const row = document.createElement("div"); row.className = "q " + q.status;
    row.innerHTML = `<span class="dot"></span><span class="t"></span><span class="links"><a class="open">Open folder</a>${q.status === "running" ? "" : `<a class="rm">Remove</a>`}</span>
      <span class="meta"><span class="what"></span><span class="mini"><i style="width:${q.chapters ? 100 * q.done / q.chapters : 0}%"></i></span><span>${q.done} of ${q.chapters} chapters</span></span>`;
    row.querySelector(".t").textContent = q.title;
    const what = row.querySelector(".what"); what.textContent = q.status === "error" && q.note ? "Failed: " + q.note : STATUS[q.status]; if (q.status === "error") what.className = "note";
    if (q.status === "done") { const h = document.createElement("span"); h.className = "hint";
      h.textContent = q.iphone ? "Copied to iCloud Drive. On your iPhone: Files, iCloud Drive, Audiobooks, then open the .m4b file in an audiobook player."
                               : "The audiobook is the .m4b file in its folder."; row.append(h); }
    row.querySelector(".open").onclick = () => post("/folder", { id:q.id }).catch(fail);
    const a = row.querySelector(".rm"); if (a) a.onclick = () => post("/queue/remove", { id:q.id }).then(poll).catch(fail);
    $("qlist").append(row);
  });
  const todo = s.queue.filter(q => q.status === "waiting" || q.status === "error").length;
  const started = s.queue.some(q => q.done > 0 && q.status !== "done");
  $("qstart").textContent = started ? "Resume reading" : "Start reading";
  $("qstart").classList.toggle("hide", s.running || !todo);
  $("qnote").textContent = s.running ? "" : todo ? `${todo} book${todo > 1 ? "s" : ""} waiting` : hasQueue ? "Everything in the queue has been read." : "";
  layout();
}
function showNow(s) {
  const show = s.running || !!s.error;
  $("now").classList.toggle("hide", !show); if (!show) { document.title = "book2audio"; return; }
  const segs = s.running ? s.segments : [], cur = segs.find(g => g.n === s.now), reading = s.running && s.total > 0;
  $("nowkind").textContent = s.running ? (s.kind || "Working") : "Stopped on an error";
  $("nowtitle").textContent = s.title || "";
  const cover = s.book_id && (s.queue.find(q => q.id === s.book_id) || {}).cover ? "/cover?id=" + s.book_id : "";
  $("nowcover").classList.toggle("hide", !cover); $("nowph").classList.toggle("hide", !!cover); if (cover && !$("nowcover").src.endsWith(cover)) $("nowcover").src = cover;
  $("pct").innerHTML = reading ? `${s.percent}<small>%</small>` : "";
  $("left").textContent = reading ? s.left.replace(/^about/, "About") : s.running ? "Getting the voice model ready..." : "";
  document.title = reading ? `${s.percent}% ${s.title || "book2audio"}` : "book2audio";
  const box = $("segs"); box.classList.toggle("hide", segs.length < 2);
  if (box.children.length !== segs.length) box.innerHTML = segs.map(() => "<div><i></i></div>").join("");
  const tallest = Math.max(1, ...segs.map(g => g.w));
  segs.forEach((g, k) => { const d = box.children[k];
    d.style.height = Math.max(8, Math.round(100 * Math.sqrt(g.w / tallest))) + "%"; d.className = g.n === s.now ? "now" : "";
    d.title = `${g.title}: ${g.p}%`; d.firstChild.style.height = g.p + "%"; });
  const cp = reading ? Math.floor(100 * s.i / s.total) : 0;
  $("chname").textContent = cur ? cur.title : "";
  $("chpct").textContent = reading ? `${cp}% of this ${segs.length ? "chapter" : "passage"}` : "";
  $("chbar").firstElementChild.style.width = cp + "%"; $("chbar").setAttribute("aria-valuenow", cp);
  const done = segs.filter(g => g.p === 100).length;
  $("segnote").textContent = segs.length > 1 ? `${done} of ${segs.length} chapters done` : "";
  $("stop").classList.toggle("hide", !s.running);
  $("nowerr").classList.toggle("hide", !s.error); $("nowerr").textContent = s.error || "";
  $("log").textContent = s.lines.join("\n"); if (s.error) $("log").classList.remove("hide");
  $("more").textContent = $("log").classList.contains("hide") ? "Show details" : "Hide details"; $("log").scrollTop = 1e9;
}
let cmpSeen = "";
function showCompare(s) {
  const rows = s.compare || []; $("cmp").classList.toggle("hide", !rows.length || !book);
  const sig = JSON.stringify(rows) + s.comparing; if (sig === cmpSeen) return; cmpSeen = sig;   // do not restart players that are playing
  $("cmprows").innerHTML = "";
  rows.forEach(r => {
    const d = document.createElement("div"); d.className = "c";
    d.innerHTML = `<b></b><span class="m"></span>` + (r.file ? `<audio controls preload="none" src="/audio?f=${encodeURIComponent(r.file)}&t=${Date.now()}"></audio>` : "");
    d.querySelector("b").textContent = r.label;
    const m = d.querySelector(".m");
    if (r.error) { m.textContent = "Failed: " + r.error; m.classList.add("bad"); }
    else if (!r.file) m.textContent = s.comparing === r.model ? "Reading..." : "Waiting";
    else m.textContent = `${r.speed}x real time: ${r.audio_s} s of audio in ${r.compute_s} s`;
    $("cmprows").append(d);
  });
}
function poll() {
  clearTimeout(timer);
  api("/status").then(s => {
    if ((restarting || offline) && s.boot !== boot) { location.reload(); return; }     // it is back: start from a clean page
    if (restarting) { timer = setTimeout(poll, 700); return; }             // the old one is still closing
    boot = s.boot; speeds = s.speeds || {}; if (book) countSelected();
    if (s.machine && s.machine.memory_gb) $("mac").textContent = `on ${s.machine.chip}, ${s.machine.memory_gb} GB`;
    showQueue(s); showNow(s); showCompare(s); showVoices(s);
    $("preview").disabled = $("compare").disabled = s.running;
    if (s.preview && s.preview_at !== lastPreview) { lastPreview = s.preview_at; $("player").src = "/audio?f=" + encodeURIComponent(s.preview) + "&t=" + s.preview_at; $("player").classList.remove("hide"); $("player").play().catch(() => {}); }
    if (book && s.book_done) s.book_done.forEach(n => { const row = $("chapters").children[n - 1]; if (row && !row.classList.contains("done")) { row.classList.add("done"); row.classList.remove("skip");
      const k = row.querySelector(".tagk") || row.querySelector(".t").appendChild(Object.assign(document.createElement("span"), { className:"tagk" })); k.textContent = "done"; } });
    if (wasRunning && !s.running && !s.error && s.kind && s.kind.startsWith("Book") && s.queue.every(q => q.status !== "waiting")) toast("The queue is finished.");
    wasRunning = s.running;
    if (s.running) timer = setTimeout(poll, 1000);
  }).catch(() => { if (restarting) { timer = setTimeout(poll, 700); return; }
    offline = true; boot = boot || "?";                // turned off, closed or crashed: say so, and keep listening for its return
    document.querySelector("main").classList.add("hide"); $("gone").classList.remove("hide");
    $("power").className = "off"; $("powertext").textContent = "Off"; $("restart").disabled = $("quit").disabled = true; document.title = "book2audio (off)";
    timer = setTimeout(poll, 2000); });
}
$("drop").onclick = () => $("file").click();
$("drop").onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } };
$("file").onchange = e => { upload(e.target.files[0]); e.target.value = ""; };   // so the same book can be chosen again
["dragover","dragenter"].forEach(n => document.addEventListener(n, e => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave","drop"].forEach(n => document.addEventListener(n, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
document.addEventListener("drop", e => upload(e.dataTransfer.files[0]));              // a book can be dropped anywhere on the page
$("lang").onchange = () => { fillModels(); countSelected(); };
$("model").onchange = () => { fillVoices(); countSelected(); }; $("chapters").onchange = countSelected;
const setAll = fn => { document.querySelectorAll("#chapters .ch").forEach(r => r.querySelector("input").checked = fn(r)); countSelected(); };
$("all").onclick = () => setAll(r => !r.classList.contains("skip")); $("none").onclick = () => setAll(() => false);
$("todo").onclick = () => setAll(r => !r.classList.contains("done") && !r.classList.contains("skip"));
$("closebook").onclick = closeBook;
$("vall").onclick = () => makeSamples(voiceList().filter(v => !samples.has(v.stem)).map(v => v.id));
$("preview").onclick = () => { const d = settings(); if (d) post("/preview", d).then(poll).catch(fail); };
$("compare").onclick = () => { const d = settings(); if (d) post("/compare", d).then(poll).catch(fail); };
$("add").onclick = () => { const d = settings(); if (d) post("/queue/add", d).then(() => { toast(`Added to the queue: ${book.title}`); closeBook(); poll(); window.scrollTo({ top:0, behavior:"smooth" }); }).catch(fail); };
$("qstart").onclick = () => post("/queue/start").then(poll).catch(fail);
$("stop").onclick = () => { $("stop").disabled = true; post("/stop").then(() => toast("Pausing after this passage. The position is saved.")).catch(fail).finally(() => { setTimeout(() => { $("stop").disabled = false; }, 3000); poll(); }); };
$("folder").onclick = () => post("/folder").catch(fail);
$("more").onclick = () => { $("log").classList.toggle("hide"); $("more").textContent = $("log").classList.contains("hide") ? "Show details" : "Hide details"; $("log").scrollTop = 1e9; };
$("restart").onclick = () => { restarting = true; clearTimeout(timer);
  document.querySelector("main").classList.add("hide"); $("back").classList.remove("hide");
  post("/restart").catch(() => {}).finally(() => { timer = setTimeout(poll, 700); }); };
$("quit").onclick = () => { quitting = true; post("/quit").catch(() => {}).finally(() => setTimeout(poll, 1200)); };
layout(); poll();
</script></body></html>
"""

QUEUE_FILE = Path.home() / "Audiobooks" / ".queue.json"
# The tab icon as plain PNG files (64 and 180 pixels): some browsers do not draw an SVG one.
ICON_64 = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAABmJLR0QA/wD/AP+gvaeTAAAEF0lEQVR4nO2bz28bRRTH33s7s57Nxk7jH0UqiJORpSAnov0LOBXEhQOCcoQeOPIHILUCFPVYTi0HxJEiIY5IhQP5CxBNIqdKolZBQj3UzcZN6nh2vTuPgxuURm4StfFO6t3PzTsz3u98/Xb0Zj0P4QgajUYRQFw0xnzKDLOIVEXkEgDSUWPThQ0zbjObR4i4SIQ/CwG3W63Wk8NG4fMams3mtNbxVQD8Io77kda9Uhz3wRgDxjAA8Eim8eIgECEQEQghQSlvWwjpMvP3nie+WV5e3ho+agiNxszHzPCD1j3R7T7xjDEjlz8KiAh8v6iVUn1EuLy6uvLLwT7Ogc9Yr8/MA/C1TieY1LonmU/bL318mBmiKBRRFBaknPigUqn5QdBe2N/nGQPq9Zl5Y5IvO51NP0mS1AWPCmMMhGFPFgrqfKXymh8E7T/32v434GnYX+t0Nv1XNeQPg5khDLWrlH+hUjm7FgTtFdhbA5rN5nQYJv9sbW0W4zi2rXWkCCFgerqyE0XumxsbdzoEAKB1fFXrnhj3yQMAxHEMWmspRHgFAAAbjUbRGOdhELTVOIb+MIgIyuWadl2sEYDzfhxHUVYmD08XxTjuR1EEF8kY/kRrXbItKm201kUAvkQAOBfHfdt6UidJ+sjMcwSA1SyF/x5JkgAi1QiRi4PcPlsYw4DIJRrs6rJnwGDOSKdsS5s+uQG2BdgmN8C2ANuItG/41bvrh7bPL7yVmhbIIyA34GQfgQcP/h16/dy5N07F9w0j8xGQG2BbgG2OvQak8Ty+CC+rK/MRkBtgW4BtcgNsC7BNboBtAbbJDbAtwDbHzgRtZ3zP42V1ZT4CcgNsC7DNib4ROul1Io11J/MRkHkDUv9fIO33/keR+QjIDbAtwDa5AbYF2IYA2BxSNzHGIACwIWbcJsqeAUQEzLBNzOaR4xysmxh/BgbwQ0KEJccRmTsnJ4RkRFwiIrqllLdjW1DaKKV2EOknEgJuCyFdytCRwUFlmSuVoj+cdrsdlcvVKhHNRVEobYtLg8nJUs9x6Mbdu63fCAAgjgtfe54XC5H63ih1pJSglOob438Le4nQxsadDiJcnpqqdMf5UUAkKBbLu8z02f37fz2G/VVjm5vtlXK55hcK6nwYanfcDlAjEpw5U+5K6VxfW2vd2Lv+TAIQBO2FavWs53kTF/r90B2XOgIpJUxNVXeldL5bXW1d2d82NAWs19/+iIh/1FqLbnfnVS+d7Sml+sz4+fp669eDfYamgEHQXikWX78pJU5MTPjvuG5hFwAKiAO/Tmc17aBw2nEEuK7iycniju+XjOOIm0nifXjv3uLfw0cdwezsrN/rJe8B8CVmnkOkGiJPAeAp20AwM+NjZtNGxEUAuuV59PvS0lL3sFH/Ae2yjAgONgbYAAAAAElFTkSuQmCC"
ICON_180 = "iVBORw0KGgoAAAANSUhEUgAAALQAAAC0CAYAAAA9zQYyAAAABmJLR0QA/wD/AP+gvaeTAAANLElEQVR4nO3dW2ycV4EH8P8533xz83h8AZJVmqWBuhkbJ0VNty70LaCuAkgUUQR9IC/lIrSLhHYlXtoKeGgqrVZaHhZ2F3GRENEKBCwVDzQBUcQDhN6c4lw8Y09aB7XQJE1ij+3xzHc5Zx9mvslnZ+rMeO4n/9+LZ77LmfPwz8mZc853RqAHpqenU56HGUBkADWplMgA4g4h9BCAUQApAHYv6kYNcwGsAVjWWqwD+g0AWSlFDtC5WMx6fm5ubr3blRLd+pxMZvpBAEe0xmEAMwys8VwAzwN4TkpxMps9ewqA7vSHdjTQExMH7pISRwF9FMB7t79aQykNrXXtNfUvKQUAASFE7fUtXADEcaXwo3z+7IVO1asjgd6/f3oGwOMAPl7vM3zfh+s68DwXnudDKQ++73eiKtQllmVByggiEQuRiA3bjsKyrHqXagC/BHBsYeHci+2uR1sDvX//9P0AjgF4aOs513VRKhXhug7De5uwLAu2HUU8noRt39zD1BonpRRP5nJnX2rXZ7Yl0AcPHhwrlfynhRBfBCCD41orbGwUUSptMMS3OcuKIB6PIx5PQkoZPqUA/E8sJp88c+bM9VY/p+VAZzLv+7TW4j8B7KrVUPkoFtexsbHRje8BNFAEEokkkskkpNzUJbkshP5yLnf+p62VvkP79u2L23bym0KIL904qrG+vo6NjfXqlzui+oQAEokUhoaGNsVQa/y3667/69LSUmlH5e7kpsrohf45gPcHxxynjLW1ArsW1BTLiiCVGkY0GgsffgWQjywsnHm12fKaDvTU1NQB35cnAexB5V8UisVVFItdH0Mng8TjCQwPp8ORvKSU/kg+f/50M+U0FehMZvqw1ngGQBrV4bdC4To8z2umGKK6bNvG8PBoeLivIAQ+kcud+12jZTQc6GqYfwUgDgCe52Jl5TqUUjuoOlF9Ukqk02PhYb6SEPhoo6FuKNCZzIF7tNa/r66zgOM4KBSu84sfdYQQAun0aLhfXVBKHc7n52dvee+tLqh+AfwDgN0A4LplrKwsM8zUUUIAIyPjsO1ocOhNIawHc7m517a7T253ct++fXEp9U+DMFe6GQwzdZ7WwPLydXieExz6O639Z/bu/WBiu/u2DbRtJ78J4F5UvwCurLCbQd2ksbKyHB4KvieZLPz7dnfUXT2C6gwgIP4teL+yco1jzNR1Wmt4notYLAlR6SDPjI/vOnvt2pX5etfX7UMfPHhwrFxW2WA6e22tgI2NYmdrTrSNRCKJVCodvL3kOPbk0tIry1uvq9vlKJX8p4MwO06ZYaae29gownFq/end0aj7VL3rbmqhq0tA/1QJu8a1a2+xq0F9wbIsjI+/M4itL6V8IJs983L4mnot9LHg+Pr6GsNMfcP3/fASC0spdVMrvSnQ1SdNHkJ1CSi7GtRvisU1KFVrZI9kMgf+IXx+U6C1xhM3buQSUOo/WmNTQ6u1fjx8vtaHvuuu6QnLwgIAobXCW29d4eJ86lMC73jHu4InXzQgJ4KlprUW2rJwNAh45V8Aw0z9SodbaQGozwZv5I2DqB0slTa6XUOippTLmzJaa4wlKktDHwz2zXBdlyMb1PeCrTCqJiYmph5AqIU+EpwplTiyQYOhXL7x2KGU1hEEgdYaHwpOuK7bm9oRNclxyuG3hwFATE9Pp1wX1wDYSvm4evVKzypI1Kzx8XcFj2w58bg1Liu7gFY2TgzNlRMNhFA/Ouo46n6pNSaDI57H7gYNFt/3Qq9VRgqhM8EBz+PoBg2W8I4DQoiMrGw2XhFOO9Eg2DzELDISEHeg+mRAaNEH0UCoNMLBrLb+eymlTqMaaKJBFMrusNQaqerhHlaJaOc2Bbr6Az1soWlgbQ10FNV1pkSDKJTd6Lb7chANGgaajMJAk1EYaDIKA01GYaDJKAw0GYWBJqMw0GQUBpqMwkCTURhoMgoDTUZhoMkoDDQZhYEmozDQZJRIryswqHanpvDQ3U92pOzfLD6FS2t1f4aPboGB3qG4ncadozMdK5t2hl0OMgoDTUZhoMkoDDQZhYEmozDQZBQjh+2mpqZw553vbqmM2dnTuHz5ctvq1E67du3CoUP3tlTGxYt/wfy8eWPdRgb60Uc/gy984fMtlfHYY5/DiRMn21andjp06F784Affb6mM7373e/j617/Rtjr1C3Y5yCgMNBmFgSajMNBkFAaajMJAk1EYaDIKA01GYaDJKF2fKRweHoYQoqUyCoVC2+pDby+dbu3JGa01VldX21afRnQ90L/97W+wd+/elsrYs6e1+6kx2ez5lu5//fXXMTPzgbbVpxHscpBRGGgyCgNNRmGgySgMNBmFgSajMNBkFAaajMJAk1EYaDIKA01GYaDJKAw0GYWBJqMw0GQUBpqMwkCTURhoMgoDTUZhoMkoDDQZhYEmozDQZBQGmozCQJNRGGgyCgNNRun63nYf/vBDLW/WSN0xOfm+lu7XWretLo3qeqC7vRsl7dwg7vLKLgcZhYEmozDQZBQGmozCQJNRGGgyCgNNRmGgySgMNBml6zOF3fDjH/8Ep06daqmM2dnTbatPu83OnsZjj32upTIuXvxL2+rTT4wM9Pz8PObn53tdjY65fPkyTpw42etq9CV2OcgoDDQZhYEmozDQZBQGmozCQJNRjBy264aSW8DF5Rc6VjbtjNi/f1oDgOd5uH79rV7Xh6hpY2PvRCRSaZvZ5SCjMNBkFAaajMJAk1EYaDIKA01GYaDJKAw0GYWBJqMw0GQUBpqMwkCTURhoMgoDTUZhoMkoDDQZhYEmo0gADgDwh6loUIWy60gAa5WDTDQNplB2V6UQlUADDDQNplCgC1IpUQBbaBpgm1poQL8RHJTS6mnFiJplWVaodyFel1Lq3I2T3KaDBsvmzOqc1FrUAh2JsIWmwRLsx4HKb4vnpBCYv3HS7lW9iHYk3EJrbWXl6mr8+WAsOhqN9rJuRE0LZdZJJsVL8q9/fbkI4EUAkNKqdrKJ+p9lWbWBDCFwam5ubl0CgNZ4LrjItmM9rCJR46LRG1nVWv8OwVoOyxInghPxeKI3tSNqUiwWr73WGicQBDqbPftHAAsAYNs2ux3U9yzLgm1X+s9aY3Fx8fwL2LzaTv9v8IqtNPW7WOxGRqXEjwBohAOtlDweHEwkklzbQX1MVDMKAFCAdTx4Uwt0Pn/2AoBfAoAQEokEW2nqT4lEElJWoisEfpHLzb0WnNu6wP9Y8CKZHOIaaepLyWStdda+r54On9sU6IWFcy8KgV+jOiadSKS6WE2iWxsaSoUW0Yln8/n52fD5Oo9giScA+JWbh7hgifqGZVlIJmuNrK+UenLrNTcFOpc7+xKA71TeCaRSwx2vKFEjUql06J34r3z+/Omt19R9SNbz4o8D+BuqszGhb5REPZFMDoVnBi85TuRr9a6rG+hXX315RQj9leD90FB60zI9om6KROxwVwNai39aWnplud61bzslePXqlfPj47t2C4H7hai01OVyCVrrDlWb6GZCSIyOjoWfpvrW4uK5/3i767fdl0Pr8r8IIWZR7ZCn02N89pC6RgiB0dHx8MDEnOOsf3W7e7YNdD6fLwPyUwDeRHWdx8jIGMenqeOEAEZGxsJd3b9J6T+8tLRU2u6+W+6clMvNvSal/EcAywBg21Gk0+NsqaljhBBIp8dqi48AFJTSH8tms0u3urehrcCy2TNnhMAnAZRQfUpgZGS8Nv1I1C5SSoyMjIdHNEpKyY/XG6Krp6lmNpOZPqw1fgFgBAB838fq6jJc191B1Yk2s20bw8Mj4T7zilLy4Xz+zO8bLaPpfsPU1NQB35cnANyBylpUFIurKBbXmy2KqCYeTyCVGgl/P3tTa3xkcfHcK82Us6OOcCZzz3sA9TOt9aHgmOM4WFsrwPe9nRRJtynLspBKpTc9TgXgZSn9TzXSZ76pvJ1U4urVS8tjY+kfChEZAzATVCyRSEAIAc9zdlIs3WaGhlJIp0e3rhf6llLlRxcWFq7upMyWhyruvnv6ESHwbQC7g2NK+djYKKJYLAbPDBAB1RGMRCJZXdO8qT19U2vxz4uLZ/+vlfJbfnjw2rUr88PDd3zPstQwgPsASCFkbQ2IEAJK+ZxhvM1V/gcfQjo9ilgsDiFqI2Q+IL7tefFHLlz4c1P95XraOpg8OXnwPqXUUwCObD3nug7K5RJctwzP89v5sdSnIhELth1DLBYPjymHPauUfqLRIblGdGR2pBrsJwA8XG+sWykfjuPA9z14ngff9/llcsBZVgSWZSESicCyIohGo2+3m60C8IxS6tjWxfnt0NHpvkzmnvdo7R8FcBTAxK2uV0oB0NBaQyl2UfqZlKI6WywanWBbEALHAet4+BnAduva/PXExNQHpLSOAPpDAB4AwI30zFYGxPOAeg4Qzy4snHuhGx/akwUZe/bcl0ynyzO+j0kh9H4Ak9WJmhSAsepfboXa39zq7/Ncr/59A0BWa53T2soWi9EXq/smdtX/A4AeJZP0EvviAAAAAElFTkSuQmCC"


NOTIFIER = Path.home() / "Library" / "Application Support" / "book2audio" / "book2audio.app"


def notify(message: str) -> None:
    """A macOS notification, for when the page is not in front. Never an error if it cannot be shown.
    The installer makes a small helper app for it, so that the notification carries the book2audio icon."""
    if sys.platform != "darwin":
        return
    try:
        if NOTIFIER.is_dir():
            (NOTIFIER.parent / "notification.txt").write_text(f"book2audio\n{' '.join(message.split())}\n", encoding="utf-8")
            if subprocess.run(["open", "-g", str(NOTIFIER)], capture_output=True, timeout=5).returncode == 0:
                return
        subprocess.run(["osascript", "-e", f'display notification {json.dumps(message)} with title "book2audio"'],
                       capture_output=True, timeout=5)
    except Exception:
        pass


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
        self.kind = ""                     # "Book 1 of 3", "Preview", "Comparing models"
        self.title = ""                    # the book being read
        self.book_id = ""
        self.results: list[dict] = []      # model comparison, shown under the book
        self.comparing = ""
        self.segments: list[dict] = []     # one per chapter of the book being read
        self.now = 0                       # number of the chapter being read
        self.sampling = ""                 # the voice sample being made
        self.sample_errors: dict[str, str] = {}

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
            self.kind = f"Comparing models: {MODEL_LABELS.get(running, running)}"

    def voices(self, done: int, total: int, label: str, stem: str, error: str = "") -> None:
        if error:
            self.sample_errors[stem] = error
            self.log(f"{label}: {error}")
            return
        self.sampling, self.i, self.total = stem, 0, 0
        self.percent, self.left = int(100 * done / max(total, 1)), ""
        if stem:
            self.sample_errors.pop(stem, None)
            self.title = f"{label}, voice {done + 1} of {total}" if total > 1 else label

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
    session: dict = {"book": None, "out": None, "epub": None, "restart": ""}
    uploads = Path(tempfile.mkdtemp(prefix="book2audio-"))
    boot = str(int(time.time() * 1000))              # lets the page tell a fresh start from the old one
    restarted = os.environ.pop("BOOK2AUDIO_RESTARTED", "")
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
        return argparse.Namespace(**{**vars(args), "model": job.get("model") or args.model, "lang": job.get("lang") or args.lang,
                                     "voice": job.get("voice") or args.voice, "iphone": bool(job.get("iphone"))})

    def language_for(book: Book, opts) -> str:
        if "kokoro" in (opts.model or "") and (opts.voice or "").startswith("b"):
            return "en-gb"
        return (opts.lang or book.language or "en").lower()

    def catalog() -> dict:
        """For the page: which models speak which language on this Mac, and the one to pick for each."""
        return {"languages": LANGUAGES,
                "speaks": {m: v["langs"].split() for m, v in MODELS.items()},
                "fits": {m: model_fit(m) for m in MODELS},
                "recommended": {code: recommend(code)["recommended"] for code in LANGUAGES},
                "voices": {m: {code: [{"id": v, "name": name, "note": note, "stem": voice_stem(m, code, v)} for v, name, note in rows]
                               for code, rows in by_lang.items()} for m, by_lang in VOICE_CATALOG.items()}}

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

        def send_file(self, data: bytes, kind: str, cache: bool = False):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            if cache:
                self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(data)

        def send_audio(self, path: Path):
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
                             "cover": Path(q["out"]).exists() and any(Path(q["out"]).glob("cover.*")),
                             "iphone": bool(q.get("iphone")) and ICLOUD.exists(),
                             "chapters": len(q["chapters"]), "done": len(files_done(Path(q["out"]), q["chapters"]))} for q in items]
                self.reply({"running": state.running, "lines": state.lines[-12:], "i": state.i, "total": state.total,
                            "error": state.error, "percent": state.percent, "left": state.left, "current": state.current,
                            "preview": state.preview, "preview_at": state.preview_at, "queue": view,
                            "compare": state.results, "comparing": state.comparing,
                            "segments": state.segments if state.running else [], "now": state.now,
                            "speeds": load_speeds(), "machine": {"chip": machine_info()["chip"], "memory_gb": machine_info()["memory_gb"]},
                            "samples": sorted(set(voice_samples()) | set(_published)), "sampling": state.sampling if state.running else "",
                            "sample_errors": state.sample_errors, "my_voices": my_voices(),
                            "boot": boot, "kind": state.kind, "title": state.title, "book_id": state.book_id if state.kind.startswith("Book") else "",
                            "book_done": files_done(session["out"], list(range(1, len(book.chapters) + 1))) if book else []})
            elif url.path == "/cover":                # the cover of the book being set up, or of a queued one
                wanted = parse_qs(url.query).get("id", [""])[0]
                data, kind = b"", "image/jpeg"
                with lock:
                    folder = next((Path(q["out"]) for q in items if q["id"] == wanted), None) if wanted else None
                if folder:
                    found = next(iter(sorted(folder.glob("cover.*"))), None) if folder.exists() else None
                    if found:
                        data, kind = found.read_bytes(), "image/png" if found.suffix == ".png" else "image/jpeg"
                elif session["book"] and session["book"].cover:
                    data, kind = session["book"].cover, "image/png" if session["book"].cover_ext == ".png" else "image/jpeg"
                if not data:
                    return self.reply({"error": "no cover"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif url.path == "/audio" and session["out"]:
                name = Path(parse_qs(url.query).get("f", [""])[0]).name
                path = session["out"] / name
                if not name or not path.is_file():
                    return self.reply({"error": "not found"}, 404)
                self.send_audio(path)
            elif url.path == "/voice":                # a voice sample made earlier on this Mac
                stem = parse_qs(url.query).get("s", [""])[0]
                path = voice_samples().get(stem) or fetch_published(stem)
                if not path:
                    return self.reply({"error": "not found"}, 404)
                self.send_audio(path)
            elif url.path in ("/favicon.ico", "/favicon.png", "/apple-touch-icon.png"):
                import base64
                self.send_file(base64.b64decode(ICON_180 if "apple" in url.path else ICON_64), "image/png", cache=True)
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
                                "catalog": catalog(), "cover": bool(book.cover),
                                "chapters": [{"n": n, "title": c.title, "minutes": round(c.chars / 900), "done": n in done,
                                              "skip": is_front_matter(c)}
                                             for n, c in enumerate(book.chapters, 1)]})
                elif self.path == "/compare":
                    if state.running:
                        return self.reply({"error": "Already working. Stop it first."}, 409)
                    if not session["book"]:
                        return self.reply({"error": "Choose a book first."}, 400)
                    state.running, state.stop, state.error, state.lines, state.current = True, False, "", [], "Comparing models"
                    state.kind, state.title = "Comparing models", session["book"].title
                    state.percent, state.left, state.segments, state.now, state.results = 0, "", [], 0, []
                    jobs.put({"kind": "compare", **json.loads(body)})
                    self.reply({"ok": True})
                elif self.path == "/myvoice":
                    if self.headers.get("X-Consent") != "yes":
                        return self.reply({"error": "Confirm first that this is your voice, or that the person agrees."}, 400)
                    if len(body) > 60_000_000:
                        return self.reply({"error": "This recording is too big. 10 to 20 seconds are enough."}, 400)
                    try:
                        voice = save_my_voice(unquote(self.headers.get("X-Name", "")), body, self.headers.get("X-Ext", ""))
                    except ValueError as e:
                        return self.reply({"error": str(e)}, 400)
                    self.reply({"ok": True, "voice": voice, "my_voices": my_voices()})
                elif self.path == "/myvoice/delete":
                    if state.running:
                        return self.reply({"error": "Wait until nothing is being read."}, 409)
                    delete_my_voice(str(json.loads(body).get("id", "")))
                    self.reply({"ok": True, "my_voices": my_voices()})
                elif self.path == "/voices":
                    if state.running:
                        return self.reply({"error": "Voice samples can be created when nothing else is being read."}, 409)
                    job = json.loads(body)
                    model, lang = str(job.get("model", "")), str(job.get("lang", ""))
                    known = [v for v, _, _ in VOICE_CATALOG.get(model, {}).get(lang, [])]
                    if model == "clone" and lang in MODELS["clone"]["langs"].split():
                        known = [v["id"] for v in my_voices()]
                    wanted = [v for v in job.get("voices", []) if v in known]
                    if not wanted:
                        return self.reply({"error": "This model has no such voice in this language."}, 400)
                    if model_fit(model) == "no":
                        return self.reply({"error": "This model is too big for this Mac."}, 400)
                    state.running, state.stop, state.error, state.lines, state.current = True, False, "", [], "Voice samples"
                    state.kind, state.title = "Creating voice samples" if len(wanted) > 1 else "Creating a voice sample", MODEL_LABELS.get(model, model)
                    state.percent, state.left, state.segments, state.now = 0, "", [], 0
                    jobs.put({"kind": "voices", "model": model, "lang": lang, "voices": wanted})
                    self.reply({"ok": True})
                elif self.path == "/preview":
                    if state.running:
                        return self.reply({"error": "Already working. Stop it first."}, 409)
                    if not session["book"]:
                        return self.reply({"error": "Choose a book first."}, 400)
                    state.running, state.stop, state.error, state.lines, state.current = True, False, "", [], "Preview"
                    state.kind, state.title = "Preview", session["book"].title
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
                    if book.cover:
                        (out / f"cover{book.cover_ext}").write_bytes(book.cover)
                    with lock:
                        if any(q["out"] == str(out) and q["status"] == "running" for q in items):
                            return self.reply({"error": "This book is being read right now."}, 409)
                        items[:] = [q for q in items if not (q["out"] == str(out) and q["status"] != "done")]
                        items.append({"id": str(int(time.time() * 1000)), "title": book.title, "epub": str(source), "out": str(out),
                                      "chapters": chapters, "model": job.get("model"), "voice": job.get("voice"), "lang": job.get("lang"),
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
                    state.kind, state.title = "Getting ready", ""
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
                elif self.path == "/restart":
                    session["restart"] = "queue" if state.running and state.current.startswith("Book ") else "idle"
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
            state.kind, state.title, state.book_id = f"Book {position} of {len(items)}", item["title"], item["id"]
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
            notify(f"{item['title']} is ready to listen to." if status == "done" else f"{item['title']} could not be finished.")
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
    threading.Thread(target=load_published, daemon=True).start()
    print(f"book2audio is running at {url}  (close this window or press Ctrl+C to quit)")
    if open_browser and not restarted:
        webbrowser.open(url)
    if restarted == "queue":                         # it was reading when asked to restart: carry on
        state.running, state.stop, state.kind = True, False, "Getting ready"
        jobs.put({"kind": "queue"})

    # The voice model runs here, in the main thread: the web server only hands over jobs.
    while True:
        job = jobs.get()
        if job is None:
            break
        try:
            if job["kind"] == "compare":
                book, opts = session["book"], options(job)
                first = min(n for n in job.get("chapters", [1]) if 1 <= n <= len(book.chapters))
                run_compare(sample_of(book, first), language_for(book, opts), opts, session["out"])
            elif job["kind"] == "voices":
                run_voice_samples(job["model"], job["lang"], job["voices"], args)
            elif job["kind"] == "preview":
                book, opts = session["book"], options(job)
                first = min(n for n in job.get("chapters", [1]) if 1 <= n <= len(book.chapters))
                target = run_preview(book, first, language_for(book, opts), opts, session["out"])
                state.preview, state.preview_at = target.name, int(time.time())
            else:
                run_queue()
        except Stopped:
            state.log("Stopped. The samples already made are kept." if job["kind"] == "voices"
                      else "Stopped. Everything read so far is kept: resume the queue to continue.")
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            state.error = f"{type(e).__name__}: {e}" if not isinstance(e, SystemExit) else str(e)
            import traceback
            traceback.print_exc()
        finally:
            state.running = False
            state.i = state.total = 0
            state.sampling = ""
    server.shutdown()
    server.server_close()
    if session["restart"]:
        # Start again in place: same command, same window, with whatever version is now installed.
        for old in _synth_cache.values():
            if hasattr(old, "close"):
                old.close()
        os.environ["BOOK2AUDIO_RESTARTED"] = session["restart"]
        print("Restarting book2audio...")
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable, *sys.argv])
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
