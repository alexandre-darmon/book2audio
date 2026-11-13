# book2audio

Turns an ebook into an audiobook, chapter by chapter, with a voice model that runs on your Mac. Nothing leaves the machine.

- Input: EPUB. AZW3 and MOBI work if Calibre is installed and the file has no DRM.
- Output: one `.m4a` per chapter, plus one `.m4b` audiobook with chapter markers.
- Languages: French and English out of the box, picked from the book.
- Needs an Apple Silicon Mac.

## Start

Double-click `Book2Audio.command`. A page opens in your browser:

1. Drop a book on the page.
2. Click "1-minute preview" to hear the voice.
3. Tick the chapters you want and click "Create audiobook".

The first launch installs what is missing (uv, Python, the libraries) and downloads the voice model once. After that, models are loaded from the copy on your Mac in `~/.cache/huggingface`, without contacting Hugging Face.

Audio files go to `~/Audiobooks/<book title>/`, from the page and from the command line alike (`-o` picks another folder).

While a book is being read, the page shows the percentage done and the time left, and the Mac is kept awake. Keep the lid open and the charger plugged in: closing the lid still puts it to sleep.

You can stop at any time, and a crash or a reboot loses nothing: the position is saved after every passage. When you reopen the same book, the page offers "Resume where I left off", which continues the interrupted chapter at the passage where it stopped, with the same voice. The `.m4b` audiobook is rebuilt at the end of each run with every chapter finished so far.

For the audiobook file you also need ffmpeg: `brew install ffmpeg`.

## Listen on the iPhone

Keep "Copy to iCloud Drive for the iPhone" ticked. The `.m4b` lands in `iCloud Drive/Audiobooks`. On the phone, open it from the Files app, or import it into an audiobook player such as BookPlayer, which keeps your position and shows the chapters.

## Voices

| Book language | Model | Voices |
|---|---|---|
| French | Voxtral (Mistral), natural, about real-time speed | `fr_female` (default), `fr_male` |
| English | Kokoro, small and fast | `af_heart` (default), `af_bella`, `am_michael`, `am_adam`, `bf_emma`, `bm_george` |

Both models can be picked for either language in the page. Voxtral is an 8 GB download and its licence is non-commercial (CC BY-NC 4.0): fine for listening to your own books, not for selling audio.

## Command line

Everything the page does is also available in the terminal:

```bash
uv run book2audio.py book.epub --list             # check the chapters
uv run book2audio.py book.epub --preview          # 1-minute sample
uv run book2audio.py book.epub --chapters 4-38 --iphone   # run it again to resume
```

Options: `--model kokoro|voxtral|<hugging face id>`, `--voice`, `--lang fr|en|en-gb`, `--speed 1.1`, `--format mp3`, `--no-titles`, `--no-m4b`, `--min-chars 500`, `-o <folder>`.

## How it works

One file, `book2audio.py`, in six steps: read the EPUB and its table of contents, cut the text into sentences-sized passages, read each passage with the model through [mlx-audio](https://github.com/Blaizzy/mlx-audio), add short silences between sentences and paragraphs, encode each chapter with ffmpeg, then join the chapters into one `.m4b`.

## Limits

- Books bought on Amazon or Apple are DRM-protected and cannot be read.
- Chapters come from the book's table of contents. A book that puts everything in one file gives one long chapter.
- Footnote numbers and tables are skipped.
- Only convert books you own, for your own listening.
