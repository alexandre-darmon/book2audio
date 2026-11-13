<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/book2audio-mark-dark.svg">
    <img src="assets/book2audio-mark-light.svg" width="72" alt="book2audio">
  </picture>
</p>

# book2audio

**Turn an EPUB into a chaptered audiobook on your Mac, with a voice model that runs locally.**

French and English. No account, no API key, nothing uploaded. Listen on your iPhone.

## Why book2audio?

- **French that sounds read, not synthesized.** French books use Voxtral, English books use Kokoro. You pick the voice and hear a 1-minute preview first.
- **It uses the Mac's GPU.** Models run through MLX on Apple Silicon. On a MacBook Pro M6 with 32 GB, Voxtral reads at about real-time speed: 9 minutes of audio in 9 minutes.
- **A 28-hour book survives interruptions.** The position is saved after every passage. Stop, crash or reboot, then resume at the same sentence.
- **Queue several books and leave.** They are read one after the other. The queue is kept on disk.

Input: EPUB. AZW3 and MOBI work if Calibre is installed and the file has no DRM.
Output: one `.m4a` per chapter, plus one `.m4b` audiobook with chapter markers.

## Start

Double-click `Book2Audio.command`. A page opens in your browser:

1. Drop a book on the page.
2. Click "1-minute preview" to hear the voice.
3. Tick the chapters you want and click "Add to the queue".
4. Add more books if you like, then click "Start the queue".

Books are read one after the other, in the order of the list, chapters in order. The queue is saved on disk (`~/Audiobooks/.queue.json`) at every change.

The first launch installs what is missing (uv, Python, the libraries) and downloads the voice model once. After that, models are loaded from the copy on your Mac in `~/.cache/huggingface`, without contacting Hugging Face.

Audio files go to `~/Audiobooks/<book title>/`, from the page and from the command line alike (`-o` picks another folder).

While a book is being read, the page shows the percentage done and the time left, and the Mac is kept awake. Keep the lid open and the charger plugged in: closing the lid still puts it to sleep.

You can stop at any time, and a crash or a reboot loses nothing: the queue is kept, and the position inside the current chapter is saved after every passage. At the next launch the page shows the queue as it was, and "Resume the queue" continues at the exact passage where it stopped, with the same voice. A book that fails is marked as failed and the next one starts. The `.m4b` audiobook of a book is rebuilt at the end of each run with every chapter finished so far.

For the audiobook file you also need ffmpeg: `brew install ffmpeg`.

## Listen on the iPhone

Keep "Copy to iCloud Drive for the iPhone" ticked. The `.m4b` lands in `iCloud Drive/Audiobooks`. On the phone, open it from the Files app, or import it into an audiobook player such as BookPlayer, which keeps your position and shows the chapters.

## Voices

| Book language | Model | Voices |
|---|---|---|
| French | Voxtral (Mistral), natural, about real-time speed. A compressed 4-bit version is also offered, to compare speed and quality | `fr_female` (default), `fr_male` |
| English | Kokoro, small and fast | `af_heart` (default), `af_bella`, `am_michael`, `am_adam`, `bf_emma`, `bm_george` |

Both models can be picked for either language in the page. Voxtral is an 8 GB download and its licence is non-commercial (CC BY-NC 4.0): fine for listening to your own books, not for selling audio.

## Command line

Everything the page does is also available in the terminal:

```bash
uv run book2audio.py book.epub --list             # check the chapters
uv run book2audio.py book.epub --preview          # 1-minute sample
uv run book2audio.py book.epub --chapters 4-38 --iphone   # run it again to resume
```

Options: `--model kokoro|voxtral|voxtral-4bit|<hugging face id>`, `--voice`, `--lang fr|en|en-gb`, `--speed 1.1`, `--format mp3`, `--no-titles`, `--no-m4b`, `--min-chars 500`, `-o <folder>`.

## How it works

One file, `book2audio.py`, in six steps: read the EPUB and its table of contents, cut the text into sentences-sized passages, read each passage with the model through [mlx-audio](https://github.com/Blaizzy/mlx-audio), add short silences between sentences and paragraphs, encode each chapter with ffmpeg, then join the chapters into one `.m4b`.

## Limits

- Books bought on Amazon or Apple are DRM-protected and cannot be read.
- Chapters come from the book's table of contents. A book that puts everything in one file gives one long chapter.
- Footnote numbers and tables are skipped.
- Only convert books you own, for your own listening.
