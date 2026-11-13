<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/book2audio-mark-dark.svg">
    <img src="assets/book2audio-mark-light.svg" width="72" alt="book2audio">
  </picture>
</p>

# book2audio

**Turn an ebook into an audiobook that sounds read, in French or English, on any Apple Silicon Mac.**

book2audio checks your Mac and picks the voice model that fits it. A bigger Mac gets a more natural voice, a smaller one takes a lighter model. No account, no API key, nothing uploaded. Listen on your iPhone.

## Why book2audio?

- **French that sounds read, not synthesized.** French books use Voxtral (4-bit) when the Mac has room for it, English books use Kokoro. You pick the voice and hear a 1-minute preview first.
- **It uses the Mac's GPU.** Models run through MLX on Apple Silicon. On a MacBook Pro M6 with 32 GB, the default French model reads 2.7 times faster than real time: 82 seconds of audio in 30 seconds.
- **A 28-hour book survives interruptions.** The position is saved after every passage. Stop, crash or reboot, then resume at the same sentence.
- **Queue several books and leave.** They are read one after the other. The queue is kept on disk.

Input: EPUB. AZW3 and MOBI work if Calibre is installed and the file has no DRM.
Output: one `.m4a` per chapter, plus one `.m4b` audiobook with chapter markers.

## Install

One command, in the Terminal:

```bash
curl -LsSf https://raw.githubusercontent.com/alexandre-darmon/book2audio/main/install.sh | sh
```

It installs everything book2audio needs, including Python and ffmpeg, without touching the rest of your Mac. Run the same command again to update.

Then double-click **book2audio** in the Applications folder of your home folder, or type `book2audio` in a new Terminal window.

A page opens in your browser:

1. Drop a book on the page.
2. Click "1-minute preview" to hear the voice.
3. Tick the chapters you want and click "Add to the queue".
4. Add more books if you like, then click "Start the queue".

Copyright pages, tables of contents and the like are left unticked. The page tells you how long the audio is and about how long it will take to create with the chosen model.

Books are read one after the other, in the order of the list, chapters in order. The queue is saved on disk (`~/Audiobooks/.queue.json`) at every change.

The voice model downloads once, the first time you use it. After that, models are loaded from the copy on your Mac in `~/.cache/huggingface`, without contacting Hugging Face.

Audio files go to `~/Audiobooks/<book title>/`, from the page and from the command line alike (`-o` picks another folder).

While a book is being read, the page shows the percentage done and the time left, and the Mac is kept awake. Keep the lid open and the charger plugged in: closing the lid still puts it to sleep.

You can stop at any time, and a crash or a reboot loses nothing: the queue is kept, and the position inside the current chapter is saved after every passage. At the next launch the page shows the queue as it was, and "Resume the queue" continues at the exact passage where it stopped, with the same voice. A book that fails is marked as failed and the next one starts. The `.m4b` audiobook of a book is rebuilt at the end of each run with every chapter finished so far.

To remove it: `uv tool uninstall book2audio` and delete `~/Applications/book2audio.app`. Audio files stay in `~/Audiobooks`, models in `~/.cache/huggingface`.

To stop it, click **Quit** at the top of the page. Starting it twice is harmless: the second time simply reopens the page.

## Listen on the iPhone

Keep "Copy to iCloud Drive for the iPhone" ticked. The `.m4b` lands in `iCloud Drive/Audiobooks`. On the phone, open it from the Files app, or import it into an audiobook player such as BookPlayer, which keeps your position and shows the chapters.

## Voices

| Book language | Model | Voices |
|---|---|---|
| French | Voxtral 4-bit (Mistral), natural, 2.7x real time. The full-size model is also offered: same voices, about 2.5x slower | `fr_female` (default), `fr_male` |
| English | Kokoro, small and fast | `af_heart` (default), `af_bella`, `am_michael`, `am_adam`, `bf_emma`, `bm_george` |

Two more options install themselves in a separate environment the first time you pick them: Pocket TTS (Kyutai), which runs on the CPU, and Kyutai TTS 1.6B, built for French and English. Kyutai TTS is experimental: it has not been confirmed on a real Mac yet.

**Compare models.** The "Compare models" button reads the same one-minute passage with each model and shows, for each, the audio to listen to and its measured speed on your Mac. From the terminal: `book2audio book.epub --compare`.

All models can be picked for either language in the page. Voxtral 4-bit is a 2.5 GB download (8 GB for the full size) and its licence is non-commercial (CC BY-NC 4.0): fine for listening to your own books, not for selling audio.

## Command line

Everything the page does is also available in the terminal. If you cloned the repo instead of installing, replace `book2audio` with `uv run book2audio.py`, or double-click `Book2Audio.command`.

```bash
book2audio book.epub --list             # check the chapters
book2audio book.epub --preview          # 1-minute sample
book2audio book.epub --chapters 4-38 --iphone   # run it again to resume
```

Options: `--model kokoro|voxtral|voxtral-4bit|pocket|kyutai|<hugging face id>`, `--voice`, `--lang fr|en|en-gb`, `--speed 1.1`, `--format mp3`, `--no-titles`, `--no-m4b`, `--min-chars 500`, `-o <folder>`.

## How it works

One file, `book2audio.py`, in six steps: read the EPUB and its table of contents, cut the text into sentences-sized passages, read each passage with the model through [mlx-audio](https://github.com/Blaizzy/mlx-audio), add short silences between sentences and paragraphs, encode each chapter with ffmpeg, then join the chapters into one `.m4b`.

## Benchmark

French, one minute of the same text, measured on a MacBook Pro M6 with 32 GB:

| Model | Speed | Characters per second | Runs on | Memory needed | Suits a Mac with |
|---|---|---|---|---|---|
| Voxtral 4-bit | 2.71x real time | 46.5 | GPU | about 3.5 GB | 8 GB if other apps are closed, 16 GB and up |
| Voxtral, full size | 1.05x | 18.7 | GPU | about 9 GB | 16 GB if other apps are closed, 24 GB and up |
| Pocket TTS | 5.38x | 104.1 | CPU | about 1.5 GB | 8 GB and up |

2x real time means one hour of audio takes half an hour to create. Memory figures are estimates. Kyutai TTS and Kokoro are not measured yet.

Run it on your own Mac:

```bash
book2audio --check    # your chip, your memory, and the models that suit them
book2audio --demo     # reads a short built-in text with each model, writes the audio and a BENCHMARK file
```

## How it reads

The book is read as written. A few things are said the way a narrator would say them, not the way they are printed:

- Abbreviations: "M." becomes "Monsieur", "Mme" becomes "Madame", "etc." becomes "et cetera".
- Roman numerals and ordinals: "Chapitre XII", "Louis XIV", "XIXe siècle", "1er", "3e".
- Clock times: "7 h 12".
- Dialogue dashes are not pronounced, and a dash inside a sentence becomes a short breath.
- A scene break ("* * *") becomes a longer silence. Footnote marks are skipped.

`--plain` turns all of this off.

## Roadmap

See [ROADMAP.md](ROADMAP.md). Next up: checking the machine and recommending the models that fit it.

## Limits

- Books bought on Amazon or Apple are DRM-protected and cannot be read.
- Chapters come from the book's table of contents. A book that puts everything in one file gives one long chapter.
- Footnote numbers and tables are skipped.
- Only convert books you own, for your own listening.
