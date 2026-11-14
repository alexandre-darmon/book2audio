# book2audio, in detail

The [README](../README.md) is the short version. This page holds the rest.

## Day to day

Copyright pages, tables of contents and the like are left unticked. The page tells you how long the audio is and about how long it will take to create with the chosen model.

Books are read one after the other, in the order of the list, chapters in order. The queue is saved on disk (`~/Audiobooks/.queue.json`) at every change.

The voice model downloads once, the first time you use it. After that, models are loaded from the copy on your Mac in `~/.cache/huggingface`, without contacting Hugging Face.

Audio files go to `~/Audiobooks/<book title>/`, from the page and from the command line alike (`-o` picks another folder).

While a book is being read, the page shows the percentage done and the time left, and the Mac is kept awake. Keep the lid open and the charger plugged in: closing the lid still puts it to sleep.

**Pause** stops after the current passage. You can pause at any time, and a crash or a reboot loses nothing: the queue is kept, and the position inside the current chapter is saved after every passage. At the next launch the page shows the queue as it was, and "Resume reading" continues at the exact passage where it stopped, with the same voice. A book that fails is marked as failed and the next one starts. The `.m4b` audiobook of a book is rebuilt at the end of each run with every chapter finished so far.

To remove it: `uv tool uninstall book2audio` and delete `~/Applications/book2audio.app`. Audio files stay in `~/Audiobooks`, models in `~/.cache/huggingface`.

At the top of the page, a green light shows that book2audio is on. The page follows your Mac's light or dark appearance. The circular arrow restarts book2audio, which is how you pick up a new version after an update: a book being read goes on where it stopped. The power button turns it off; to turn it on again, open it from your Applications folder and the page reconnects by itself. Starting it twice is harmless: the second time simply reopens the page.

## Languages and voices

The language is read from the book, and you can change it in the page. For each language, book2audio offers the models that speak it and fit your Mac.

| Model | Languages | Notes |
|---|---|---|
| Voxtral 4-bit (Mistral) | French, English, Spanish, German, Italian, Portuguese, Dutch, Hindi, Arabic | The default wherever it speaks the language. Natural, 2.7x real time on an M6. A full-size version is also offered: same voices, about 2.5x slower |
| Kokoro | English, French, Spanish, Italian, Portuguese, Hindi, Japanese, Chinese | Small and fast. The default for Japanese and Chinese |
| Pocket TTS (Kyutai) | French, English, German, Spanish, Italian, Portuguese | Runs on the CPU, for Macs with little memory |
| Kyutai TTS 1.6B | French, English | Experimental: not confirmed on a real Mac yet |

### Choosing a voice

Each model comes with several voices: 2 in French and 5 in English for Voxtral, 6 for Pocket TTS in each of its languages, 24 in English for Kokoro. Under each voice, a line says whether it is a woman or a man, and its tone or accent. The page lists them all. Press play next to a voice to hear it say the same two sentences as the others, so only the voice differs.

A sample that has been published with the project plays at once, without loading a model. Any other sample is created on your Mac the first time you ask for it, then kept in `~/Audiobooks/.voices`. "Create all samples" loads the model once and makes every missing one.

### Your own voice (experimental)

Pick "Your own voice" in the model list to have a book read with a voice you record: yours, or the voice of someone who agrees. Read the short text shown in the page for about 15 seconds, or choose a recording you already have, give it a name and save it. The recording stays on your Mac, in `~/Audiobooks/.voices/mine`, and can be deleted from the page.

This uses Chatterbox (MIT licence), which imitates the voice in the recording and speaks 23 languages. A quiet room matters more than a long recording: the model listens to the first 10 seconds. It is never picked automatically. This part has not been run on a real Mac yet.

Only record a voice with the agreement of the person it belongs to.

To make every sample from the command line, in French and English, for each model that fits your Mac:

```bash
book2audio --voice-samples
```

French is the language tested most. English and the others use the same models and the same code, but have had less listening.

Pocket TTS and Kyutai TTS install themselves in a separate environment the first time you pick them. Voxtral 4-bit is a 2.5 GB download (8 GB for the full size) and its licence is non-commercial (CC BY-NC 4.0): fine for listening to your own books, not for selling audio.

**Compare models.** The "Compare models" button reads the same one-minute passage with each model and shows, for each, the audio to listen to and its measured speed on your Mac. From the terminal: `book2audio book.epub --compare`.

All models can be picked for either language in the page. Voxtral 4-bit is a 2.5 GB download (8 GB for the full size) and its licence is non-commercial (CC BY-NC 4.0): fine for listening to your own books, not for selling audio.

## Benchmark

French, one minute of the same text, measured on a MacBook Pro M6 with 32 GB:

| Model | Speed | Characters per second | Runs on | Memory needed | Suits a Mac with |
|---|---|---|---|---|---|
| Voxtral 4-bit | 2.71x real time | 46.5 | GPU | about 3.5 GB | 8 GB if other apps are closed, 16 GB and up |
| Voxtral, full size | 1.05x | 18.7 | GPU | about 9 GB | 16 GB if other apps are closed, 24 GB and up |
| Pocket TTS | 5.38x | 104.1 | CPU | about 1.5 GB | 8 GB and up |

2x real time means one hour of audio takes half an hour to create. Memory figures are estimates.

On a MacBook Pro M1 with 8 GB, the smallest Mac it was tried on, book2audio picks the model itself. These are first runs on one short chapter each, model loading included, so longer chapters should do a little better:

| Book | Model picked | Speed |
|---|---|---|
| French | Pocket TTS | 0.6x to 0.8x real time |
| English | Kokoro | 2.4x |
| French, Voxtral 4-bit chosen by hand | Voxtral 4-bit | 0.4x |

Below 1x, a book takes longer to create than to listen to: a 10 hour French book needs 13 to 17 hours on that Mac, so let it run overnight. It can be stopped and resumed. Kyutai TTS is not measured yet.

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
