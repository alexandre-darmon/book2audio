# Roadmap

One goal: French audiobooks that sound read, made on your own Mac, with an install anyone can do.

## Done

- EPUB to chaptered audiobook (`.m4a` per chapter, `.m4b` with chapter markers), copy to iCloud Drive for the iPhone
- Natural French with Voxtral 4-bit, English with Kokoro, Pocket TTS on the CPU
- Queue of several books, kept on disk
- Resume at the exact passage after a stop, a crash or a reboot
- Progress per chapter, time left, estimate of the time to create a book
- "Compare models": same passage, each model, timed on your Mac
- Install in one command, with ffmpeg included and a double-click launcher

## Next

1. **Check the machine, recommend the models.** At first launch, read the Mac's chip and memory, then list the models that fit, with the expected speed of each, on CPU or GPU. Today the app assumes a recent Mac with plenty of memory.
2. **Proof by ear.** The same public-domain French paragraph read by each model, at the top of the README.
3. **Better French reading.** Numbers, abbreviations, dialogue dashes and guillemets, footnote marks.
4. **More voices.** Kyutai TTS 1.6B, built for French and English.

## Later

- **A Mac app.** A real `.app` to download, not signed by Apple at first: macOS will ask for a confirmation on first launch.
- **Faster French.** Any model that keeps Voxtral's quality at a higher speed.
- **Optional cloud voice.** Reading through a paid API for those who prefer speed to privacy. Off by default: the point of book2audio is that nothing leaves the Mac.

## Not planned

- Removing DRM. book2audio reads files you can already open.
- Rewriting the text. The book is read as written.
