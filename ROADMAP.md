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
- Machine check: the app reads the Mac's chip and memory and recommends the models that fit
- Reading rules for French and English: abbreviations, Roman numerals, ordinals, clock times, dialogue dashes
- Kyutai TTS 1.6B as an experimental model
- `--demo`: built-in sample read by each model, with a benchmark file

## Next

1. **Finish the benchmark.** Audio samples for each model in the repo, speeds for Kokoro and Kyutai TTS, and results from Macs other than a 32 GB M6. Run `book2audio --demo` and share the BENCHMARK file.
2. **Confirm Kyutai TTS on a real Mac**, then decide whether it replaces Voxtral 4-bit for French.
3. **English as good as French.** Compare Kokoro and Voxtral 4-bit by ear on English books.
4. **Reading rules, round two.** Numbers and dates in full, foreign words, names.

## Later

- **A Mac app.** A real `.app` to download, not signed by Apple at first: macOS will ask for a confirmation on first launch.
- **Faster French.** Any model that keeps Voxtral's quality at a higher speed.

## Not planned

- Removing DRM. book2audio reads files you can already open.
- Rewriting the text. The book is read as written.
- A paid cloud voice. At API prices, a long book costs about as much as its professionally narrated audiobook. Reading locally is free once the Mac is yours, and that is the point.
