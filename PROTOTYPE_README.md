# translate_audio.py — Single-Script Audio Translator

A standalone prototype that translates audio/video from one language to another using a fully automated pipeline. Built on top of the [SoniTranslate](https://github.com/r3gm/SoniTranslate) project.

## Pipeline

```
Input (mp3/mp4/wav/...)
  │
  ├─ 1.  Extract audio (ffmpeg → WAV)
  ├─ 1.5 Separate vocals from background (Demucs, optional)
  ├─ 2.  Transcribe speech (WhisperX)
  ├─ 2.5 Diarize speakers (pyannote, optional)
  ├─ 3.  Translate text (Google Translate)
  ├─ 3.5 Detect speaker genders (librosa pitch analysis)
  ├─ 4.  Text-to-Speech (Edge TTS, gender-matched voices)
  ├─ 5.  Assemble TTS on timeline
  ├─ 6.  Mix with original/background audio
  ├─ 7.  Enforce exact input duration (ms-accurate)
  └─ 8.  Export SRT subtitles
  │
  ▼
Output audio + .srt files
```

## Features

- **Automatic language detection** — source language is auto-detected if not specified
- **Speaker diarization** — identifies multiple speakers and assigns distinct voices
- **Gender-matched TTS** — auto-detects speaker gender via pitch analysis and picks appropriate voices
- **Vocal separation** — removes original vocals while preserving background music (Demucs)
- **Smart speech rate** — Edge TTS `rate` parameter adjusts speed naturally to fit timing windows
- **SRT subtitles** — exports both translated and original-language `.srt` files
- **Exact duration matching** — output is padded/trimmed to match the input duration to the millisecond
- **macOS MPS acceleration** — Demucs runs on Apple Silicon GPU automatically

## Requirements

- **Python 3.10+**
- **ffmpeg** installed and on PATH (`brew install ffmpeg` on macOS)
- **HuggingFace token** (only for diarization) — needs accepted access to:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`

Install Python dependencies:

```bash
pip install -r requirements_prototype.txt
```

## Usage

### Basic translation

```bash
python translate_audio.py input.mp4 --target es
```

### Specify source language + voice

```bash
python translate_audio.py lecture.wav --source en --target fr --voice fr-FR-DeniseNeural
```

### Multi-speaker with diarization

```bash
python translate_audio.py interview.mp3 --target de --max-speakers 3 --hf-token hf_xxxxx
```

### Remove vocals, keep background music

```bash
python translate_audio.py video.mp4 --target hi --remove-vocals --hf-token hf_xxxxx
```

### Custom output path + larger Whisper model

```bash
python translate_audio.py podcast.mp3 --target ja --model medium --output dubbed.wav
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `input` | *(required)* | Input audio or video file |
| `--target`, `-t` | *(required)* | Target language code (`es`, `fr`, `ja`, `hi`, …) |
| `--source`, `-s` | auto-detect | Source language code |
| `--output`, `-o` | `<input>_translated_<lang>.wav` | Output file path |
| `--model`, `-m` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`) |
| `--voice`, `-v` | auto-select | Edge TTS voice name (e.g. `en-US-GuyNeural`) |
| `--min-speakers` | `1` | Minimum speakers for diarization |
| `--max-speakers` | `1` | Max speakers (set >1 to enable diarization) |
| `--hf-token` | `$HF_TOKEN` env var | HuggingFace token for pyannote models |
| `--remove-vocals` | off | Strip original vocals, keep background audio |
| `--original-volume` | `0.15` | Volume of original audio in final mix (0.0–1.0) |
| `--keep-temp` | off | Keep temp directory for debugging |

## Output Files

For an input `video.mp4 --target hi`, the script produces:

| File | Contents |
|------|----------|
| `video_translated_hi.wav` | Translated audio (same duration as input) |
| `video_translated_hi.srt` | Translated subtitles |
| `video_translated_hi_original.srt` | Original-language subtitles |

## Supported Languages (common)

`en` English, `es` Spanish, `fr` French, `de` German, `it` Italian, `pt` Portuguese, `ja` Japanese, `ko` Korean, `zh` Chinese, `ru` Russian, `ar` Arabic, `hi` Hindi, `nl` Dutch, `pl` Polish, `tr` Turkish, and [many more](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support).

## Device Usage

| Component | Device | Notes |
|-----------|--------|-------|
| Demucs (vocal separation) | **MPS** (Apple Silicon) | Auto-detected, falls back to CPU |
| WhisperX (transcription) | CPU | CTranslate2 backend doesn't support MPS |
| pyannote (diarization) | CPU | PyTorch model, could use MPS in future |
| Edge TTS | Cloud | Microsoft servers |
| Gender detection (librosa) | CPU | NumPy-based |
