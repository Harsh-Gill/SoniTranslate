#!/usr/bin/env python3
"""
Batch Audio Translator — Process all files in a folder with the same settings.

Usage:
    python batch_translate.py ./my_videos/ --target all
    python batch_translate.py ./my_videos/ --target es
    python batch_translate.py ./lectures/ --target all --remove-vocals --max-speakers 2 --hf-token hf_xxx
"""

import argparse
import os
import sys
import time
from pathlib import Path

from translate_audio import translate_audio

# File extensions we'll try to process
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv",  # video
    ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma",  # audio
}

# 15 target languages (source is English)
ALL_LANGUAGES = [
    "hi",   # Hindi
    "zh",   # Chinese (Simplified)
    "es",   # Spanish
    "ja",   # Japanese
    "de",   # German
    "fr",   # French
    "ko",   # Korean
    "pt",   # Portuguese
    "el",   # Greek
    "id",   # Indonesian
    "th",   # Thai
    "vi",   # Vietnamese
    "ru",   # Russian
    "my",   # Myanmar (Burmese)
    "ar",   # Arabic
]

# Default male TTS voice for each language
DEFAULT_VOICES = {
    "hi": "hi-IN-MadhurNeural",
    "zh": "zh-CN-YunxiNeural",
    "es": "es-ES-AlvaroNeural",
    "ja": "ja-JP-KeitaNeural",
    "de": "de-DE-ConradNeural",
    "fr": "fr-FR-HenriNeural",
    "ko": "ko-KR-InJoonNeural",
    "pt": "pt-BR-AntonioNeural",
    "el": "el-GR-NestorasNeural",
    "id": "id-ID-ArdiNeural",
    "th": "th-TH-NiwatNeural",
    "vi": "vi-VN-NamMinhNeural",
    "ru": "ru-RU-DmitryNeural",
    "my": "my-MM-ThihaNeural",
    "ar": "ar-SA-HamedNeural",
}

# Default female TTS voice for each language
FEMALE_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "es": "es-ES-ElviraNeural",
    "ja": "ja-JP-NanamiNeural",
    "de": "de-DE-KatjaNeural",
    "fr": "fr-FR-DeniseNeural",
    "ko": "ko-KR-SunHiNeural",
    "pt": "pt-BR-FranciscaNeural",
    "el": "el-GR-AthinaNeural",
    "id": "id-ID-GadisNeural",
    "th": "th-TH-PremwadeeNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "my": "my-MM-NilarNeural",
    "ar": "ar-SA-ZariyahNeural",
}


def find_media_files(folder: str) -> list[str]:
    """Find all supported audio/video files in a folder (non-recursive)."""
    folder_path = Path(folder)
    files = []
    for f in sorted(folder_path.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(f))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Batch translate all audio/video files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  READY-TO-USE COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # MALE voice — all 15 languages
  python batch_translate.py ./male-voice --target all --source en --remove-vocals --model base

  # FEMALE voice — all 15 languages
  python batch_translate.py ./female-voice --target all --source en --remove-vocals --female --model base

  # MIXED voice (auto gender detection via pitch) — all 15 languages
  python batch_translate.py ./mixed-voice --target all --source en --remove-vocals --max-speakers 8 --hf-token YOUR_HF_TOKEN --model base

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

General examples:
  python batch_translate.py ./videos/ --target all                  # every language (male)
  python batch_translate.py ./videos/ --target es,fr,de,ja          # specific languages
  python batch_translate.py ./videos/ --target es                   # single language
  python batch_translate.py ./videos/ --target all --female         # female voices
  python batch_translate.py ./videos/ --target all --max-speakers 8 --hf-token hf_xxx  # auto gender detect
        """,
    )

    parser.add_argument(
        "folder",
        help="Folder containing audio/video files to translate",
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target language code (e.g. es, fr, ja, de, hi) or 'all' for every language",
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Source language code (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same folder as input files)",
    )
    parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--voice", "-v",
        default=None,
        help="Edge TTS voice override (default: uses per-language voice from DEFAULT_VOICES)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="Minimum number of speakers for diarization (default: 1)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=1,
        help="Max speakers for diarization (default: 1, set >1 to enable)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for pyannote diarization",
    )
    parser.add_argument(
        "--remove-vocals",
        action="store_true",
        help="Remove original vocals and keep only background music",
    )
    parser.add_argument(
        "--original-volume",
        type=float,
        default=0.15,
        help="Volume of original audio in mix (0.0-1.0, default: 0.15)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files for debugging",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search subfolders recursively",
    )
    parser.add_argument(
        "--female",
        action="store_true",
        help="Use female voices instead of male (per-language defaults)",
    )

    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{args.folder}' is not a directory.")
        sys.exit(1)

    # Find files
    if args.recursive:
        files = []
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(f))
    else:
        files = find_media_files(args.folder)

    if not files:
        print(f"No supported media files found in '{args.folder}'.")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(0)

    # Determine target languages
    if args.target.lower() == "all":
        target_langs = ALL_LANGUAGES
    else:
        # Support comma-separated: --target es,fr,de
        target_langs = [l.strip() for l in args.target.split(",") if l.strip()]

    # Output directory: <input-folder>-output/ by default
    if args.output_dir:
        base_out_dir = Path(args.output_dir)
    else:
        folder_str = str(folder).rstrip("/")
        base_out_dir = Path(f"{folder_str}-output")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Build the full job list: (file, language)
    jobs = [(f, lang) for f in files for lang in target_langs]

    # Skip source language if known
    if args.source:
        jobs = [(f, lang) for f, lang in jobs if lang != args.source]

    # Determine voice mode label
    if args.voice:
        voice_mode = f"Voice override: {args.voice}"
    elif args.female:
        voice_mode = "Female voices (FEMALE_VOICES)"
    else:
        voice_mode = "Male voices (DEFAULT_VOICES)"
    if args.max_speakers > 1:
        voice_mode += " + pitch-based gender detection"

    # Summary
    print("=" * 60)
    print(f"  Batch Translate — {len(files)} file(s) × {len(target_langs)} language(s) = {len(jobs)} job(s)")
    print(f"  Languages: {', '.join(target_langs)}")
    print(f"  Model: {args.model}")
    print(f"  Voices: {voice_mode}")
    if args.remove_vocals:
        print(f"  Vocal removal: ON")
    if args.max_speakers > 1:
        print(f"  Diarization: {args.min_speakers}-{args.max_speakers} speakers")
    print(f"  Output: {base_out_dir}/")
    print("=" * 60)

    print(f"\n  Files:")
    for i, f in enumerate(files, 1):
        print(f"    {i}. {Path(f).name}")
    print()

    # Process each job
    results = []
    total_start = time.time()

    for job_idx, (input_file, target_lang) in enumerate(jobs, 1):
        input_path = Path(input_file)

        # Create per-language output directory
        lang_dir = base_out_dir / target_lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(lang_dir / f"{input_path.stem}_{target_lang}.wav")

        print(f"\n{'━' * 60}")
        # Pick voice: CLI override > female > male default
        # When --max-speakers > 1, translate_audio handles diarization + pitch detection
        if args.voice:
            lang_voice = args.voice
        elif args.female:
            lang_voice = FEMALE_VOICES.get(target_lang)
        else:
            lang_voice = DEFAULT_VOICES.get(target_lang)

        mode_label = "female" if args.female else "male"
        print(f"  [{job_idx}/{len(jobs)}] {input_path.name} → {target_lang}  ({mode_label}: {lang_voice or 'auto'})")
        print(f"{'━' * 60}\n")

        file_start = time.time()
        try:
            result = translate_audio(
                input_file=input_file,
                target_lang=target_lang,
                source_lang=args.source,
                output_file=output_file,
                whisper_model=args.model,
                voice=lang_voice,
                original_volume=args.original_volume,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                hf_token=args.hf_token,
                remove_vocals=args.remove_vocals,
                keep_temp=args.keep_temp,
            )
            elapsed = time.time() - file_start
            results.append(("OK", input_path.name, target_lang, result, elapsed))
            print(f"\n  ✓ {target_lang} done in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - file_start
            results.append(("FAIL", input_path.name, target_lang, str(e), elapsed))
            print(f"\n  ✗ {target_lang} failed after {elapsed:.1f}s: {e}")

    # Final summary
    total_elapsed = time.time() - total_start
    ok = [r for r in results if r[0] == "OK"]
    failed = [r for r in results if r[0] == "FAIL"]

    print(f"\n\n{'=' * 60}")
    print(f"  Batch Complete — {len(ok)}/{len(results)} succeeded")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    if ok:
        print(f"\n  Succeeded ({len(ok)}):")
        for _, name, lang, out, t in ok:
            print(f"    ✓ {name} → {lang} ({t:.1f}s)")

    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for _, name, lang, err, t in failed:
            print(f"    ✗ {name} → {lang}: {err}")

    print()


if __name__ == "__main__":
    main()
