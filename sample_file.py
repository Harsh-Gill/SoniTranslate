"""
=== COLAB BATCH TRANSLATE ===
Paste this entire block into a NEW Colab cell BELOW the running SoniTranslate cell.
Then run it. It calls the Gradio API locally (no tunnel needed).

BEFORE RUNNING:
1. Upload your videos to /content/ via Colab sidebar
2. Make sure Google Drive is mounted (this script will mount it if not)
"""

import os
import shutil
import time
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from gradio_client import Client, handle_file

# ============================================================
# CONFIG - Edit these if needed
# ============================================================

# Where your uploaded videos are on Colab
VIDEO_DIR = "/content/"

# Video filenames (must match exactly what you uploaded)
VIDEO_FILES = [
    "Best Practices for Electric Arc Welding On-board Ships (Part 1) 360 - WOS.mp4",
    "Best Practices for Gas Welding and Cutting On-board Ships (Part 1) 360 - WOS.mp4",
    "Best Practices for Gas Welding and Cutting On-board Ships (Part 2) 360 - WOS.mp4",
    "Line Management Plan implementation_WOS_360p.mp4",
]

# Output folder on Google Drive
DRIVE_OUTPUT = "/content/drive/MyDrive/marinepals_translations"

# 15 languages (no English - it's the source)
LANGUAGES = {
    "hi": "Hindi (hi)",
    "zh": "Chinese - Simplified (zh-CN)",
    "es": "Spanish (es)",
    "ja": "Japanese (ja)",
    "de": "German (de)",
    "fr": "French (fr)",
    "ko": "Korean (ko)",
    "pt": "Portuguese (pt)",
    "el": "Greek (el)",
    "id": "Indonesian (id)",
    "th": "Thai (th)",
    "vi": "Vietnamese (vi)",
    "ru": "Russian (ru)",
    "my": "Myanmar Burmese (my)",
    "ar": "Arabic (ar)",
}

# Male TTS voices
TTS_VOICES = {
    "hi": "hi-IN-MadhurNeural-Male",
    "zh": "zh-CN-YunxiNeural-Male",
    "es": "es-ES-AlvaroNeural-Male",
    "ja": "ja-JP-KeitaNeural-Male",
    "de": "de-DE-ConradNeural-Male",
    "fr": "fr-FR-HenriNeural-Male",
    "ko": "ko-KR-InJoonNeural-Male",
    "pt": "pt-BR-AntonioNeural-Male",
    "el": "el-GR-NestorasNeural-Male",
    "id": "id-ID-ArdiNeural-Male",
    "th": "th-TH-NiwatNeural-Male",
    "vi": "vi-VN-NamMinhNeural-Male",
    "ru": "ru-RU-DmitryNeural-Male",
    "my": "my-MM-ThihaNeural-Male",
    "ar": "ar-SA-HamedNeural-Male",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_duration_ms(filepath):
    """Get duration of a media file in milliseconds using ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    return float(json.loads(r.stdout)["format"]["duration"]) * 1000


def pad_audio_to_match(input_video, translated_mp3):
    """Pad translated audio to exactly match input video duration."""
    input_dur = get_duration_ms(input_video)
    output_dur = get_duration_ms(translated_mp3)
    diff_ms = input_dur - output_dur

    if abs(diff_ms) <= 5:
        print(f"      Duration already matches (diff: {diff_ms:.1f}ms)")
        return translated_mp3

    cutoff_sec = output_dur / 1000.0
    target_sec = input_dur / 1000.0
    print(f"      Padding: appending {diff_ms/1000:.2f}s of original audio (from {cutoff_sec:.3f}s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        padded_path = os.path.join(tmpdir, "padded.mp3")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", translated_mp3,
            "-ss", f"{cutoff_sec:.3f}", "-i", input_video,
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            "-map", "[out]",
            "-t", f"{target_sec:.6f}",
            "-acodec", "libmp3lame", "-q:a", "2",
            padded_path
        ], capture_output=True, check=True)
        shutil.move(padded_path, translated_mp3)

    final_dur = get_duration_ms(translated_mp3)
    final_diff = abs(final_dur - input_dur)
    print(f"      Padded: {final_dur/1000:.3f}s (diff: {final_diff:.1f}ms)")
    return translated_mp3


def ensure_video_exists(original_path, safe_dir="/content/videos_backup"):
    """Copy video to safe dir if not already there. Returns safe path."""
    os.makedirs(safe_dir, exist_ok=True)
    safe_path = os.path.join(safe_dir, os.path.basename(original_path))
    if not os.path.exists(safe_path):
        if os.path.exists(original_path):
            shutil.copy2(original_path, safe_path)
            print(f"      Backed up to {safe_path}")
        else:
            raise FileNotFoundError(f"Video not found: {original_path}")
    return safe_path


def translate_one(client, safe_path, lang_code, output_dir, max_retries=3):
    """Translate one video to one language via local Gradio API."""
    target_lang = LANGUAGES[lang_code]
    tts_voice = TTS_VOICES[lang_code]
    video_name = Path(safe_path).stem

    # Copy video to /content/ for SoniTranslate (it may delete it)
    working_copy = f"/content/{os.path.basename(safe_path)}"
    if not os.path.exists(working_copy):
        shutil.copy2(safe_path, working_copy)

    for attempt in range(1, max_retries + 1):
        try:
            # Re-copy if SoniTranslate deleted it
            if not os.path.exists(working_copy):
                shutil.copy2(safe_path, working_copy)

            print(f"      🔄 Processing...", flush=True)
            start = time.time()

            result = client.predict(
                [handle_file(working_copy)],               # video (list)
                "",                                        # media_link
                "",                                        # video_path
                "",                                        # hf_token
                "",                                        # preview
                "large-v3",                                # whisper_asr_model
                16,                                        # batch_size
                "float16",                                 # compute_type
                "English (en)",                            # source_language
                target_lang,                               # translate_audio_to
                1,                                         # min_speakers
                1,                                         # max_speakers
                tts_voice,                                 # tts_speaker_1
                tts_voice,                                 # tts_speaker_2
                tts_voice,                                 # tts_speaker_3
                tts_voice,                                 # tts_speaker_4
                tts_voice,                                 # tts_speaker_5
                tts_voice,                                 # tts_speaker_6
                tts_voice,                                 # tts_speaker_7
                tts_voice,                                 # tts_speaker_8
                tts_voice,                                 # tts_speaker_9
                tts_voice,                                 # tts_speaker_10
                tts_voice,                                 # tts_speaker_11
                tts_voice,                                 # tts_speaker_12
                video_name,                                # file_name
                "Mixing audio with sidechain compression", # audio_mixing_method
                1.4,                                       # max_audio_acceleration
                True,                                      # acceleration_rate_regulation
                1.2,                                       # volume_original_audio
                1.6,                                       # volume_translated_audio
                "srt",                                     # subtitle_type
                False,                                     # parameter_102
                False,                                     # edit_generated_subtitles
                "",                                        # generated_subtitles
                True,                                      # overlap_reduction
                False,                                     # sound_cleanup
                True,                                      # literalize_numbers
                15,                                        # segment_duration_limit
                "pyannote_3.1",                            # diarization_model
                "google_translator_batch",                 # translation_process
                None,                                      # upload_srt_file
                "audio (mp3)",                             # output_type
                True,                                      # voiceless_track
                False,                                     # active_voice_imitation
                5,                                         # max_samples
                False,                                     # dereverb
                False,                                     # remove_previous_samples
                "freevc",                                  # method
                True,                                      # dereverb_audio
                "sentence",                                # text_segmentation_scale
                "",                                        # redivide_text_segments_by
                True,                                      # soft_subtitles
                False,                                     # burn_subtitles
                False,                                     # retrieve_progress
                False,                                     # enable (batch)
                1,                                         # workers
                False,                                     # parameter_87
                api_name="/batch_multilingual_media_conversion_1"
            )

            elapsed = time.time() - start
            print(f"      ✅ API done in {elapsed:.1f}s", flush=True)

            if not isinstance(result, (list, tuple)):
                result = [result]

            saved_files = []
            for item in result:
                if not isinstance(item, str) or not os.path.exists(item):
                    continue
                ext = Path(item).suffix.lower()
                basename = os.path.basename(item)
                if '_origin' in basename:
                    continue

                if ext in ('.mp3', '.wav', '.m4a', '.ogg'):
                    dest = os.path.join(output_dir, f"{video_name}__{lang_code}{ext}")
                    shutil.copy2(item, dest)
                    try:
                        pad_audio_to_match(safe_path, dest)
                    except Exception as e:
                        print(f"      ⚠️  Padding failed: {e}")
                    saved_files.append(dest)
                elif ext == '.srt':
                    dest = os.path.join(output_dir, f"{video_name}__{lang_code}.srt")
                    shutil.copy2(item, dest)
                    saved_files.append(dest)

            return True, saved_files

        except Exception as e:
            print(f"      ❌ Attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(10)
            else:
                return False, []

    return False, []


# ============================================================
# MAIN
# ============================================================

# Mount Google Drive
if os.path.exists("/content/drive/MyDrive"):
    print("✅ Drive already mounted")
else:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("✅ Drive mounted")
    except Exception as e:
        print(f"⚠️  Could not mount Drive: {e}")
        print("   Saving outputs to /content/batch_outputs/ instead")
        DRIVE_OUTPUT = "/content/batch_outputs"

# Create output dir on Drive
os.makedirs(DRIVE_OUTPUT, exist_ok=True)

# Connect to local Gradio
print("\n📡 Connecting to local Gradio API...")
client = Client("http://127.0.0.1:7860")
print("✅ Connected!\n")

# Verify videos exist and back them up
videos = []
for fname in VIDEO_FILES:
    fpath = os.path.join(VIDEO_DIR, fname)
    if os.path.exists(fpath):
        safe = ensure_video_exists(fpath)
        videos.append((fname, safe))
        dur = get_duration_ms(safe)
        print(f"  ✅ {fname} ({dur/1000:.1f}s)")
    else:
        print(f"  ❌ NOT FOUND: {fpath}")

if not videos:
    raise RuntimeError("No videos found! Upload them to /content/ first.")

all_langs = list(LANGUAGES.keys())
total_jobs = len(videos) * len(all_langs)

print(f"\n{'#'*70}")
print(f"  BATCH TRANSLATION")
print(f"  Videos: {len(videos)} | Languages: {len(all_langs)} | Total: {total_jobs}")
print(f"  Output: {DRIVE_OUTPUT}")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'#'*70}\n")

results = {}
total_start = time.time()
job_num = 0

for vi, (fname, safe_path) in enumerate(videos):
    video_name = Path(safe_path).stem

    # Per-video output folder on Drive
    video_output = os.path.join(DRIVE_OUTPUT, video_name)
    os.makedirs(video_output, exist_ok=True)

    video_dur = get_duration_ms(safe_path)
    print(f"\n{'='*70}")
    print(f"  VIDEO [{vi+1}/{len(videos)}]: {fname}")
    print(f"  Duration: {video_dur/1000:.1f}s | Output: {video_output}")
    print(f"{'='*70}")

    for lang_code in all_langs:
        job_num += 1
        lang_name = LANGUAGES[lang_code]

        # Skip if already done
        expected_mp3 = os.path.join(video_output, f"{video_name}__{lang_code}.mp3")
        if os.path.exists(expected_mp3):
            print(f"\n  [{job_num}/{total_jobs}] {lang_name} — SKIPPED (already exists)")
            results[(video_name, lang_code)] = True
            continue

        print(f"\n  [{job_num}/{total_jobs}] {lang_name}")
        start = time.time()

        success, files = translate_one(client, safe_path, lang_code, video_output)
        elapsed = time.time() - start
        results[(video_name, lang_code)] = success

        if success:
            print(f"    ✅ Done in {elapsed:.1f}s — {len(files)} files saved")
        else:
            print(f"    ❌ Failed after {elapsed:.1f}s")

total_elapsed = time.time() - total_start
succeeded = sum(1 for v in results.values() if v)
failed = sum(1 for v in results.values() if not v)

print(f"\n\n{'#'*70}")
print(f"  BATCH COMPLETE")
print(f"  Time: {total_elapsed/60:.1f} minutes")
print(f"  ✅ Succeeded: {succeeded}/{total_jobs}")
print(f"  ❌ Failed: {failed}/{total_jobs}")
print(f"  Output: {DRIVE_OUTPUT}")
print(f"{'#'*70}")

if failed > 0:
    print("\n  Failed translations:")
    for (v, l), ok in results.items():
        if not ok:
            print(f"    - {v} → {LANGUAGES[l]}")
