#!/usr/bin/env python3
"""
SoniTranslate Prototype — Single-script audio translator.

Pipeline: Audio → Transcribe (WhisperX) → Translate (Google) → TTS (Edge TTS) → Mix → Output

Usage:
    python translate_audio.py input.mp3 --target es
    python translate_audio.py input.wav --source en --target fr --voice fr-FR-DeniseNeural
    python translate_audio.py input.mp4 --target ja --output translated.wav

Requirements:
    pip install whisperx torch deep-translator edge-tts pydub soundfile numpy tqdm

    Also requires ffmpeg installed on your system.
"""

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import gc
import math

import edge_tts
import nest_asyncio
import numpy as np
import soundfile as sf
import torch
import whisperx
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from tqdm import tqdm

# Allow nested event loops (needed for repeated asyncio.run calls on macOS)
nest_asyncio.apply()


# ─── Helpers ────────────────────────────────────────────────────────────────

def run_cmd(cmd: str, check: bool = True):
    """Run a shell command, raise on failure if check=True."""
    result = subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {cmd}\n"
            f"{result.stderr.decode()}"
        )
    return result


def get_media_duration_ms(path: str) -> int:
    """Get the exact duration of a media file in milliseconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    duration_sec = float(result.stdout.decode().strip())
    return int(round(duration_sec * 1000))


def extract_audio(input_file: str, output_wav: str):
    """Extract/convert any media file to 16-bit 44.1kHz stereo WAV via ffmpeg."""
    print(f"[1/6] Extracting audio from: {input_file}")
    run_cmd(
        f'ffmpeg -y -i "{input_file}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{output_wav}"'
    )
    if not os.path.exists(output_wav):
        raise FileNotFoundError(f"Failed to extract audio: {output_wav}")


# ─── Transcription ──────────────────────────────────────────────────────────

def transcribe(
    audio_wav: str,
    source_lang: str | None = None,
    model_name: str = "base",
    device: str | None = None,
    batch_size: int = 8,
) -> dict:
    """
    Transcribe audio using WhisperX.

    Returns dict with keys: 'segments' (list of {text, start, end}), 'language'.
    """
    print(f"[2/6] Transcribing audio (model={model_name})...")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=source_lang,
    )

    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(audio, batch_size=batch_size)

    detected_lang = result.get("language", source_lang or "en")
    print(f"    Detected language: {detected_lang}")
    print(f"    Found {len(result['segments'])} segments")

    # Align for better timestamps
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(f"    Warning: Alignment failed ({e}), using raw timestamps")

    # Clean up GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"segments": result["segments"], "language": detected_lang}


# ─── Diarization ────────────────────────────────────────────────────────────

def diarize(
    audio_wav: str,
    result: dict,
    min_speakers: int = 1,
    max_speakers: int = 2,
    hf_token: str | None = None,
    device: str | None = None,
) -> dict:
    """
    Assign speaker labels to transcription segments using pyannote.

    Requires a HuggingFace token with access to:
      - pyannote/speaker-diarization-3.1
      - pyannote/segmentation-3.0

    If max_speakers <= 1, skips diarization and labels everything SPEAKER_00.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if max_speakers > 1:
        if not hf_token:
            print("    Warning: No HuggingFace token provided. Diarization requires")
            print("    access to pyannote models. Set --hf-token or HF_TOKEN env var.")
            print("    Falling back to single-speaker mode.")
            return _assign_single_speaker(result)

        print(f"[2.5/6] Diarizing speakers (min={min_speakers}, max={max_speakers})...")
        try:
            from whisperx.diarize import DiarizationPipeline

            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                token=hf_token,
                device=device,
            )

            audio = whisperx.load_audio(audio_wav)
            diarize_segments = diarize_model(
                audio, min_speakers=min_speakers, max_speakers=max_speakers
            )

            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Ensure every segment has a speaker label
            for seg in result["segments"]:
                if "speaker" not in seg:
                    seg["speaker"] = "SPEAKER_00"

            # Re-encode speakers as SPEAKER_00, SPEAKER_01, ...
            result = _reencode_speakers(result)

            speakers = set(seg["speaker"] for seg in result["segments"])
            print(f"    Found {len(speakers)} speaker(s): {', '.join(sorted(speakers))}")

            del diarize_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            print(f"    Warning: Diarization failed ({e}), using single speaker")
            return _assign_single_speaker(result)
    else:
        return _assign_single_speaker(result)


def _assign_single_speaker(result: dict) -> dict:
    """Label all segments as SPEAKER_00."""
    for seg in result["segments"]:
        seg["speaker"] = "SPEAKER_00"
    return result


def _reencode_speakers(result: dict) -> dict:
    """Normalize speaker labels to SPEAKER_00, SPEAKER_01, etc."""
    first = result["segments"][0].get("speaker", "")
    if first == "SPEAKER_00":
        return result

    mapping = {}
    counter = 0
    for seg in result["segments"]:
        old = seg["speaker"]
        if old not in mapping:
            mapping[old] = f"SPEAKER_{counter:02d}"
            counter += 1
        seg["speaker"] = mapping[old]
    return result


# ─── Translation ────────────────────────────────────────────────────────────

def translate_segments(
    segments: list[dict],
    target_lang: str,
    source_lang: str | None = None,
) -> list[dict]:
    """
    Translate segment texts using Google Translate.

    Tries batch translation first (joining with |||||), falls back to
    one-by-one if the batch result doesn't split correctly.
    """
    print(f"[3/6] Translating {len(segments)} segments → {target_lang}...")

    src = source_lang if source_lang else "auto"
    translator = GoogleTranslator(source=src, target=target_lang)

    translated = []
    for seg in tqdm(segments, desc="Translating"):
        seg_copy = dict(seg)
        text = seg_copy.get("text", "").strip()
        if not text:
            translated.append(seg_copy)
            continue
        try:
            seg_copy["text"] = translator.translate(text)
        except Exception as e:
            print(f"    Warning: Translation failed for '{text[:40]}...': {e}")
            # Keep original text on failure
        translated.append(seg_copy)

    return translated


# ─── Text-to-Speech ─────────────────────────────────────────────────────────

def detect_speaker_genders(
    audio_wav: str,
    segments: list[dict],
) -> dict[str, str]:
    """
    Detect gender of each speaker using pitch (F0) analysis.

    Extracts audio for each speaker's segments, computes median fundamental
    frequency, and classifies:
      - Median F0 < 165 Hz → "Male"
      - Median F0 >= 165 Hz → "Female"

    Returns dict mapping speaker labels to "Male" or "Female".
    """
    import librosa

    print("[3.5/6] Detecting speaker genders (pitch analysis)...")

    # Load full audio
    y, sr = librosa.load(audio_wav, sr=22050, mono=True)

    # Group segments by speaker
    speaker_segments: dict[str, list[dict]] = {}
    for seg in segments:
        spk = seg.get("speaker", "SPEAKER_00")
        if spk not in speaker_segments:
            speaker_segments[spk] = []
        speaker_segments[spk].append(seg)

    genders = {}
    for spk, segs in sorted(speaker_segments.items()):
        # Concatenate audio from this speaker's segments
        speaker_audio = []
        for seg in segs:
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            speaker_audio.append(y[start_sample:end_sample])

        if not speaker_audio:
            genders[spk] = "Male"  # default
            continue

        combined = np.concatenate(speaker_audio)

        # Extract F0 (fundamental frequency)
        try:
            f0, voiced_flag, _ = librosa.pyin(
                combined,
                fmin=librosa.note_to_hz('C2'),   # ~65 Hz
                fmax=librosa.note_to_hz('C6'),   # ~1047 Hz
                sr=sr,
            )
            # Filter to only voiced frames
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]

            if len(voiced_f0) > 0:
                median_f0 = float(np.median(voiced_f0))
                gender = "Female" if median_f0 >= 165.0 else "Male"
                print(f"    {spk}: median F0 = {median_f0:.0f} Hz → {gender}")
            else:
                gender = "Male"  # default if no voiced frames
                print(f"    {spk}: no voiced frames detected → {gender} (default)")
        except Exception as e:
            gender = "Male"
            print(f"    {spk}: pitch analysis failed ({e}) → {gender} (default)")

        genders[spk] = gender

    return genders


def get_edge_voices_by_gender(
    target_lang: str,
) -> dict[str, list[str]]:
    """
    Get Edge TTS voices for a language, separated by gender.

    Returns {"Male": [...], "Female": [...]}.
    """
    voices = asyncio.run(edge_tts.list_voices())
    lang_prefix = target_lang.split("-")[0]

    matching = [
        v
        for v in voices
        if v["ShortName"].lower().startswith(lang_prefix.lower() + "-")
    ]

    result = {"Male": [], "Female": []}
    for v in matching:
        g = v.get("Gender", "Male")
        if g in result:
            result[g].append(v["ShortName"])

    # Fallbacks if a gender has no voices for this language
    if not result["Male"]:
        result["Male"] = ["en-US-GuyNeural"]
    if not result["Female"]:
        result["Female"] = ["en-US-JennyNeural"]

    return result


def get_edge_voices_for_lang(target_lang: str) -> list[str]:
    """
    Get all available Edge TTS voices for a language, sorted.
    Tries to return a mix of Male and Female voices.
    """
    voices = asyncio.run(edge_tts.list_voices())
    lang_prefix = target_lang.split("-")[0]

    matching = [
        v
        for v in voices
        if v["ShortName"].lower().startswith(lang_prefix.lower() + "-")
    ]

    if not matching:
        return ["en-US-GuyNeural", "en-US-JennyNeural"]

    # Sort so we alternate genders for multi-speaker variety
    males = [v["ShortName"] for v in matching if v.get("Gender") == "Male"]
    females = [v["ShortName"] for v in matching if v.get("Gender") == "Female"]

    # Interleave: male, female, male, female...
    interleaved = []
    for m, f in zip(males, females):
        interleaved.extend([m, f])
    interleaved.extend(males[len(females):])
    interleaved.extend(females[len(males):])

    return interleaved if interleaved else [matching[0]["ShortName"]]


def assign_voices_to_speakers(
    segments: list[dict],
    target_lang: str,
    speaker_genders: dict[str, str] | None = None,
    voice_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Assign an Edge TTS voice to each speaker found in segments.

    When speaker_genders is provided, picks a voice matching the
    detected gender for each speaker.

    Args:
        segments:         List of dicts with 'speaker' key.
        target_lang:      Target language code.
        speaker_genders:  Optional {speaker: "Male"/"Female"} from gender detection.
        voice_map:        Optional explicit {speaker: voice} override.

    Returns:
        Dict mapping speaker labels to Edge TTS voice names.
    """
    speakers = sorted(set(seg.get("speaker", "SPEAKER_00") for seg in segments))

    if voice_map:
        # Fill in any missing speakers with auto-selected voices
        available = get_edge_voices_for_lang(target_lang)
        for i, spk in enumerate(speakers):
            if spk not in voice_map:
                voice_map[spk] = available[i % len(available)]
        return voice_map

    if speaker_genders:
        # Gender-aware voice assignment
        voices_by_gender = get_edge_voices_by_gender(target_lang)
        male_idx = 0
        female_idx = 0
        mapping = {}

        for spk in speakers:
            gender = speaker_genders.get(spk, "Male")
            pool = voices_by_gender[gender]
            if gender == "Male":
                mapping[spk] = pool[male_idx % len(pool)]
                male_idx += 1
            else:
                mapping[spk] = pool[female_idx % len(pool)]
                female_idx += 1

        for spk, voice in mapping.items():
            g = speaker_genders.get(spk, "?")
            print(f"    {spk} ({g}) → {voice}")

        return mapping

    # Fallback: alternate voices without gender info
    available = get_edge_voices_for_lang(target_lang)
    mapping = {}
    for i, spk in enumerate(speakers):
        mapping[spk] = available[i % len(available)]

    for spk, voice in mapping.items():
        print(f"    {spk} → {voice}")

    return mapping


async def _generate_tts_segment(
    text: str, voice: str, output_path: str, rate: str = "+0%"
):
    """Generate a single TTS audio file using Edge TTS.

    Args:
        text:        Text to synthesise.
        voice:       Edge TTS voice name.
        output_path: Destination file path.
        rate:        Speech rate adjustment, e.g. "+15%", "-10%", "+0%".
    """
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)


def _estimate_tts_rate(text: str, available_sec: float) -> str:
    """
    Estimate the Edge TTS rate parameter needed to fit *text* into
    *available_sec* seconds.

    Heuristic: Edge TTS at +0% speaks roughly 15 chars/sec for Latin
    scripts and ~6 chars/sec for CJK.  We estimate the natural duration
    and, if it exceeds the window, request a speed-up — capped at +30 %
    to keep the output sounding natural.

    Returns an Edge TTS rate string like "+0%", "+15%", "+30%".
    """
    if available_sec <= 0.3 or not text.strip():
        return "+0%"

    # Rough chars-per-second at normal rate (Edge TTS default)
    avg_cps = 14.0  # works for most Latin-script languages
    estimated_sec = len(text) / avg_cps

    if estimated_sec <= available_sec:
        return "+0%"  # fits naturally, no boost needed

    # How much faster do we need?  ratio > 1 means we're over
    ratio = estimated_sec / available_sec
    # Convert to percentage boost, cap at 30 %
    boost_pct = min(int((ratio - 1.0) * 100), 30)
    if boost_pct <= 2:
        return "+0%"

    return f"+{boost_pct}%"


def generate_tts(
    segments: list[dict],
    target_lang: str,
    audio_dir: str,
    voice: str | None = None,
    voice_map: dict[str, str] | None = None,
) -> list[str]:
    """
    Generate TTS audio for each translated segment.

    Uses per-speaker voice assignments when diarization is active.
    Falls back to a single voice if voice_map is not provided.

    Returns list of audio file paths, one per segment.
    """
    print(f"[4/6] Generating TTS audio...")

    if voice_map is None:
        # Single speaker mode: build a trivial map
        if voice:
            clean = voice.replace("-Male", "").replace("-Female", "")
            selected_voice = clean
        else:
            available = get_edge_voices_for_lang(target_lang)
            selected_voice = available[0]
            print(f"    Auto-selected voice: {selected_voice}")
        voice_map = {"SPEAKER_00": selected_voice}
        print(f"    Using voice: {selected_voice}")

    os.makedirs(audio_dir, exist_ok=True)
    audio_files = []

    for i, seg in enumerate(tqdm(segments, desc="TTS")):
        text = seg.get("text", "").strip()
        out_mp3 = os.path.join(audio_dir, f"{i:04d}.mp3")
        out_ogg = os.path.join(audio_dir, f"{i:04d}.ogg")

        if not text:
            # Generate silence for empty segments
            duration_ms = int((seg.get("end", 0) - seg.get("start", 0)) * 1000)
            silence = AudioSegment.silent(duration=max(duration_ms, 100))
            silence.export(out_ogg, format="ogg")
            audio_files.append(out_ogg)
            continue

        # Pick the voice for this segment's speaker
        speaker = seg.get("speaker", "SPEAKER_00")
        seg_voice = voice_map.get(speaker, list(voice_map.values())[0])

        # Calculate smart rate to fit TTS into the available window
        available_sec = seg.get("end", 0) - seg.get("start", 0)
        rate = _estimate_tts_rate(text, available_sec)

        try:
            asyncio.run(_generate_tts_segment(text, seg_voice, out_mp3, rate=rate))

            # Convert to OGG and trim silence padding
            data, sr = sf.read(out_mp3)
            if len(data.shape) > 1:
                data = data.mean(axis=1)  # mono

            # Trim leading/trailing near-silence
            valid = np.where(np.abs(data) > 0.001)[0]
            if len(valid) > 0:
                pad = int(0.05 * sr)
                start = max(0, valid[0] - pad)
                end = min(len(data), valid[-1] + 1 + pad)
                data = data[start:end]

            sf.write(out_ogg, data, sr, format="ogg", subtype="vorbis")
            audio_files.append(out_ogg)

        except Exception as e:
            print(f"    Warning: TTS failed for segment {i}: {e}")
            # Generate silence as fallback
            duration_ms = int((seg.get("end", 0) - seg.get("start", 0)) * 1000)
            silence = AudioSegment.silent(duration=max(duration_ms, 100))
            silence.export(out_ogg, format="ogg")
            audio_files.append(out_ogg)

    return audio_files


# ─── Audio Assembly ─────────────────────────────────────────────────────────

def assemble_audio(
    segments: list[dict],
    audio_files: list[str],
    output_wav: str,
    avoid_overlap: bool = True,
):
    """
    Place each TTS segment at its timestamp on a silent base track.

    Optionally shifts overlapping segments to avoid collisions.
    """
    print(f"[5/6] Assembling translated audio timeline...")

    if not segments:
        raise ValueError("No segments to assemble")

    total_duration_ms = int(segments[-1]["end"] * 1000) + 2000  # +2s buffer
    base = AudioSegment.silent(duration=total_duration_ms, frame_rate=44100)

    last_end_ms = 0

    for seg, audio_file in tqdm(
        zip(segments, audio_files), total=len(segments), desc="Assembling"
    ):
        try:
            tts_audio = AudioSegment.from_file(audio_file)
        except Exception as e:
            print(f"    Warning: Could not load {audio_file}: {e}")
            continue

        start_ms = int(seg["start"] * 1000)

        if avoid_overlap and last_end_ms > start_ms:
            # Shift forward to avoid overlapping previous segment
            start_ms = last_end_ms + 50  # 50ms gap

        # Last-resort speedup via ffmpeg atempo — only for extreme overruns
        # that the Edge TTS rate parameter couldn't fully handle.
        seg_duration_ms = int((seg["end"] - seg["start"]) * 1000)
        tts_duration_ms = len(tts_audio)

        if tts_duration_ms > seg_duration_ms * 1.5 and seg_duration_ms > 200:
            # Mild atempo, capped at 1.35x to preserve quality
            speed_factor = min(tts_duration_ms / seg_duration_ms, 1.35)
            try:
                tts_audio = _speed_up_audio(tts_audio, speed_factor)
            except Exception:
                pass  # Keep original speed if speedup fails

        base = base.overlay(tts_audio, position=start_ms)
        last_end_ms = start_ms + len(tts_audio)

    base.export(output_wav, format="wav")
    print(f"    Assembled audio: {output_wav}")


def _speed_up_audio(audio_seg: AudioSegment, factor: float) -> AudioSegment:
    """Speed up audio by factor using ffmpeg atempo filter."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        audio_seg.export(tmp_in.name, format="wav")

        # atempo filter only supports 0.5-2.0, chain for higher
        tempo_filters = []
        remaining = factor
        while remaining > 2.0:
            tempo_filters.append("atempo=2.0")
            remaining /= 2.0
        if remaining > 0.5:
            tempo_filters.append(f"atempo={remaining:.4f}")

        filter_str = ",".join(tempo_filters) if tempo_filters else "atempo=1.0"
        run_cmd(
            f'ffmpeg -y -i "{tmp_in.name}" -filter:a "{filter_str}" "{tmp_out.name}"'
        )

        result = AudioSegment.from_wav(tmp_out.name)
        os.unlink(tmp_in.name)
        os.unlink(tmp_out.name)
        return result


# ─── Vocal Separation ───────────────────────────────────────────────────────

def separate_vocals(
    audio_wav: str,
    output_dir: str,
    device: str | None = None,
) -> str:
    """
    Separate vocals from background music/ambience using Demucs (Meta).

    Returns the path to the instrumental (no-vocals) track.
    The model splits audio into: drums, bass, other, vocals.
    We recombine everything except vocals → instrumental track.
    """
    print("[1.5/6] Separating vocals from background audio (demucs)...")
    print("    This may take a minute on first run (model download)...")

    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torchaudio

        if device is None:
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(device)

        print(f"    Using device: {dev}")

        # Load the htdemucs model
        model = get_model("htdemucs")
        model.to(dev)

        # Load audio as tensor
        wav, sr = torchaudio.load(audio_wav)

        # Resample to model's sample rate if needed
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
            sr = model.samplerate

        # Ensure stereo
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        # Add batch dimension: (batch, channels, samples)
        wav = wav.unsqueeze(0).to(dev)

        # Apply the model — returns (batch, sources, channels, samples)
        print("    Running separation model...")
        with torch.no_grad():
            sources = apply_model(model, wav, device=dev)

        # sources shape: (1, n_sources, 2, samples)
        # model.sources gives the stem names, e.g. ['drums', 'bass', 'other', 'vocals']
        sources = sources.squeeze(0)  # (n_sources, 2, samples)

        # Sum all stems except vocals → instrumental
        instrumental = None
        for i, stem_name in enumerate(model.sources):
            if stem_name == "vocals":
                print(f"    Removing: {stem_name}")
                continue
            print(f"    Keeping:  {stem_name}")
            if instrumental is None:
                instrumental = sources[i].clone()
            else:
                instrumental += sources[i]

        # Save instrumental track
        instrumental_path = os.path.join(output_dir, "instrumental.wav")
        torchaudio.save(
            instrumental_path,
            instrumental.cpu(),
            sr,
        )

        del model, sources, instrumental, wav
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        print(f"    Instrumental track saved: {instrumental_path}")
        return instrumental_path

    except ImportError as e:
        print(f"    Error: demucs not installed properly ({e})")
        print("    Install with: pip install demucs")
        print("    Falling back to original audio (with vocals).")
        return audio_wav
    except Exception as e:
        print(f"    Warning: Vocal separation failed ({e})")
        import traceback
        traceback.print_exc()
        print("    Falling back to original audio (with vocals).")
        return audio_wav


# ─── SRT Export ─────────────────────────────────────────────────────────────

def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def export_srt(
    segments: list[dict],
    output_path: str,
):
    """Export segments as an SRT subtitle file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = _format_srt_time(seg.get("start", 0))
            end = _format_srt_time(seg.get("end", 0))
            text = seg.get("text", "").strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    print(f"    SRT saved: {output_path}")


# ─── Duration Enforcement ───────────────────────────────────────────────────

def enforce_duration(audio_path: str, target_ms: int):
    """
    Pad or trim *audio_path* in-place so its duration is exactly *target_ms*.
    Uses pydub for sub-millisecond-accurate editing.
    """
    audio = AudioSegment.from_file(audio_path)
    current_ms = len(audio)
    diff = target_ms - current_ms

    if diff == 0:
        return  # already exact

    if diff > 0:
        # Pad with silence at the end
        audio = audio + AudioSegment.silent(duration=diff, frame_rate=audio.frame_rate)
    else:
        # Trim from the end
        audio = audio[:target_ms]

    ext = Path(audio_path).suffix.lstrip(".").lower()
    fmt_map = {"mp3": "mp3", "ogg": "ogg", "wav": "wav", "m4a": "mp4"}
    audio.export(audio_path, format=fmt_map.get(ext, "wav"))
    print(f"    Duration adjusted by {diff:+d} ms → exactly {target_ms} ms")


# ─── Mix with Original ──────────────────────────────────────────────────────

def mix_audio(
    original_wav: str,
    translated_wav: str,
    output_file: str,
    original_volume: float = 0.15,
    translated_volume: float = 1.0,
):
    """
    Mix the background audio with the translated TTS audio.

    When vocal separation is active, original_wav is already the
    instrumental track (no vocals), so you get: music + translated speech.
    """
    print(f"[6/6] Mixing final audio...")

    original = AudioSegment.from_wav(original_wav)
    translated = AudioSegment.from_wav(translated_wav)

    # Match durations
    if len(translated) < len(original):
        translated += AudioSegment.silent(
            duration=len(original) - len(translated)
        )
    elif len(translated) > len(original):
        translated = translated[:len(original)]

    # Adjust volumes (in dB)
    orig_db = 20 * math.log10(max(original_volume, 0.01))
    trans_db = 20 * math.log10(max(translated_volume, 0.01))

    mixed = original.apply_gain(orig_db).overlay(
        translated.apply_gain(trans_db)
    )

    # Determine output format from extension
    ext = Path(output_file).suffix.lstrip(".").lower()
    format_map = {"mp3": "mp3", "ogg": "ogg", "wav": "wav", "m4a": "mp4"}
    out_format = format_map.get(ext, "wav")

    mixed.export(output_file, format=out_format)
    print(f"    Output saved: {output_file}")


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def translate_audio(
    input_file: str,
    target_lang: str = "es",
    source_lang: str | None = None,
    output_file: str | None = None,
    whisper_model: str = "base",
    voice: str | None = None,
    original_volume: float = 0.15,
    min_speakers: int = 1,
    max_speakers: int = 1,
    hf_token: str | None = None,
    remove_vocals: bool = False,
    keep_temp: bool = False,
):
    """
    Complete pipeline: Audio/Video → Transcribe → Diarize → Translate → TTS → Mix → Output.

    Args:
        input_file:       Path to input audio or video file.
        target_lang:      Target language code (e.g. "es", "fr", "ja", "de").
        source_lang:      Source language code, or None for auto-detection.
        output_file:      Output file path. Defaults to input_translated.wav.
        whisper_model:    WhisperX model size (tiny, base, small, medium, large-v2).
        voice:            Edge TTS voice name, or None for auto-selection.
        original_volume:  Volume of original audio in mix (0.0 - 1.0). Set to 0 for TTS-only.
        min_speakers:     Minimum number of speakers for diarization.
        max_speakers:     Maximum number of speakers for diarization.
        hf_token:         HuggingFace token for pyannote diarization models.
        remove_vocals:    Remove original vocals, keep background music only.
        keep_temp:        Keep temporary files for debugging.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Setup output path
    if output_file is None:
        output_file = str(
            input_path.parent / f"{input_path.stem}_translated_{target_lang}.wav"
        )

    # Create temp directory
    work_dir = tempfile.mkdtemp(prefix="sonitranslate_")
    audio_wav = os.path.join(work_dir, "audio.wav")
    tts_dir = os.path.join(work_dir, "tts_segments")
    translated_wav = os.path.join(work_dir, "translated.wav")

    try:
        # 0. Probe the exact input duration
        input_duration_ms = get_media_duration_ms(input_file)
        print(f"    Input duration: {input_duration_ms} ms")

        # 1. Extract audio
        extract_audio(input_file, audio_wav)

        # 1.5. Vocal separation (remove original speech, keep music)
        if remove_vocals:
            bg_audio = separate_vocals(audio_wav, work_dir)
        else:
            bg_audio = audio_wav

        # 2. Transcribe
        result = transcribe(
            audio_wav,
            source_lang=source_lang,
            model_name=whisper_model,
        )
        segments = result["segments"]
        detected_lang = result["language"]

        if not segments:
            raise ValueError("No speech segments found in audio")

        # Print a preview of transcription
        print("\n    ── Transcription Preview ──")
        for seg in segments[:5]:
            t = seg.get("text", "").strip()
            print(f"    [{seg['start']:.1f}s - {seg['end']:.1f}s] {t}")
        if len(segments) > 5:
            print(f"    ... and {len(segments) - 5} more segments\n")

        # 2.5. Diarize (speaker identification)
        token = hf_token or os.environ.get("HF_TOKEN")
        result = diarize(
            audio_wav,
            result,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            hf_token=token,
        )
        segments = result["segments"]

        # 3. Translate
        src = source_lang or detected_lang
        if src == target_lang:
            print("    Source and target languages are the same — skipping translation.")
            translated_segments = segments
        else:
            translated_segments = translate_segments(
                segments, target_lang=target_lang, source_lang=src
            )

        # Print a preview of translation
        print("\n    ── Translation Preview ──")
        for orig, trans in zip(segments[:3], translated_segments[:3]):
            print(f"    {orig.get('text', '').strip()}")
            print(f"    → {trans.get('text', '').strip()}\n")

        # 3.5. Detect speaker genders (pitch analysis)
        speaker_genders = detect_speaker_genders(audio_wav, segments)

        # 4. Generate TTS (with gender-matched per-speaker voices)
        if max_speakers > 1 or not voice:
            voice_map = assign_voices_to_speakers(
                translated_segments, target_lang,
                speaker_genders=speaker_genders,
            )
        else:
            voice_map = None

        audio_files = generate_tts(
            translated_segments,
            target_lang=target_lang,
            audio_dir=tts_dir,
            voice=voice,
            voice_map=voice_map,
        )

        # 5. Assemble on timeline
        assemble_audio(
            translated_segments, audio_files, translated_wav
        )

        # 6. Mix with background audio
        mix_source = bg_audio if remove_vocals else audio_wav
        if original_volume > 0 or remove_vocals:
            mix_vol = max(original_volume, 0.5) if remove_vocals else original_volume
            mix_audio(
                original_wav=mix_source,
                translated_wav=translated_wav,
                output_file=output_file,
                original_volume=mix_vol,
            )
        else:
            # Just copy the translated audio
            from shutil import copy2
            copy2(translated_wav, output_file)
            print(f"[6/6] Output saved (TTS only): {output_file}")

        # 7. Enforce exact input duration on output
        enforce_duration(output_file, input_duration_ms)

        # 8. Export SRT subtitles (both source and translated)
        srt_base = str(Path(output_file).with_suffix(""))
        export_srt(translated_segments, f"{srt_base}.srt")
        export_srt(segments, f"{srt_base}_original.srt")

        print("\n" + "=" * 60)
        print(f"  Done! Translated audio saved to: {output_file}")
        print(f"  SRT subtitles: {srt_base}.srt")
        print(f"  {len(segments)} segments | {detected_lang} → {target_lang}")
        print("=" * 60)

        return output_file

    finally:
        if not keep_temp:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"\n    Temp files kept at: {work_dir}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SoniTranslate Prototype — Translate audio/video to another language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python translate_audio.py podcast.mp3 --target es
  python translate_audio.py lecture.wav --source en --target fr --voice fr-FR-DeniseNeural
  python translate_audio.py video.mp4 --target ja --model medium --output dubbed.wav
  python translate_audio.py interview.mp3 --target de --original-volume 0  # TTS only
  python translate_audio.py debate.mp3 --target es --max-speakers 3 --hf-token hf_xxxxx
  python translate_audio.py song.mp3 --target es --remove-vocals  # keep music, replace voice

Supported languages (common):
  en (English), es (Spanish), fr (French), de (German), it (Italian),
  pt (Portuguese), ja (Japanese), ko (Korean), zh (Chinese), ru (Russian),
  ar (Arabic), hi (Hindi), nl (Dutch), pl (Polish), tr (Turkish), ...

Edge TTS voices (examples):
  en-US-GuyNeural, en-US-JennyNeural, es-ES-AlvaroNeural,
  fr-FR-DeniseNeural, de-DE-ConradNeural, ja-JP-NanamiNeural
        """,
    )

    parser.add_argument(
        "input",
        help="Input audio or video file",
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target language code (e.g. es, fr, ja, de)",
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Source language code (default: auto-detect)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output audio file path (default: <input>_translated_<lang>.wav)",
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
        help="Edge TTS voice name (default: auto-select for target language)",
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
        help="Max speakers for diarization (default: 1, set >1 to enable diarization)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for pyannote diarization (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--remove-vocals",
        action="store_true",
        help="Remove original vocals and keep only background music/ambience",
    )
    parser.add_argument(
        "--original-volume",
        type=float,
        default=0.15,
        help="Volume of original/background audio in mix, 0.0-1.0 (default: 0.15)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files for debugging",
    )

    args = parser.parse_args()

    translate_audio(
        input_file=args.input,
        target_lang=args.target,
        source_lang=args.source,
        output_file=args.output,
        whisper_model=args.model,
        voice=args.voice,
        original_volume=args.original_volume,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        hf_token=args.hf_token,
        remove_vocals=args.remove_vocals,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()
