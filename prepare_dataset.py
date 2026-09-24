"""
prepare_dataset.py

Splits multi-utterance "rob-sounding" raw audio recordings
from D:\Audio Samples\rob sounding datasit into individual 1-word WAV clips
with 200ms silence padding before and after each word.
Outputs to D:\kws-hackathon\confusable_split for manual review.
"""

import os
import re
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence

# --- CONFIGURATION ---
RAW_ROOT = Path(r"D:\Audio Samples\rob sounding datasit")
OUTPUT_SPLIT_DIR = Path(r"D:\kws-hackathon\confusable_split")

TARGET_SAMPLE_RATE = 16000

# Silence detection settings
MIN_SILENCE_LEN_MS = 250      # Minimum gap between words to count as a split
SILENCE_THRESH_OFFSET = 14    # Sensitivity threshold (dB below clip average)
KEEP_SILENCE_MS = 240         # 240ms margin preserved on both sides of each word

SUPPORTED_EXTENSIONS = {".ogg", ".aac", ".m4a", ".mp4", ".mp3", ".wav"}


def sanitize(name: str) -> str:
    name = name.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-.]", "", name)


def load_audio(path: Path) -> AudioSegment:
    audio = AudioSegment.from_file(path)
    return audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)


def find_audio_files(folder: Path):
    if not folder.exists():
        print(f"  [WARN] Folder not found: {folder}")
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def segment_multi(path: Path, out_dir: Path, prefix: str, counter: list):
    try:
        audio = load_audio(path)
    except Exception as e:
        print(f"  [SKIP] Could not read {path.name}: {e}")
        return

    chunks = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN_MS,
        silence_thresh=audio.dBFS - SILENCE_THRESH_OFFSET,
        keep_silence=KEEP_SILENCE_MS,  # 200ms padding
    )

    if not chunks:
        counter[0] += 1
        out_name = f"{prefix}_{counter[0]:03d}_{sanitize(path.stem)}.wav"
        audio.export(out_dir / out_name, format="wav")
        return

    for chunk in chunks:
        counter[0] += 1
        out_name = f"{prefix}_{counter[0]:03d}_{sanitize(path.stem)}.wav"
        chunk.export(out_dir / out_name, format="wav")

    print(f"  {path.name} -> split into {len(chunks)} word clips")


def main():
    OUTPUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    non_rob_counter = [0]

    print(f"Segmenting raw audio from: {RAW_ROOT}\n")
    audio_files = find_audio_files(RAW_ROOT)
    print(f"Found {len(audio_files)} multi-word file(s) to process...\n")

    for f in audio_files:
        segment_multi(f, OUTPUT_SPLIT_DIR, "confusable", non_rob_counter)

    print(f"\nDone! Created {non_rob_counter[0]} padded clips (200ms padding) in -> {OUTPUT_SPLIT_DIR}")


if __name__ == "__main__":
    main()