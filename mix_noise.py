"""
mix_noise.py

Takes clean, verified confusable clips from confusable_reviewed,
mixes them with background noise chunks, and exports clean + noisy versions
directly into dataset/non_rob.
"""

import os
import glob
import random
from pathlib import Path
from pydub import AudioSegment

# --- PATH CONFIGURATION ---
BASE_DIR = Path(r"D:\kws-hackathon")
REVIEWED_DIR = BASE_DIR / "confusable_reviewed"
NOISE_SOURCE_DIR = BASE_DIR / "dataset" / "non_rob"  # Source for bg_noise_*.wav files
OUTPUT_NON_ROB = BASE_DIR / "dataset" / "non_rob"

TARGET_SR = 16000


def mix_audio_with_noise():
    OUTPUT_NON_ROB.mkdir(parents=True, exist_ok=True)

    # Load all verified word clips
    word_files = list(REVIEWED_DIR.glob("*.wav"))
    if not word_files:
        print(f"No WAV files found in {REVIEWED_DIR}! Please copy verified clips there first.")
        return

    # Load all 1-second background noise chunks
    noise_files = list(NOISE_SOURCE_DIR.glob("bg_noise_*.wav"))
    if not noise_files:
        print(f"Warning: No bg_noise_*.wav files found in {NOISE_SOURCE_DIR}. Exporting clean clips only.")

    print(f"Found {len(word_files)} reviewed word clips and {len(noise_files)} noise samples.")
    total_exported = 0

    for wf in word_files:
        base_name = wf.stem
        word_audio = AudioSegment.from_file(wf).set_frame_rate(TARGET_SR).set_channels(1)

        # 1. Export Clean Copy
        clean_out = OUTPUT_NON_ROB / f"clean_{base_name}.wav"
        word_audio.export(clean_out, format="wav")
        total_exported += 1

        # 2. Export Noise-Mixed Copy
        if noise_files:
            noise_file = random.choice(noise_files)
            noise_audio = AudioSegment.from_file(noise_file).set_frame_rate(TARGET_SR).set_channels(1)

            # Reduce noise volume by 10dB to 20dB relative to word
            gain_reduction = random.randint(10, 20)
            adjusted_noise = noise_audio - gain_reduction

            # Overlay word on top of noise
            noisy_audio = word_audio.overlay(adjusted_noise)

            noisy_out = OUTPUT_NON_ROB / f"noisy_{base_name}.wav"
            noisy_audio.export(noisy_out, format="wav")
            total_exported += 1

    print(f"\nDone! Exported {total_exported} files (clean + noise-mixed) to -> {OUTPUT_NON_ROB}")


if __name__ == "__main__":
    mix_audio_with_noise()