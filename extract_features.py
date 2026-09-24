"""
extract_features.py (With 3x Rob Augmentation)

Extracts 16kHz 1-second MFCC features (49, 10, 1) matching DS-CNN requirements.
Generates 1 clean version + 2 distinct noise-augmented versions for every 'rob' clip
to balance against a larger non_rob dataset.
"""

import os
import glob
import random
import numpy as np
import librosa
from pathlib import Path

# --- CONFIGURATION ---
DATASET_ROOT = Path(r"D:\kws-hackathon\dataset")
ROB_DIR = DATASET_ROOT / "rob"
NON_ROB_DIR = DATASET_ROOT / "non_rob"
OUTPUT_DIR = Path(r"D:\kws-hackathon\features")

SAMPLE_RATE = 16000
CLIP_DURATION_SEC = 1.0
CLIP_LENGTH_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_SEC)

# MFCC parameters
N_MFCC = 13
WINDOW_MS = 40
STRIDE_MS = 20
N_FFT = int(SAMPLE_RATE * WINDOW_MS / 1000)
HOP_LENGTH = int(SAMPLE_RATE * STRIDE_MS / 1000)

# Number of noisy variations to generate per clean 'rob' sample
ROB_NOISE_AUGMENTATIONS_PER_CLIP = 0


def load_and_pad(path: Path) -> np.ndarray:
    """Load WAV, resample to 16kHz mono, pad/trim to 1 second."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) > CLIP_LENGTH_SAMPLES:
        start = (len(audio) - CLIP_LENGTH_SAMPLES) // 2
        audio = audio[start:start + CLIP_LENGTH_SAMPLES]
    elif len(audio) < CLIP_LENGTH_SAMPLES:
        pad_total = CLIP_LENGTH_SAMPLES - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode="constant")
    return audio


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """Extract MFCC array of shape (49, 10)."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return mfcc.T


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Background Noise Chunks for Augmentation
    noise_files = list(NON_ROB_DIR.glob("bg_noise_*.wav"))
    noise_samples = []
    print(f"Loading {len(noise_files)} background noise chunks for augmentation...")
    for nf in noise_files:
        try:
            noise_samples.append(load_and_pad(nf))
        except Exception:
            pass

    features, labels, filenames = [], [], []

    # 2. Process 'rob' Clips (Clean + 2x Noise Augmentations)
    rob_files = sorted([f for f in ROB_DIR.glob("*.wav") if f.is_file()])
    print(f"\nProcessing {len(rob_files)} 'rob' clips...")
    
    for f in rob_files:
        try:
            clean_audio = load_and_pad(f)
            
            # Save original clean version
            features.append(extract_mfcc(clean_audio))
            labels.append(1)
            filenames.append(str(f))

            # Save multiple noise-mixed variations
            if noise_samples:
                for aug_idx in range(ROB_NOISE_AUGMENTATIONS_PER_CLIP):
                    selected_noise = random.choice(noise_samples)
                    # Random noise intensity factor (10% to 35% gain)
                    snr = random.uniform(0.10, 0.35)
                    noisy_audio = clean_audio + (snr * selected_noise)
                    
                    features.append(extract_mfcc(noisy_audio))
                    labels.append(1)
                    filenames.append(f"{f.stem}_aug_noise_{aug_idx+1}")
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")

    # 3. Process 'non_rob' Clips
    non_rob_files = sorted([f for f in NON_ROB_DIR.glob("*.wav") if f.is_file()])
    print(f"\nProcessing {len(non_rob_files)} 'non_rob' clips...")
    
    for f in non_rob_files:
        try:
            audio = load_and_pad(f)
            features.append(extract_mfcc(audio))
            labels.append(0)
            filenames.append(str(f))
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")

    # Convert arrays and reshape for CNN input: (samples, time, features, 1)
    X = np.array(features, dtype=np.float32)[..., np.newaxis]
    y = np.array(labels, dtype=np.int32)

    print(f"\n--- Feature Extraction Complete ---")
    print(f"Final X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class Distribution: {np.sum(y == 1)} 'rob' vs {np.sum(y == 0)} 'non_rob'")

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    with open(OUTPUT_DIR / "filenames.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(filenames))

    print(f"Features successfully saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()