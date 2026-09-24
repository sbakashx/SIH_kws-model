"""
live_test.py

Continuously listens through your laptop's default microphone and runs
the trained KWS model on a rolling 1-second window, printing the
activation score in real time.

Run from inside your activated virtual environment:
    python live_test.py
"""

import numpy as np
import sounddevice as sd
import librosa
import time
from pathlib import Path
from tensorflow import keras

# ---------------- CONFIG ----------------

MODEL_DIR = Path(r"D:\kws-hackathon\model")
MODEL_PATH = MODEL_DIR / "best_model.keras"
NORM_STATS_PATH = MODEL_DIR / "normalization.npy"

SAMPLE_RATE = 16000
CLIP_LENGTH_SAMPLES = SAMPLE_RATE * 1  # 1 second

# MFCC settings — MUST match extract_features.py (N_MFCC = 13)
N_MFCC = 13
WINDOW_MS = 40
STRIDE_MS = 20
N_FFT = int(SAMPLE_RATE * WINDOW_MS / 1000)
HOP_LENGTH = int(SAMPLE_RATE * STRIDE_MS / 1000)

DETECTION_THRESHOLD = 0.75     # Raised from 0.50 to suppress soft confusables
INFERENCE_INTERVAL_SEC = 0.2   # 200ms rolling step
COOLDOWN_SEC = 1.0             # Don't re-trigger within this window
BLOCK_SIZE = 1600              # ~100ms chunk

# -----------------------------------------


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """Same feature extraction as extract_features.py."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return mfcc.T  # (time, features)


def main():
    if not MODEL_PATH.exists() or not NORM_STATS_PATH.exists():
        print(f"Error: Model files not found in {MODEL_DIR}. Run train_model.py first!")
        return

    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    mean, std = np.load(NORM_STATS_PATH)
    print(f"Model loaded. Normalization stats: mean={mean:.4f}, std={std:.4f}")

    rolling_buffer = np.zeros(CLIP_LENGTH_SAMPLES, dtype=np.float32)
    score_history = [0.0, 0.0]  # Buffer for 2-frame moving average smoothing
    last_detection_time = 0.0

    def audio_callback(indata, frames, time_info, status):
        nonlocal rolling_buffer
        if status:
            print(f"  [audio status] {status}")
        new_samples = indata[:, 0]
        rolling_buffer = np.roll(rolling_buffer, -len(new_samples))
        rolling_buffer[-len(new_samples):] = new_samples

    print("\nListening... speak your keyword. Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            while True:
                time.sleep(INFERENCE_INTERVAL_SEC)

                audio_snapshot = rolling_buffer.copy()
                mfcc = extract_mfcc(audio_snapshot)
                mfcc = (mfcc - mean) / std
                mfcc = mfcc[np.newaxis, ..., np.newaxis]

                raw_score = model.predict(mfcc, verbose=0)[0][0]

                # Update moving average score buffer
                score_history.pop(0)
                score_history.append(float(raw_score))
                smoothed_score = float(np.mean(score_history))

                now = time.time()
                if smoothed_score > DETECTION_THRESHOLD and (now - last_detection_time) > COOLDOWN_SEC:
                    print(f"\n  >>> DETECTED 'ROB'  (score={smoothed_score:.3f}, raw={raw_score:.3f}) <<<\n")
                    last_detection_time = now
                else:
                    bar_len = int(smoothed_score * 20)
                    bar = "#" * bar_len + "-" * (20 - bar_len)
                    print(f"  score={smoothed_score:.3f}  [{bar}]", end="\r")

        except KeyboardInterrupt:
            print("\n\nStopped.")


if __name__ == "__main__":
    main()