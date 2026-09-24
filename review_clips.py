"""
review_clips.py

Fast manual QC pass over your split clips.
Plays each WAV file and waits for a single keypress:
    k = keep (moves file to D:\kws-hackathon\confusable_reviewed)
    d = discard (moves file to D:\kws-hackathon\confusable_split\rejected)
    r = replay this clip
    q = quit early

Windows only (uses winsound + msvcrt).

Usage:
    python review_clips.py
    (Or specify custom path: python review_clips.py <folder_path>)
"""

import sys
import winsound
import msvcrt
from pathlib import Path

# Default path targets
DEFAULT_SPLIT_DIR = Path(r"D:\kws-hackathon\confusable_split")
DEFAULT_REVIEWED_DIR = Path(r"D:\kws-hackathon\confusable_reviewed")


def review_folder(folder: Path, approved_dir: Path):
    wav_files = sorted([f for f in folder.glob("*.wav") if f.is_file()])
    if not wav_files:
        print(f"No .wav files found in {folder}")
        return

    rejected_dir = folder / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    total = len(wav_files)
    kept = 0
    discarded = 0

    print(f"Reviewing {total} clips in {folder}")
    print(f"Approved clips will move to: {approved_dir}")
    print("Controls: [k] keep   [d] discard   [r] replay   [q] quit\n")

    i = 0
    while i < len(wav_files):
        f = wav_files[i]
        print(f"[{i+1}/{total}] {f.name}", end="  ", flush=True)
        winsound.PlaySound(str(f), winsound.SND_FILENAME)

        key = msvcrt.getch().decode(errors="ignore").lower()

        if key == "k":
            dest = approved_dir / f.name
            f.rename(dest)
            print("-> kept (moved to reviewed)")
            kept += 1
            i += 1
        elif key == "d":
            dest = rejected_dir / f.name
            f.rename(dest)
            print("-> discarded")
            discarded += 1
            i += 1
        elif key == "r":
            print("-> replaying")
            continue  # stay on same index
        elif key == "q":
            print("-> quitting early")
            break
        else:
            print("-> unrecognized key, replaying...")
            continue

    print(f"\nDone. Kept: {kept}  Discarded: {discarded}")


if __name__ == "__main__":
    target_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SPLIT_DIR
    target_approved = DEFAULT_REVIEWED_DIR

    review_folder(target_folder, target_approved)