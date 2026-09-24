import os
import glob
from pydub import AudioSegment

# --- PATH CONFIGURATION ---
INPUT_NOISE_DIR = r"D:\kws-hackathon\raw_noise"  # Put your long noise files here
OUTPUT_DIR = r"D:\kws-hackathon\dataset\non_rob"  # Slices go directly into non_rob
os.makedirs(OUTPUT_DIR, exist_ok=True)

# KWS Model Standard Settings
TARGET_SR = 16000  # 16kHz
CHUNK_LENGTH_MS = 1000  # 1 second = 1000ms

def split_long_noise():
    # Supported audio extensions
    extensions = ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac")
    noise_files = []
    for ext in extensions:
        noise_files.extend(glob.glob(os.path.join(INPUT_NOISE_DIR, ext)))

    if not noise_files:
        print(f"No audio files found in {INPUT_NOISE_DIR}!")
        print("Please place your long background noise files in that folder and run again.")
        return

    print(f"Found {len(noise_files)} noise file(s) to process...\n")
    total_saved_chunks = 0

    for file_path in noise_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing: {base_name}...")

        # Load audio and enforce 16kHz Mono
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(TARGET_SR).set_channels(1)

        duration_ms = len(audio)
        num_chunks = duration_ms // CHUNK_LENGTH_MS

        print(f"  Duration: {duration_ms / 1000:.2f}s -> Slicing into {num_chunks} 1-second chunks...")

        for i in range(num_chunks):
            start_ms = i * CHUNK_LENGTH_MS
            end_ms = start_ms + CHUNK_LENGTH_MS
            chunk = audio[start_ms:end_ms]

            output_filename = f"bg_noise_{base_name}_{i+1:03d}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # Export as 16-bit PCM WAV
            chunk.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            total_saved_chunks += 1

    print(f"\n Done! Successfully created {total_saved_chunks} 1-second WAV noise files in:")
    print(f"  {OUTPUT_DIR}")

if __name__ == "__main__":
    split_long_noise()