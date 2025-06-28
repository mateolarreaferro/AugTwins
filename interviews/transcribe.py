import os
import sys
from openai import OpenAI

API_KEY = (
)

ALLOWED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python transcribe_folder.py <input_subfolder> [output_folder]")
        sys.exit(1)

    # Folder where this script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Sub‑directory of audio files to process (relative to script dir)
    input_subfolder = sys.argv[1]
    input_folder = os.path.join(script_dir, input_subfolder)

    # Where to save transcripts
    output_folder = (
        os.path.abspath(sys.argv[2])
        if len(sys.argv) >= 3
        else os.path.join(input_folder, "transcripts")
    )

    # Allow environment variable to override hard‑coded key if present
    api_key = os.getenv("OPENAI_API_KEY", API_KEY)
    if not api_key:
        print("ERROR: No OpenAI API key found (hard‑coded key is empty and OPENAI_API_KEY not set).")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    os.makedirs(output_folder, exist_ok=True)

    # Process files
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(ALLOWED_EXTENSIONS):
            print(f"Skipping non‑audio file: {filename}")
            continue

        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, f"{filename}.txt")

        if os.path.exists(out_path):
            print(f"✅ Already transcribed: {filename}")
            continue

        print(f"🎧 Transcribing {filename} …")
        try:
            with open(in_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1"
                )

            # Save transcript
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(transcript.text)
            print(f"✅ Saved transcript → {out_path}")

        except Exception as e:
            print(f"❌ Error transcribing {filename}: {e}")


if __name__ == "__main__":
    main()
