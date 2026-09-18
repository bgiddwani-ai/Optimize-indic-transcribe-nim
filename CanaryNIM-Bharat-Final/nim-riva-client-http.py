import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ----------------------------
# Config
# ----------------------------
URL = "http://0.0.0.0:9000/v1/audio/transcriptions"
AUDIO_FOLDER = "/data/dataset/hi_audio_cleaned/002128.wav"
OUTPUT_FILE = "transcriptions.jsonl"

LANGUAGE = "hi"
MAX_WORKERS = 256  # Tune based on server capacity

# Supported extensions
EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


# ----------------------------
# Session per thread
# ----------------------------
thread_local = {}


def get_session():
    import threading

    thread_id = threading.get_ident()

    if thread_id not in thread_local:
        session = requests.Session()

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=3,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        thread_local[thread_id] = session

    return thread_local[thread_id]


# ----------------------------
# Inference
# ----------------------------
def transcribe(audio_path):
    session = get_session()

    try:
        with open(audio_path, "rb") as f:
            response = session.post(
                URL,
                data={"language": LANGUAGE},
                files={
                    "file": (
                        audio_path.name,
                        f,
                        "application/octet-stream",
                    )
                },
                timeout=600,
            )

        response.raise_for_status()

        try:
            result = response.json()
        except Exception:
            result = response.text

        return {
            "file": str(audio_path),
            "success": True,
            "result": result,
        }

    except Exception as e:
        return {
            "file": str(audio_path),
            "success": False,
            "error": str(e),
        }


# ----------------------------
# Main
# ----------------------------
def main():
    folder = Path(AUDIO_FOLDER)

    audio_files = [
        f
        for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ]

    print(f"Found {len(audio_files)} audio files")

    completed = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            futures = {
                executor.submit(transcribe, audio_file): audio_file
                for audio_file in audio_files
            }

            for future in as_completed(futures):
                result = future.result()

                out_f.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                out_f.flush()

                completed += 1

                if completed % 10 == 0 or completed == len(audio_files):
                    print(
                        f"Completed {completed}/{len(audio_files)} "
                        f"({100 * completed / len(audio_files):.1f}%)"
                    )

    print(f"Done. Results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
