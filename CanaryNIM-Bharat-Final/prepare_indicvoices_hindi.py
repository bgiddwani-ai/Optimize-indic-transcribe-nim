#!/usr/bin/env python3
"""Materialize the IndicVoices Hindi valid split and make two Riva manifests."""

import json
import shutil
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from huggingface_hub import HfApi, hf_hub_download


DATASET = "ai4bharat/IndicVoices"
ROOT = Path("/data")
SOURCE = ROOT / "indicvoices_hindi_valid_source"
AUDIO = ROOT / "indicvoices_hindi_valid_audio"
DEFAULT = ROOT / "hi_manifest_cleaned_default.json"
SORTED = ROOT / "hi_manifest_cleaned_sorted.json"
SUMMARY = ROOT / "hi_manifest_cleaned_summary.json"


def main() -> None:
    files = [
        name
        for name in HfApi().list_repo_files(DATASET, repo_type="dataset")
        if name.startswith("hindi/valid-") and name.endswith(".parquet")
    ]
    if not files:
        raise RuntimeError("No Hindi valid parquet found")
    SOURCE.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    parquet = [
        hf_hub_download(DATASET, filename=name, repo_type="dataset", local_dir=SOURCE)
        for name in files
    ]
    dataset = load_dataset("parquet", data_files=parquet, split="train").cast_column(
        "audio_filepath", Audio(decode=False)
    )
    rows = []
    filtered = 0
    for index, row in enumerate(dataset):
        text = str(row.get("text", ""))
        if "<unintelligible>" in text.casefold():
            filtered += 1
            continue
        audio = row["audio_filepath"]
        original = Path(audio.get("path") or f"{index:06d}.wav")
        suffix = original.suffix or ".wav"
        target = AUDIO / f"{index:06d}_{original.stem}{suffix}"
        if not target.exists():
            if audio.get("bytes") is not None:
                target.write_bytes(audio["bytes"])
            elif original.exists():
                shutil.copy2(original, target)
            else:
                raise RuntimeError(f"No audio bytes/path for row {index}")
        duration = row.get("duration")
        if duration is None:
            duration = sf.info(target).duration
        rows.append(
            {
                "audio_filepath": str(target),
                "duration": float(duration),
                "text": text,
            }
        )

    with DEFAULT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sorted_rows = sorted(rows, key=lambda row: row["duration"])
    with SORTED.open("w", encoding="utf-8") as handle:
        for row in sorted_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": DATASET,
        "split": "valid",
        "language": "hindi",
        "source_rows": len(dataset),
        "filtered_unintelligible": filtered,
        "cleaned_rows": len(rows),
        "total_duration_seconds": sum(row["duration"] for row in rows),
        "default_manifest": str(DEFAULT),
        "sorted_manifest": str(SORTED),
        "sorted_non_decreasing": all(
            sorted_rows[i]["duration"] <= sorted_rows[i + 1]["duration"]
            for i in range(max(0, len(sorted_rows) - 1))
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
