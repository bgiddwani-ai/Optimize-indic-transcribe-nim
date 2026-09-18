#!/usr/bin/env python3
"""
manifest.py - Standalone manifest arranger (no external dependencies)

Usage:
    python manifest.py --input_path /path/to/audio/dir \
        sorted_output.json worst_output.json sampled_output.json

    python manifest.py --input_path /path/to/manifest.json \
        sorted_output.json worst_output.json sampled_output.json
"""

import argparse
import json
import random
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# dir2manifest: scan a directory for audio files and build manifest entries
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.opus', '.m4a'}

def dir2manifest(input_path: Path):
    manifests = []
    for root, _, files in os.walk(input_path):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in AUDIO_EXTENSIONS:
                fpath = Path(root) / fname
                entry = {
                    "audio_filepath": str(fpath),
                    "duration": get_audio_duration(fpath),
                    "text": ""
                }
                manifests.append(entry)
    return manifests


def get_audio_duration(fpath: Path) -> float:
    """Try to read duration from WAV header; fall back to file-size estimate."""
    try:
        import wave
        if fpath.suffix.lower() == '.wav':
            with wave.open(str(fpath), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return round(frames / float(rate), 4)
    except Exception:
        pass
    # crude fallback: assume 16 kHz 16-bit mono
    size = fpath.stat().st_size
    return round(size / (16000 * 2), 4)


# ---------------------------------------------------------------------------
# load_manifest: read an existing JSONL / JSON manifest file
# ---------------------------------------------------------------------------
def load_manifest(input_path: Path):
    manifests = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                manifests.append(json.loads(line))
    return manifests


# ---------------------------------------------------------------------------
# Arranger: sort, find worst-case batch, and produce sampled runs
# ---------------------------------------------------------------------------
class Arranger:
    def __init__(self, sampled_runs: int = 10):
        self.sampled_runs = sampled_runs

    def shuffle(self, manifests, batch_size: int = 8):
        if not manifests:
            return [], [], []

        # --- best/sorted: ascending duration (easiest batches first) ---
        best = sorted(manifests, key=lambda x: x.get('duration', 0))

        # --- worst: pack the longest utterances into the same batches ---
        worst = sorted(manifests, key=lambda x: x.get('duration', 0), reverse=True)

        # --- sampled: pick the random arrangement with the highest total
        #              duration in the first batch (simulates worst-case
        #              across N random orderings) ---
        best_sampled = None
        best_score = -1
        for _ in range(self.sampled_runs):
            candidate = manifests[:]
            random.shuffle(candidate)
            first_batch = candidate[:batch_size]
            score = sum(e.get('duration', 0) for e in first_batch)
            if score > best_score:
                best_score = score
                best_sampled = candidate

        sampled = best_sampled if best_sampled is not None else manifests[:]

        return best, worst, sampled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sorted, worst-case, and sampled manifests."
    )
    parser.add_argument(
        '--input_path', '-i', required=True,
        help='Path to an audio directory OR an existing .json manifest file'
    )
    parser.add_argument(
        '--batch_size', '-bs', type=int, default=8,
        help='Batch size for worst-case manifest generation (default: 8)'
    )
    parser.add_argument(
        '--sampled_runs', '-runs', type=int, default=10,
        help='Number of random sampling runs (default: 10)'
    )
    parser.add_argument(
        'sorted_mf',
        help='Output path for sorted (best-case) manifest'
    )
    parser.add_argument(
        'worst_mf',
        help='Output path for worst-case manifest'
    )
    parser.add_argument(
        'sampled_mf',
        help='Output path for sampled manifest'
    )
    return parser.parse_args()


def write_manifest(entries, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    print(f"  Wrote {len(entries)} entries → {path}")


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    # Load manifests
    if not input_path.exists():
        raise FileNotFoundError(f"input_path not found: {input_path}")

    if input_path.suffix.lower() == '.json':
        print(f"Loading manifest from file: {input_path}")
        manifests = load_manifest(input_path)
    else:
        print(f"Scanning directory for audio files: {input_path}")
        manifests = dir2manifest(input_path)

    print(f"Total entries: {len(manifests)}")
    for entry in manifests:
        print(entry)

    # Arrange
    arranger = Arranger(sampled_runs=args.sampled_runs)
    best, worst, sampled = arranger.shuffle(manifests, batch_size=args.batch_size)

    # Write outputs
    write_manifest(best,    args.sorted_mf)
    write_manifest(worst,   args.worst_mf)
    write_manifest(sampled, args.sampled_mf)

    print("Done.")


if __name__ == '__main__':
    main()