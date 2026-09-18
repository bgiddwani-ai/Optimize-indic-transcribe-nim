#!/usr/bin/env python3
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path("/data/indicvoices_riva_benchmark")
STATUS = ROOT / "run_status.tsv"
OUT_JSON = ROOT / "benchmark_summary.json"
OUT_CSV = ROOT / "benchmark_summary.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(path: Path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    mapped = {row["audio_filepath"]: row.get("text", "") for row in rows}
    return rows, mapped


def main():
    with STATUS.open(encoding="utf-8") as handle:
        statuses = list(csv.DictReader(handle, delimiter="\t"))
    baseline = {}
    results = []
    for status in statuses:
        kind = status["manifest"]
        concurrency = int(status["concurrency"])
        output = Path(status["output"])
        log_path = Path(status["log"])
        log = log_path.read_text(encoding="utf-8", errors="replace")
        latency = re.search(
            r"Latencies \(ms\):\s*\n\s*Median\s+90th\s+95th\s+99th\s+Avg\s*\n\s*"
            r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
            log,
        )
        runtime = re.search(r"Run time:\s*([0-9.]+) sec", log)
        audio = re.search(r"Total audio processed:\s*([0-9.]+) sec", log)
        rtfx = re.search(r"Throughput:\s*([0-9.]+) RTFX", log)
        responses = re.search(r"Done processing\s+(\d+) responses", log)
        if not all((latency, runtime, audio, rtfx, responses)):
            raise RuntimeError(f"Missing metrics in {log_path}")
        rows, mapped = load_predictions(output)
        if concurrency == 1:
            baseline[kind] = mapped
        result = {
            "manifest": kind,
            "concurrency": concurrency,
            "exit_code": int(status["exit_code"]),
            "wall_seconds": int(status["elapsed_seconds"]),
            "riva_runtime_seconds": float(runtime.group(1)),
            "audio_seconds": float(audio.group(1)),
            "rtfx": float(rtfx.group(1)),
            "latency_median_ms": float(latency.group(1)),
            "latency_p90_ms": float(latency.group(2)),
            "latency_p95_ms": float(latency.group(3)),
            "latency_p99_ms": float(latency.group(4)),
            "latency_avg_ms": float(latency.group(5)),
            "responses_reported": int(responses.group(1)),
            "jsonl_rows": len(rows),
            "unique_audio_paths": len(mapped),
            "nonempty_transcripts": sum(bool(row.get("text", "").strip()) for row in rows),
            "output": str(output),
            "output_sha256": sha256(output),
            "log": str(log_path),
        }
        results.append(result)

    for result in results:
        _, mapped = load_predictions(Path(result["output"]))
        base = baseline[result["manifest"]]
        result["mismatches_vs_same_manifest_c1"] = sum(
            mapped.get(path) != text for path, text in base.items()
        )

    default_base = baseline["default"]
    sorted_base = baseline["sorted"]
    cross_mismatches = sum(sorted_base.get(path) != text for path, text in default_base.items())
    payload = {
        "source_rows": 5530,
        "filtered_unintelligible": 205,
        "cleaned_rows": 5325,
        "manifest_audio_seconds": 30439.3161875,
        "cross_order_c1_mismatches": cross_mismatches,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
