#!/usr/bin/env bash

set -uo pipefail

result_dir="/data/indicvoices_riva_benchmark"

mkdir -p "$result_dir/logs"

summary="$result_dir/run_status.tsv"

printf 'order\tmanifest\tconcurrency\tstart_utc\tend_utc\telapsed_seconds\texit_code\toutput\tlog\n' > "$summary"

order=0

for kind in default sorted; do
    manifest="/data/hi_manifest_cleaned_${kind}.json"

    for concurrency in 1 8 16 32 64 128; do
        order=$((order + 1))

        output="/data/${concurrency}_indicvoices_${kind}_indic_transcribe_nim_riva_trt.json"
        log="$result_dir/logs/${kind}_c${concurrency}.log"

        start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        start_epoch="$(date +%s)"

        echo "RUN_START order=$order kind=$kind concurrency=$concurrency utc=$start_utc"

        riva_asr_client \
            --automatic_punctuation=false \
            --word_time_offsets=false \
            --print_transcripts=false \
            --num_iterations=1 \
            --language_code=hi \
            --audio_file="$manifest" \
            --num_parallel_requests="$concurrency" \
            --output_filename="$output" \
            > "$log" 2>&1

        exit_code=$?

        end_epoch="$(date +%s)"
        end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        elapsed=$((end_epoch - start_epoch))

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$order" "$kind" "$concurrency" "$start_utc" "$end_utc" \
            "$elapsed" "$exit_code" "$output" "$log" >> "$summary"

        echo "RUN_END order=$order kind=$kind concurrency=$concurrency exit=$exit_code elapsed=$elapsed utc=$end_utc"
    done
done

echo "MATRIX_COMPLETE summary=$summary"
