# indic-canary-optimization

# CanaryNIM-Bharat-Final

End-to-end workflow for converting a NeMo Canary model into a Riva/TensorRT-LLM deployment, serving it through Riva/NIM, running inference, and evaluating ASR performance.

## Repository Structure

```text
├── nemo.sh                 # Container startup script (outside workflow directory)
├── riva_trt.sh             # Container startup script (outside workflow directory)

CanaryNIM-Bharat-Final/
    ├── 01_nemo2riva.sh
    ├── 02_conformer_onnx_to_trt.sh
    ├── 03_riva2rmir.sh
    ├── 04_rmir2repo.sh
    ├── 05_replace_trt_encoder_and_canary_bls.sh
    ├── 06_start_riva_or_nim_server.sh
    ├── 07_riva-client-grpc.sh
    ├── 08_riva-client-http-just-test.sh
    ├── 09_merge-gt-and-transcript.sh
    ├── 10_eval.sh
    ├── 11_batch-canary-nemo-infer.sh
    ├── 12_sort-dataset.sh
    │
    ├── shuffle_dataset_utils/
    ├── NeMo/
    └── nemo2riva/

```

---

# Prerequisites

## Containers

The workflow assumes the environment is launched using:

* `nemo.sh` – nemo2riva and checkpoint conversion environment
* `riva_trt.sh` – TensorRT-LLM/Riva deployment environment

These scripts are located outside the `CanaryNIM-Bharat-Final` directory and are used to start the required containers.

## Dependencies

### NeMo

Custom NeMo fork:

https://github.com/Bodhan-NeMo/NeMo

### nemo2riva

Custom nemo2riva fork:

https://github.com/bgiddwani-ai/nemo2riva

---

# Workflow

## Step 1: Convert NeMo Checkpoint to Riva + TRT-LLM Checkpoint

Run:

```bash
bash 01_nemo2riva.sh
```

This script:

1. Installs required Python packages.
2. Installs custom `nemo2riva`.
3. Reinstalls the custom NeMo fork.
4. Converts `.nemo` → `.riva`.
5. Converts the NeMo checkpoint into TensorRT-LLM checkpoint format.

Inputs:

```text
/data/indic-canary.nemo
```

Outputs:

```text
/data/indic-canary.riva
/data/indic-canary-trtllm-checkpoint
/data/indic-canary-trtllm-engine
```

---

## Step 2: Build TensorRT Encoder Engine

Run:

```bash
bash 02_conformer_onnx_to_trt.sh
```

This script builds the Conformer encoder TensorRT engine.

Output:

```text
/data/indic-canary-trtllm-engine
```

Configured for:

* Max Batch Size = 64
* Opt Batch Size = 64
* Max Feature Length = 3001

---

## Step 3: Build RMIR

Run:

```bash
bash 03_riva2rmir.sh
```

This creates the Riva RMIR package using:

* TensorRT-LLM decoder
* Canary decoder
* Offline ASR pipeline

Configured for:

* 64 batch inference
* 24 Indic languages
* Decoupled TRT-LLM mode

Output:

```text
/data/indic-canary-trt-bs64-decoupled.rmir
```

---

## Step 4: Deploy RMIR to Triton Repository

Run:

```bash
bash 04_rmir2repo.sh
```

This converts RMIR into a Triton model repository.

Output:

```text
/data/models
```

---

## Step 5: Replace Encoder and Canary BLS

Run:

```bash
bash 05_replace_trt_encoder_and_canary_bls.sh
```

This step:

1. Replaces the generated encoder engine with the optimized TensorRT encoder.
2. Copies the custom `canary.py` Business Logic Script into the deployed model.

---

## Step 6: Start Riva / NIM Server

Run:

```bash
bash 06_start_riva_or_nim_server.sh
```

Default:

```bash
start-riva \
  --asr_service=true \
  --nlp_service=false \
  --tts_service=false
```

Alternative NIM startup commands are included but commented out.

---

# Inference

## Step 7: Riva gRPC Client

Run:

```bash
bash 07_riva-client-grpc.sh
```

Features:

* Parallel requests = 64
* Hindi language evaluation
* Manifest-based batch processing

Output:

```text
64_indicvoices_random_indic_canary_nim_riva_trt.json
```

---

## Step 8: HTTP Test Client

Run:

```bash
bash 08_riva-client-http-just-test.sh
```

Executes:

```bash
python3 nim-riva-client-http.py
```

Useful for quick endpoint validation.

---

# Evaluation

## Step 9: Merge Ground Truth and Predictions

Run:

```bash
bash 09_merge-gt-and-transcript.sh
```

Creates a merged manifest containing:

* Ground truth transcripts
* Model predictions

Output:

```text
hi_manifest_clean_final_inference_results.json
```

---

## Step 10: Compute WER / CER

Run:

```bash
bash 10_eval.sh
```

Uses NeMo evaluation scripts:

```bash
speech_to_text_eval.py
```

Evaluates:

1. Riva/TensorRT deployment results
2. Native NeMo inference results

Metrics:

* CER
* WER (if configured)

---

# Native NeMo Baseline

## Step 11: Batch NeMo Inference

Run:

```bash
bash 11_batch-canary-nemo-infer.sh
```

Performs direct NeMo inference without Riva/TensorRT deployment.

Input:

```text
hi_manifest_cleaned_final.json
```

Output:

```text
hi_manifest_cleaned_final_nemo_prediction.json
```

Used as a baseline comparison against deployed inference.

---

# Dataset Utilities

## Step 12: Dataset Sorting / Sampling

Run:

```bash
bash 12_sort-dataset.sh
```

Generates:

```text
*_sorted_mf.json
*_worst_mf.json
*_sampled_mf.json
```

Useful for:

* Error analysis
* Sampling datasets
* Performance benchmarking

---

# End-to-End Execution Order

Run the scripts sequentially:

```text
01_nemo2riva.sh
02_conformer_onnx_to_trt.sh
03_riva2rmir.sh
04_rmir2repo.sh
05_replace_trt_encoder_and_canary_bls.sh
06_start_riva_or_nim_server.sh
07_riva-client-grpc.sh
08_riva-client-http-just-test.sh
09_merge-gt-and-transcript.sh
10_eval.sh
```

Optional:

```text
11_batch-canary-nemo-infer.sh
12_sort-dataset.sh
```

---
