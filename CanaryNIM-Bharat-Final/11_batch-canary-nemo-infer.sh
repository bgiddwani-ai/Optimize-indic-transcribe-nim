export PYTHONPATH=/home/CanaryNIM-Bharat-Final/NeMo:$PYTHONPATH
python3 batch-canary-nemo-infer.py  --ckpt_path '/home/CanaryNIM-Bharat-Final/indic-canary.nemo' --manifest_path /home/CanaryNIM-Bharat-Final/dataset/hi_manifest_cleaned_final.json --language_code hi --batch_size 64 --output_path '/home/CanaryNIM-Bharat-Final/dataset/hi_manifest_cleaned_final_nemo_prediction.json'
