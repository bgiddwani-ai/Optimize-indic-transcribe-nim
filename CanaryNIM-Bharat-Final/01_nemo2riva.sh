pip install nvidia-pyindex
pip install --extra-index-url https://pypi.nvidia.com nvidia-eff
cd /home/CanaryNIM-Bharat-Final/nemo2riva
pip install --no-deps .
cd /home/CanaryNIM-Bharat-Final/NeMo
pip install '.[all]'           
export PYTHONPATH=/home/CanaryNIM-Bharat-Final/NeMo:$PYTHONPATH
export PYTHONPATH=/home/CanaryNIM-Bharat-Final/nemo2riva:$PYTHONPATH
nemo2riva --key=bodhan --out /data/indic-canary.riva /data/indic-transcribe-flex/nemo/indic_transcribe_flex.nemo

cd ../
python3 convert_checkpoint.py \
	--dtype=bfloat16 \
	--model_name "nvidia/canary-1b" \
        --model_path /data/indic-transcribe-flex/nemo/indic_transcribe_flex.nemo  \
        --output_dir /data/indic-canary-trtllm-checkpoint \
	/data/indic-canary-trtllm-engine
