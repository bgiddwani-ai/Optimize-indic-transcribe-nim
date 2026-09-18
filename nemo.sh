docker run -it --ipc=host -p 8000:8000 --gpus '"device=0"' -v $PWD/CanaryNIM-Bharat-Final:/home/CanaryNIM-Bharat-Final -v /data:/data nvcr.io/nvidia/nemo:26.02
