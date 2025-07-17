
#python3.11 with torch gpu
sudo apt-get install ffmpeg
pip install torch torchvision torchaudio torchcodec --use-deprecated=legacy-resolver  --no-cache-dir -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
pip install datasets torchcodec decord wheel transformers==4.50.0 deepspeed
pip install flash-attn==2.2.0 --no-build-isolation
