conda create -y -n digital-human python=3.10
conda activate digital-human
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install gradio opencv-python sentence_transformers einops imageio==2.9.0 imageio-ffmpeg omegaconf diffusers==0.21.4 accelerate xformers av
