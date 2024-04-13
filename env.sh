conda create -y -n digital-human python=3.10
conda activate digital-human
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install gradio opencv-python sentence_transformers einops
