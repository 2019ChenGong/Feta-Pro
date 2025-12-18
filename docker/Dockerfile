FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
RUN apt update
RUN apt install -y screen
RUN apt install -y zip
RUN apt install -y build-essential
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install wget
# RUN apt-get -y install libopenmpi-dev

WORKDIR /workspace
RUN git clone https://github.com/2019ChenGong/Feta-Pro.git

WORKDIR /workspace/Feta-Pro
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0
RUN pip install -r requirements.txt;

WORKDIR /workspace/Feta-Pro/opacus_
RUN pip install -e .

WORKDIR /workspace/Feta-Pro/models/PE
RUN pip install improved-diffusion@git+https://github.com/fjxmlzn/improved-diffusion.git@8f6677c3c47d1c1ad2e22ad2603eaec4cc639805
RUN git clone https://github.com/openai/guided-diffusion.git
WORKDIR /workspace/Feta-Pro/models/PE/guided-diffusion
RUN pip install -e .

WORKDIR /workspace/Feta-Pro/models/DP_LORA/peft
RUN pip install -e .
RUN pip install transformers==4.27.4

WORKDIR /workspace/Feta-Pro/models
RUN gdown https://drive.google.com/uc?id=1yVTWzaSqJVDJy8CsZKtqDoBNeM6154D4
RUN unzip pretrained_models.zip

WORKDIR /workspace/Feta-Pro

# docker build -t feta-pro:latest .
# docker run -it --gpus all feta-pro:latest bash