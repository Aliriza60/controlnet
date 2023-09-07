FROM nvidia/cuda:12.1.0-base-ubuntu20.04
RUN apt update
RUN apt install -y git python3 python3-pip wget
WORKDIR /controlnet
COPY controlnet.py /controlnet/
RUN pip3 install torch torchvision torchaudio
RUN pip install diffusers["torch"] transformers
RUN pip install accelerate
RUN pip install scipy
RUN pip install safetensors
RUN pip install xformers
RUN pip install numpy
RUN pip install opencv-contrib-python
RUN pip install pillow
RUN pip install controlnet-aux
CMD ["python","controlnet.py"]
