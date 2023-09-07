FROM nvidia/cuda:12.1.0-base-ubuntu20.04
RUN apt update
RUN apt install -y git python3 python3-pip wget
WORKDIR /controlnet
COPY controlnet.py /controlnet/
COPY requirements.txt /controlnet/
RUN pip install -r requirements.txt
RUN pip install scipy
CMD ["python","controlnet.py"]
