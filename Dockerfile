FROM python:3.10.8-slim-bullseye

RUN apt-get update -qq && \
    apt-get install -y zip unzip htop screen libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/cache/apk/*

COPY requirements.txt /requirements.txt
RUN pip --no-cache-dir install -r /requirements.txt

WORKDIR /workspace
CMD ["bash"]
