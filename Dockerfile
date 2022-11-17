FROM python:3.10.8-slim-bullseye

RUN apt-get update -qq && \
    apt-get install -y zip unzip htop screen libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/cache/apk/*

COPY requirements/* /requirements
RUN pip --no-cache-dir install -r /requirements-main.txt
RUN pip --no-cache-dir install -r /requirements-test.txt
RUN pip --no-cache-dir install -r /requirements-dev.txt
RUN pip --no-cache-dir install -r /requirements-streamlit.txt

WORKDIR /workspace
CMD ["bash"]
