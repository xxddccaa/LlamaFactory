# https://hub.docker.com/r/hiyouga/pytorch/tags
ARG BASE_IMAGE=hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel
FROM ${BASE_IMAGE}

# Installation arguments
ARG PIP_INDEX=https://pypi.org/simple
ARG INSTALL_FLASHATTN=false
ARG HTTP_PROXY=""

# Define environments
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV http_proxy="${HTTP_PROXY}"
ENV https_proxy="${HTTP_PROXY}"

# Chinese locale / timezone
ENV LANG=zh_CN.UTF-8
ENV LC_ALL=zh_CN.UTF-8
ENV LANGUAGE=zh_CN:zh
ENV TZ=Asia/Shanghai

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /app

# Install locale/fonts/tools (optional but convenient)
RUN apt-get update && apt-get install -y \
    locales \
    fonts-noto-cjk \
    language-pack-zh-hans \
    tzdata \
    openssh-server \
    zip \
    unzip \
    tree \
    vim \
    htop \
    tmux \
    curl \
    wget \
    git \
    file \
    net-tools \
    libibverbs1 \
    libibverbs-dev \
    libgl1-mesa-glx \
    libglx-mesa0 \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen zh_CN.UTF-8 && \
    update-locale LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8 && \
    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools editables "hatchling>=1.18.0"

# Copy the application into the image
COPY . /app

# Rebuild flash attention
RUN if [ "${INSTALL_FLASHATTN}" == "true" ]; then \
        pip uninstall -y ninja && \
        pip install --no-cache-dir ninja && \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi

RUN apt-get update && apt-get install -y ffmpeg sox libsox-dev

RUN pip install --no-cache-dir --no-build-isolation swanlab soundfile librosa audioread

RUN pip install --no-cache-dir --no-build-isolation -r requirements/minicpm-v.txt


# Install LLaMA Factory
RUN pip install --no-cache-dir --no-build-isolation --upgrade-strategy only-if-needed -e . && \
    pip install --no-cache-dir --no-build-isolation --upgrade-strategy only-if-needed -r requirements/metrics.txt -r requirements/deepspeed.txt


# Customize bash prompt
RUN echo 'export PS1="\\[\\e[1;31m\\]\\D{%a %m月 %d} \\[\\e[0m\\]# \\[\\e[1;32m\\]\\t \\[\\e[0m\\]# \\[\\e[1;34m\\]\\w \\[\\e[0m\\]# "' >> /root/.bashrc


# Set up volumes
# VOLUME [ "/root/.cache/huggingface", "/app/shared_data", "/app/output" ]

# Expose port 7860 for LLaMA Board
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

# Expose port 8000 for API service
ENV API_PORT=8000
EXPOSE 8000

# unset proxy
ENV http_proxy=
ENV https_proxy=

# Reset pip config
RUN pip config unset global.index-url && \
    pip config unset global.extra-index-url
# docker build -t kevinchina/deeplearning:llamafactory-0.9.5.dev0 -f ./docker/docker-cuda/Dockerfile .