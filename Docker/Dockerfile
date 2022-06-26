FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

#############################################
# Base installations
#############################################
ENV DEBIAN_FRONTEND noninteractive

# Install some basic utilities
RUN apt update && apt install -y \
    curl \
    nano \
    ca-certificates \
    sudo \
    git \
    openssh-server \
    software-properties-common \
    build-essential \
    python3-tk \
    python3-wheel \
    libhdf5-dev \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip &&\
    apt-get clean &&\
    ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
        ln -s /usr/bin/python3.8 /usr/local/bin/python3 &&\    
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
 
#############################################
# Budgie installations
#############################################

ENV DEBIAN_FRONTEND noninteractive
ENV TZ 'Europe/Ljubljana'

RUN apt-get update && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y tzdata \
					tree && \
    rm /etc/localtime && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

#############################################
# Pytorch and other
#############################################

# Install essential Python packages
RUN rm -fr ~/.cache/pip && rm -fr /usr/local/lib/python3.8
RUN python3 -m pip --no-cache-dir install \
    numpy \
    matplotlib \
    Pillow \
    PyYAML \
    tqdm \
    einops \
    torch \
    torchvision 
    