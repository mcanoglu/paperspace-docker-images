# Paperspace Dockerfile for Gradient base image

# ==================================================================
# Initial setup
# ------------------------------------------------------------------

# Ubuntu 20.04 as base image
FROM ubuntu:20.04
RUN yes| unminimize

# Set ENV variables
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive





# ==================================================================
# Tools
# ------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    gcc \
    make \
    pkg-config \
    apt-transport-https \
    build-essential \
    ca-certificates \
    wget \
    rsync \
    git \
    vim \
    mlocate \
    libssl-dev \
    curl \
    openssh-client \
    unzip \
    unrar \
    zip \
    csvkit \
    emacs \
    joe \
    jq \
    dialog \
    man-db \
    manpages \
    manpages-dev \
    manpages-posix \
    manpages-posix-dev \
    nano \
    iputils-ping \
    sudo \
    ffmpeg \
    libsm6 \
    libxext6 \
    libboost-all-dev \
    cifs-utils \
    software-properties-common \
    python3 \
    python-is-python3


# ==================================================================
# Python
# ------------------------------------------------------------------

#Based on https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa

# ==================================================================
# Installing CUDA packages (CUDA Toolkit 12.1.0 & CUDNN 8.4.1)
# ------------------------------------------------------------------

# Based on https://developer.nvidia.com/cuda-toolkit-archive
# Based on https://developer.nvidia.com/rdp/cudnn-archive
# Based on https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install

# Installing CUDA Toolkit

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda

ENV PATH=$PATH:/usr/local/cuda-12.1/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64

# Installing CUDNN
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-get install dirmngr -y && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    apt-get update && \
    apt-get install libcudnn8=8.9.0.*-1+cuda12.1 -y && \
    apt-get install libcudnn8-dev=8.9.0.*-1+cuda12.1 -y && \
    rm /etc/apt/preferences.d/cuda-repository-pin-600


# ==================================================================
# PyTorch
# ------------------------------------------------------------------

# Based on https://pytorch.org/get-started/locally/

RUN python -m pip3 --no-cache-dir install --upgrade \
        torch \
        torchvision \
        torchaudio \ 
        torchtext \
        tensorflow \
        datasets \
        jupyterlab \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image \
        matplotlib \
        ipython \
        ipykernel \
        ipywidgets \
        cython \
        tqdm \
        gdown \
        xgboost \
        pillow \
        seaborn \
        sqlalchemy \
        spacy \
        nltk \
        boto3 \
        tabulate \
        future \
        gradient \
        jsonify \
        opencv-python \
        sentence-transformers \
        wandb \
        awscli \
        jupyterlab-snippets \
        tornado \
        transformers[torch] \
        scikit-learn

# ==================================================================
# Installing JRE and JDK
# ------------------------------------------------------------------

RUN apt-get install -y --no-install-recommends \
    default-jre \
    default-jdk


# ==================================================================
# CMake
# ------------------------------------------------------------------

RUN git clone --depth 10 https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install


# ==================================================================
# Node.js and Jupyter Notebook Extensions
# ------------------------------------------------------------------

RUN curl -sL https://deb.nodesource.com/setup_16.x | bash  && \
    apt-get install -y --no-install-recommends nodejs  && \
    python -m pip --no-cache-dir install --upgrade \
        jupyter_contrib_nbextensions \
        jupyterlab-git \
        jupyter \
        jupyter contrib \
        nbextension \
        install --user


# ==================================================================
# Startup
# ------------------------------------------------------------------

EXPOSE 8888 6006

CMD jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True