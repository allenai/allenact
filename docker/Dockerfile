# Base image
FROM nvidia/cudagl:9.0-runtime-ubuntu16.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n embodied-rl python=3.6

# Grab embodied-rl code and make folder for habitat
RUN git clone https://klemenkotar:PASSWORD@github.com/klemenkotar/embodied-rl.git
RUN cd embodied-rl && mkdir habitat

# Setup habtiat-sim
RUN cd embodied-rl/habitat && git clone https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate embodied-rl; cd embodied-rl/habitat/habitat-sim; git checkout 62bc01be8cbc148eb099e96fa0b123e16b21f95f; python setup.py install --headless"

# Install challenge specific habitat-api
RUN cd embodied-rl/habitat && git clone https://github.com/klemenkotar/habitat-api.git
RUN /bin/bash -c ". activate embodied-rl; cd embodied-rl/habitat/habitat-api; git checkout e6851021990ec43350fdbcd2862321c67ebbe512; pip install -e ."

# Install embodied-rl and thor
RUN /bin/bash -c ". activate embodied-rl; cd embodied-rl; git checkout ab1a7ef4196ac6e19aef3d317ad54ed852550796; pip install -r requirements.txt"

RUN export PYTHONPATH=$PYTHONPATH:/embodied-rl/habitat/habitat-sim/

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"
