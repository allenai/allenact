# From https://github.com/allenai/robothor-challenge/blob/master/Dockerfile

ARG CUDA_VERSION

FROM nvidia/cuda:$CUDA_VERSION-devel-ubuntu18.04

ARG NVIDIA_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install python3-pip libxrender1 libsm6 xserver-xorg-core xorg python3-venv vim pciutils wget git module-init-tools tzdata

COPY rl_robothor/docker/install_nvidia.sh /
RUN NVIDIA_VERSION=$NVIDIA_VERSION /bin/bash /install_nvidia.sh

RUN pip3 install pipenv==2018.11.26
COPY Pipfile /embodied-rl/
WORKDIR /embodied-rl
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN pipenv install --skip-lock --dev

WORKDIR /
RUN git clone https://github.com/allenai/ai2thor && cd /ai2thor && git checkout 27d67618acfb3d89f4bf23aa30ba2280d54b19c2

WORKDIR /embodied-rl
RUN pipenv install --skip-lock flask numpy pyyaml requests progressbar2 msgpack Pillow
RUN PYTHONPATH=../ai2thor:$PYTHONPATH pipenv run python3 -c "import ai2thor.controller; ai2thor.controller.Controller(download_only=True)"

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . /embodied-rl/
# run in interactive session: pipenv shell
ENTRYPOINT ["rl_robothor/docker/docker_entrypoint.sh"]
