#!/usr/bin/env bash

pipenv run python rl_robothor/docker/startx.py &> ~/logxserver &
pipenv run tensorboard --logdir ~/work/data/embodied-rl/ --bind_all &> ~/logtensorboard &

## weird bug with cuda 9.0 expected in torchvision 0.3.0
#echo "docker_entrypoint.sh: Forcing torchvision 0.3.0"
#pipenv uninstall --skip-lock torchvision
#pipenv install --skip-lock torchvision~=0.3.0

export PYTHONPATH=../ai2thor:$PYTHONPATH
echo "PYTHONPATH $PYTHONPATH"
pipenv shell
