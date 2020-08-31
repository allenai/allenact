#!/bin/bash

# Move to the directory containing this file
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" || exit

# Download, Unzip, and Remove zip
if [ "$1" = "robothor-pointnav-rgb-resnet" ]
then
    echo "Downloading RoboTHOR PointNav Dataset ..."
    wget https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/robothor-pointnav-rgb-resnet.pt
    tar -xf robothor-pointnav-rgb-resnet-weights.tar.gz && rm robothor-pointnav-rgb-resnet-weights.tar.gz
    echo "saved folder: robothor-pointnav-rgb-resnet"

else
    echo "Failed: Usage download_navigation_model_ckpts.sh robothor-pointnav-rgb-resnet"
    exit 1
fi
