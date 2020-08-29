#!/bin/bash

# Move to the directory containing this file
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" || exit

# Download, Unzip, and Remove zip
if [ "$1" = "robothor-pointnav-rgb-resnet" ]
then
    echo "Downloading RoboTHOR Pointnav Dataset ..."
    wget https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/robothor-pointnav-rgb-resnet-resnet.tar.gz
    tar -xf robothor-objectnav-rgb-resnet-weights.tar.gz && rm robothor-objectnav-rgb-resnet-weights.tar.gz
    echo "saved folder: robothor-objectnav-rgb-resnet"

else
    echo "Failed: Usage download_navigation_datasets.sh robothor-objectnav-rgb-resnet"
    exit 1
fi
