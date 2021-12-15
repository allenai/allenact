#!/bin/bash

# Move to the directory containing this file
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" || exit

# Download, Unzip, and Remove zip
if [ "$1" = "robothor-pointnav-rgb-resnet" ]
then
    echo "Downloading pretrained RoboTHOR PointNav model..."
    wget https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/robothor-pointnav-rgb-resnet.tar.gz
    tar -xf robothor-pointnav-rgb-resnet.tar.gz && rm robothor-pointnav-rgb-resnet.tar.gz
    echo "saved folder: robothor-pointnav-rgb-resnet"
elif [ "$1" = "robothor-objectnav-challenge-2021" ]
then
    echo "Downloading pretrained RoboTHOR ObjectNav model..."
    wget https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/robothor-objectnav-challenge-2021.tar.gz
    tar -xf robothor-objectnav-challenge-2021.tar.gz && rm robothor-objectnav-challenge-2021.tar.gz
    echo "saved folder: robothor-objectnav-challenge-2021"
elif [ "$1" = "armpointnav-disturb-free-2022" ]
then
    echo "Downloading pretrained ArmPointNav disturb-free models..."
    wget <>
    tar -xf <> && rm <>
    echo "saved folder: armpointnav-disturb-free-2022"
else
    echo "Failed: Usage download_navigation_model_ckpts.sh robothor-objectnav-challenge-2021"
    exit 1
fi
