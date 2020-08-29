#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "robothor-pointnav" ]
then
    echo "Downloading RoboTHOR Pointnav Dataset ..."
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/robothor-pointnav.tar.gz
    tar -xf robothor-pointnav.tar.gz && rm robothor-pointnav.tar.gz
    echo "saved folder: robothor-pointnav"

elif [ "$1" = "robothor-objectnav" ]
then
    echo "Downloading RoboTHOR Objectnav Dataset ..."
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/robothor-objectnav.tar.gz
    tar -xf robothor-objectnav.tar.gz && rm robothor-objectnav.tar.gz
    echo "saved folder: robothor-objectnav"

elif [ "$1" = "ithor-pointnav" ]
then
    echo "Downloading iTHOR Pointnav Dataset ..."
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/ithor-pointnav.tar.gz
    tar -xf ithor-pointnav.tar.gz && rm ithor-pointnav.tar.gz
    echo "saved folder: ithor-pointnav"

elif [ "$1" = "ithor-objectnav" ]
then
    echo "Downloading iTHOR Objectnav Dataset ..."
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/ithor-objectnav.tar.gz
    tar -xf ithor-objectnav.tar.gz && rm ithor-objectnav.tar.gz
    echo "saved folder: ithor-objectnav"

else
    echo "Failed: Usage download_datasets.sh robothor-pointnav | robothor-objectnav | ithor-pointnav | ithor-objectnav"
    exit 1
fi
