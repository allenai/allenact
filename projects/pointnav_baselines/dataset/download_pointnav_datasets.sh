#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "robothor" ]
then

    echo "Downloading RoboTHOR Pointnav Dataset ..."
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/robothor-pointnav.tar.gz
    tar -xf robothor-pointnav.tar.gz && rm robothor-pointnav.tar.gz
    echo "saved folder: robothor"

elif [ "$1" = "ithor" ]
then

    echo "Downloading iTHOR Pointnav Dataset ..."
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/ithor-pointnav.tar.gz
    tar -xf ithor-pointnav.tar.gz && rm ithor-pointnav.tar.gz
    echo "saved folder: ithor"


else
    echo "Failed: Usage download_data.sh json | json_feat | full"
    exit 1
fi
