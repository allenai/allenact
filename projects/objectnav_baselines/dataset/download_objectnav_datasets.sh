#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "robothor" ]
then

    echo "Downloading RoboTHOR Objectnav Dataset ..."
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/robothor-objectnav.tar.gz
    tar -xf robothor-objectnav.tar.gz && rm robothor-objectnav.tar.gz
    echo "saved folder: robothor"

elif [ "$1" = "ithor" ]
then

    echo "Downloading iTHOR Objectnav Dataset ..."
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/ithor-objectnav.tar.gz
    tar -xf ithor-objectnav.tar.gz && rm ithor-objectnav.tar.gz
    echo "saved folder: ithor"


else
    echo "Failed: Usage download_objectnav_datasets.sh ithor | robothor"
    exit 1
fi
