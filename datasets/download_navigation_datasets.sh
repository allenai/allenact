#!/bin/bash

# Move to the directory containing this file
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" || exit

install_dataset() {
    dataset_name="$1"
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/"$dataset_name".tar.gz
    mkdir "$dataset_name"
    tar -xf "$dataset_name".tar.gz -C "$dataset_name" --strip-components=1 && rm "$dataset_name".tar.gz
    echo "saved folder: "$dataset_name""
}


# Download, Unzip, and Remove zip
if [ "$1" = "robothor-pointnav" ]
then
    echo "Downloading RoboTHOR PointNav Dataset ..."
    install_dataset "$1"

elif [ "$1" = "robothor-objectnav" ]
then
    echo "Downloading RoboTHOR ObjectNav Dataset ..."
    install_dataset "$1"
    cd ..
    echo "Generating RoboTHOR ObjectNav Debug Dataset ..."
    PYTHONPATH=. python ./plugins/robothor_plugin/scripts/make_objectnav_debug_dataset.py

elif [ "$1" = "ithor-pointnav" ]
then
    echo "Downloading iTHOR PointNav Dataset ..."
    install_dataset "$1"

elif [ "$1" = "ithor-objectnav" ]
then
    echo "Downloading iTHOR ObjectNav Dataset ..."
    install_dataset "$1"
    cd ..
    echo "Generating iTHOR ObjectNav Debug Dataset ..."
    PYTHONPATH=. python ./plugins/ithor_plugin/scripts/make_objectnav_debug_dataset.py

else
    echo "Failed: Usage download_navigation_datasets.sh robothor-pointnav | robothor-objectnav | ithor-pointnav | ithor-objectnav"
    exit 1
fi
