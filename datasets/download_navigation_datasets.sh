#!/bin/bash

# Move to the directory containing this file
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" || exit

install_dataset() {
    dataset_name="$1"
    if ! mkdir "$dataset_name" ; then
      echo "Could not create directory " $(pwd)/$dataset_name "Does it already exist? If so, delete it."
      exit 1
    fi
    url_archive_name=$dataset_name-v0.tar.gz
    output_archive_name=__TO_OVERWRITE__.tar.gz
    wget https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/$url_archive_name -O $output_archive_name
    tar -xf "$output_archive_name" -C "$dataset_name" --strip-components=1 && rm $output_archive_name
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

elif [ "$1" = "all-thor" ]
then
    bash download_navigation_datasets.sh "robothor-pointnav"
    bash download_navigation_datasets.sh "robothor-objectnav"
    bash download_navigation_datasets.sh "ithor-pointnav"
    bash download_navigation_datasets.sh "ithor-objectnav"

else
    echo "\nFailed: Usage download_navigation_datasets.sh robothor-pointnav | robothor-objectnav | ithor-pointnav | ithor-objectnav | all-thor"
    exit 1
fi
