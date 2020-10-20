#!/bin/bash

# Move to the directory containing this file
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" || exit

mkdir -p habitat
mkdir -p habitat/scene_datasets
mkdir -p habitat/datasets
mkdir -p habitat/configs

cd habitat || exit

output_archive_name=__TO_OVERWRITE__.zip
deletable_dir_name=__TO_DELETE__

install_test_scenes_and_data() {
    if ! wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip -O $output_archive_name; then
      echo "Could not unzip download test scenes from http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip"
      exit 1
    fi
    if ! unzip $output_archive_name -d $deletable_dir_name; then
      echo "Could not unzip $output_archive_name to $deletable_dir_name"
      exit 1
    fi
    rsync -avz $deletable_dir_name/data/datasets . && \
    rsync -avz $deletable_dir_name/data/scene_datasets . && \
    rm $output_archive_name && \
    rm -r $deletable_dir_name
}

install_scene_data() {
  python3 ../.habitat_downloader_helper.py "$1"
}

if [ "$1" = "test-scenes" ]
then
  install_test_scenes_and_data

else
  install_scene_data $1
fi

