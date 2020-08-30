# Downloading datasets 

## PointNav
### RoboTHOR
To get the PointNav dataset and precomputed distance caches for `RoboTHOR` run the following command:
```bash
sh datasets/download_navigation_datasets.sh robothor-pointnav
```
This will download the dataset into `datasets/robothor-pointnav`
### iTHOR
To get the PointNav dataset and precomputed distance caches for `iTHOR` run the following command:
```bash
sh datasets/download_navigation_datasets.sh ithor-pointnav
```
This will download the dataset into `datasets/ithor-pointnav`

### Habitat
To get the PointNav `habitat` dataset download and install the `allenact-habitat` docker
container as described in [this tutorial](installation-framework.md). The dataset is
included in the docker image.

## ObjectNav
### RoboTHOR
To get the ObjectNav dataset and precomputed distance caches for `RoboTHOR` run the following command:

```bash
sh datasets/download_navigation_datasets.sh robothor-objectnav
```
This will download the dataset into `datasets/robothor-objectnav`
### iTHOR
To get the ObjectNav dataset and precomputed distance caches for `iTHOR` run the following command:
```bash
cd datasets
sh datasets/download_navigation_datasets.sh robothor-pointnav
```
This will download the dataset into `datasets/ithor-objectnav`
