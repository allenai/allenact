# Downloading datasets 

## PointNav
### RoboTHOR
To get the PointNav dataset and precomputed distance caches for `RoboTHOR` run the following command:

```bash
cd projects/pointnav_baselines/dataset
sh download_pointnav_dataset.sh robothor
```

### iTHOR
To get the PointNav dataset and precomputed distance caches for `iTHOR` run the following command:

```bash
cd projects/pointnav_baselines/dataset
sh download_pointnav_dataset.sh ithor
```

### Habitat
To get the PointNav `habitat` dataset download and install the `allenact-habitat` docker
container as described in [this tutorial](installation-framework.md). The dataset is
included in the docker iage

## ObjectNav
### RoboTHOR
To get the ObjectNav dataset and precomputed distance caches for `RoboTHOR` run the following command:

```bash
cd projects/objectnav_baselines/dataset
sh download_objectnav_dataset.sh robothor
```

### iTHOR
To get the ObjectNav dataset and precomputed distance caches for `iTHOR` run the following command:

```bash
cd projects/objectnav_baselines/dataset
sh download_objectnav_dataset.sh ithor
```
