import json
import os
import gzip
import shutil

from plugins.robothor_plugin.robothor_task_samplers import ObjectNavDatasetTaskSampler

SCENE = "FloorPlan_Train1_1"
TARGET = "Television"
EPISODES = [0, 7, 11, 12]
BASE_OUT = os.path.join("datasets", "robothor-objectnav", "debug")

train_path = os.path.join("datasets", "robothor-objectnav", "train")
downloaded_caches = os.path.join(train_path, "distance_caches", SCENE + ".json.gz")
downloaded_episodes = os.path.join(train_path, "episodes", SCENE + ".json.gz")

assert os.path.exists(downloaded_caches) and os.path.exists(
    downloaded_episodes
), "make sure to download the robothor objectnav dataset with datasets/download_navigation_datasets.sh"

# caches
caches_dir = os.path.join(BASE_OUT, "distance_caches")
os.makedirs(caches_dir, exist_ok=True)
caches_file = os.path.join(caches_dir, SCENE + ".json.gz")
if os.path.exists(caches_file):
    print("Overwriting previous {}".format(caches_file))
    os.remove(caches_file)
shutil.copy(downloaded_caches, caches_file)
assert os.path.exists(caches_file)

# episodes
episodes = ObjectNavDatasetTaskSampler._load_dataset(
    scene=SCENE, base_directory=os.path.join(train_path, "episodes")
)

target_episodes = [ep for ep in episodes if ep["object_type"] == TARGET]

ids = ["{}_{}_{}".format(SCENE, TARGET, epit) for epit in EPISODES]
debug_episodes = [ep for ep in target_episodes if ep["id"] in ids]

# sort by episode_ids
debug_episodes = [
    idep[1]
    for idep in sorted(
        [(int(ep["id"].split("_")[-1]), ep) for ep in debug_episodes],
        key=lambda x: x[0],
    )
]
assert len(debug_episodes) == len(EPISODES)

episodes_dir = os.path.join(BASE_OUT, "episodes")
os.makedirs(episodes_dir, exist_ok=True)
episodes_file = os.path.join(episodes_dir, SCENE + ".json.gz")

json_str = json.dumps(debug_episodes)
json_bytes = json_str.encode("utf-8")
with gzip.GzipFile(episodes_file, "w") as fout:
    fout.write(json_bytes)
assert os.path.exists(episodes_file)

read = ObjectNavDatasetTaskSampler._load_distance_cache(SCENE, episodes_dir)

assert sorted([ep["id"] for ep in read]) == sorted([ep["id"] for ep in debug_episodes])
