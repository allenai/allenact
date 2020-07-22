import ai2thor.controller
import ai2thor.util.metrics as metrics
import random
import gzip
import json
from tqdm import tqdm


controller = ai2thor.controller.Controller(agentMode='bot')

EPISODES_PER_SCENE = 1800
ROTATION_DEGREES = 45

SCENES = ["FloorPlan_Train1_1", "FloorPlan_Train1_2", "FloorPlan_Train1_3", "FloorPlan_Train1_4", "FloorPlan_Train1_5",
          "FloorPlan_Train2_1", "FloorPlan_Train2_2", "FloorPlan_Train2_3", "FloorPlan_Train2_4", "FloorPlan_Train2_5",
          "FloorPlan_Train3_1", "FloorPlan_Train3_2", "FloorPlan_Train3_3", "FloorPlan_Train3_4", "FloorPlan_Train3_5",
          "FloorPlan_Train4_1", "FloorPlan_Train4_2", "FloorPlan_Train4_3", "FloorPlan_Train4_4", "FloorPlan_Train4_5",
          "FloorPlan_Train5_1", "FloorPlan_Train5_2", "FloorPlan_Train5_3", "FloorPlan_Train5_4", "FloorPlan_Train5_5",
          "FloorPlan_Train6_1", "FloorPlan_Train6_2", "FloorPlan_Train6_3", "FloorPlan_Train6_4", "FloorPlan_Train6_5",
          "FloorPlan_Train7_1", "FloorPlan_Train7_2", "FloorPlan_Train7_3", "FloorPlan_Train7_4", "FloorPlan_Train7_5",
          "FloorPlan_Train8_1", "FloorPlan_Train8_2", "FloorPlan_Train8_3", "FloorPlan_Train8_4", "FloorPlan_Train8_5",
          "FloorPlan_Train9_1", "FloorPlan_Train9_2", "FloorPlan_Train9_3", "FloorPlan_Train9_4", "FloorPlan_Train9_5",
          "FloorPlan_Train10_1", "FloorPlan_Train10_2", "FloorPlan_Train10_3", "FloorPlan_Train10_4", "FloorPlan_Train10_5",
          "FloorPlan_Train11_1", "FloorPlan_Train11_2", "FloorPlan_Train11_3", "FloorPlan_Train11_4", "FloorPlan_Train11_5",
          "FloorPlan_Train12_1", "FloorPlan_Train12_2", "FloorPlan_Train12_3", "FloorPlan_Train12_4", "FloorPlan_Train12_5"]

OBJECTS = ['AlarmClock', 'Apple', 'BaseballBat', 'BasketBall', 'Bowl', 'GarbageCan', 'HousePlant', 'Laptop',
           'Mug', 'SprayBottle', 'Television', 'Vase']

ROTATIONS = [{"x": 0.0, "y": float(y), "z": 0.0} for y in range(0, 360, ROTATION_DEGREES)]

episodes = {scene: [] for scene in SCENES}
ep_lengths = []

for scene in tqdm(SCENES):
    controller.reset(scene=scene)
    event = controller.step(action='GetReachablePositions')
    all_reachable_points = event.metadata['actionReturn']

    import matplotlib.pyplot as plt
    xs = [pos["x"] for pos in all_reachable_points]
    ys = [pos["z"] for pos in all_reachable_points]
    plt.scatter(xs, ys)
    chosen_xs = []
    chosen_ys = []

    objects = event.metadata['objects']
    agent = event.metadata['agent']
    for o in OBJECTS:

        # Fetch the instance of the object class and assure it exists
        object_instance = None
        for oi in objects:
            if oi['objectType'] == o:
                object_instance = oi
                break

        for ep in range(EPISODES_PER_SCENE // len(OBJECTS)):
            found_valid_point = False
            while not found_valid_point:
                # Select a random point and starting orientation
                position = random.choice(all_reachable_points)  # all_reachable_points.pop(random.randint(0, len(all_reachable_points)-1))
                chosen_xs.append(position["x"])
                chosen_ys.append(position["z"])
                rotation = random.choice(ROTATIONS)

                try:
                    path = metrics.get_shortest_path_to_object(
                        controller,
                        object_instance['objectId'],
                        position,
                        rotation
                    )
                    minimum_path_length = metrics.path_distance(path)
                    found_valid_point = path and minimum_path_length
                except:
                    print("Could not find path for:", position, "_".join([scene, o, str(ep)]))
                    found_valid_point = False
            episodes[scene].append({
                "id": "_".join([scene, o, str(ep)]),
                "initial_orientation": rotation["y"],
                "initial_position": position,
                "object_id": object_instance['objectId'],
                "object_type": o,
                "scene": scene,
                "shortest_path": path,
                "shortest_path_length": minimum_path_length,
                "target_position": object_instance["position"]
            })
            ep_lengths.append(minimum_path_length)

    print("Len all reachable points:", len(all_reachable_points))
    print("Len xs:", len(chosen_xs))
    plt.scatter(chosen_xs, chosen_ys)
    plt.show()
    exit()

ep_lengths.sort()
small_delimiter = ep_lengths[len(ep_lengths)//3]
med_delimiter = ep_lengths[2 * len(ep_lengths)//3]
print("Smalll Delimeter:", small_delimiter)
print("Medium Delimeter:", med_delimiter)
exit()

for scene, episodes in episodes.items():
    for ep in episodes:
        if ep["shortest_path_length"] <= small_delimiter:
            ep["difficulty"] = "easy"
        elif ep["shortest_path_length"] <= med_delimiter:
            ep["difficulty"] = "medium"
        else:
            ep["difficulty"] = "hard"

    json_str = json.dumps(episodes)
    json_bytes = json_str.encode('utf-8')
    with gzip.GzipFile('dataset/robothor/objectnav/train/content/' + scene + '.json.gz', 'w') as fout:
        fout.write(json_bytes)



