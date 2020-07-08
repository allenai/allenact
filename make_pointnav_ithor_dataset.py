import ai2thor.controller
import ai2thor.util.metrics as metrics
import random
import gzip
import json
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import time


EPISODES_PER_SCENE = 1800
ROTATION_DEGREES = 45
GRID_SIZE = 0.25


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


# SCENES = ["FloorPlan_Val1_1", "FloorPlan_Val1_2", "FloorPlan_Val1_3", "FloorPlan_Val1_4", "FloorPlan_Val1_5",
#           "FloorPlan_Val2_1", "FloorPlan_Val2_2", "FloorPlan_Val2_3", "FloorPlan_Val2_4", "FloorPlan_Val2_5",
#           "FloorPlan_Val3_1", "FloorPlan_Val3_2", "FloorPlan_Val3_3", "FloorPlan_Val3_4", "FloorPlan_Val3_5"]


ROTATIONS = [{"x": 0.0, "y": float(y), "z": 0.0} for y in range(0, 360, ROTATION_DEGREES)]

def pos_to_str(pos):
    return "_".join([str(pos["x"]), str(pos["y"]), str(pos["z"])])

def show_graph(G, fig):
    plt.figure(fig)
    xs = [float(point.split("_")[0]) for point in G.nodes]
    ys = [float(point.split("_")[2]) for point in G.nodes]
    plt.scatter(xs, ys, s=1)
    plt.show()

coarse_controller = ai2thor.controller.Controller(agentMode='bot', gridSize=GRID_SIZE)
fine_controller = ai2thor.controller.Controller(agentMode='bot', gridSize=0.05)

def make_scene_cache(scene):
    coarse_controller.reset(scene=scene)
    fine_controller.reset(scene=scene)
    coarse_controller.step(action='GetReachablePositions')
    fine_controller.step(action='GetReachablePositions')
    G = nx.Graph()
    G_fine = nx.Graph()
    all_reachable_points = coarse_controller.last_event.metadata['actionReturn']
    all_reachable_points = [{"x": round(p["x"], 2), "y": p["y"], "z": round(p["z"], 2)} for p in all_reachable_points]
    fine_reachable_points = fine_controller.last_event.metadata['actionReturn']
    fine_reachable_points = [{"x": round(p["x"], 2), "y": p["y"], "z": round(p["z"], 2)} for p in fine_reachable_points]

    for point in all_reachable_points:
        G.add_node(pos_to_str(point))
        potential_neighbors = [
            {"x": point["x"] - GRID_SIZE, "y": point["y"], "z": point["z"]},
            {"x": point["x"] + GRID_SIZE, "y": point["y"], "z": point["z"]},
            {"x": point["x"], "y": point["y"], "z": point["z"] - GRID_SIZE},
            {"x": point["x"], "y": point["y"], "z": point["z"] + GRID_SIZE}
        ]
        for neighbor in potential_neighbors:
            if neighbor in all_reachable_points:
                G.add_edge(pos_to_str(point), pos_to_str(neighbor))

    for point in fine_reachable_points:
        potential_neighbors = [
            {"x": round(point["x"] - 0.05, 2), "y": point["y"], "z": point["z"]},
            {"x": round(point["x"] + 0.05, 2), "y": point["y"], "z": point["z"]},
            {"x": point["x"], "y": point["y"], "z": round(point["z"] - 0.05, 2)},
            {"x": point["x"], "y": point["y"], "z": round(point["z"] + 0.05, 2)}
        ]
        for neighbor in potential_neighbors:
            if neighbor in fine_reachable_points:
                G_fine.add_edge(pos_to_str(point), pos_to_str(neighbor))

    for point in all_reachable_points:
        for neighbor in all_reachable_points:
            p = pos_to_str(point)
            n = pos_to_str(neighbor)
            try:
                nx.shortest_path(G_fine, p, n)
            except:
                print("p", p, "n", n)

    try:
        return {
            pos_to_str(point): {
                pos_to_str(neighbor): {
                    "distance": 0.05 * len(nx.shortest_path(G_fine, pos_to_str(point), pos_to_str(neighbor))),
                    "path": nx.shortest_path(G, pos_to_str(point), pos_to_str(neighbor))
                } for neighbor in all_reachable_points
            } for point in all_reachable_points
        }
    except:
        print("BOOM!")


controller = ai2thor.controller.Controller(agentMode='bot', gridSize=GRID_SIZE)

data = {scene: {"episodes": [], "cache": {}} for scene in SCENES}
ep_lengths = []

for scene in tqdm(SCENES):
    controller.reset(scene=scene)
    event = controller.step(action='GetReachablePositions')
    all_reachable_points = event.metadata['actionReturn']
    agent = event.metadata['agent']
    scene_cache = make_scene_cache(scene)
    data[scene]["cache"] = scene_cache

    for ep in range(EPISODES_PER_SCENE):
        # Select a random point and starting orientation
        position = random.choice(all_reachable_points)
        rotation = random.choice(ROTATIONS)
        target = random.choice(all_reachable_points)

        data[scene]["episodes"].append({
            "id": "_".join([scene, str(ep)]),
            "initial_orientation": rotation["y"],
            "initial_position": position,
            "scene": scene,
            "shortest_path": scene_cache[pos_to_str(position)][pos_to_str(target)]["path"],
            "shortest_path_length": scene_cache[pos_to_str(position)][pos_to_str(target)]["distance"],
            "target_position": target
        })
        ep_lengths.append(scene_cache[pos_to_str(position)][pos_to_str(target)]["distance"])


ep_lengths.sort()
small_delimiter = ep_lengths[len(ep_lengths)//3]
med_delimiter = ep_lengths[2 * len(ep_lengths)//3]
print("Smalll Delimeter:", small_delimiter)
print("Medium Delimeter:", med_delimiter)

for scene, scene_data in data.items():
    for ep in scene_data["episodes"]:
        if ep["shortest_path_length"] <= small_delimiter:
            ep["difficulty"] = "easy"
        elif ep["shortest_path_length"] <= med_delimiter:
            ep["difficulty"] = "medium"
        else:
            ep["difficulty"] = "hard"

    json_str = json.dumps(scene_data)
    json_bytes = json_str.encode('utf-8')
    with gzip.GzipFile('dataset/robothor/pointnav/train/content/' + scene + '.json.gz', 'w') as fout:
        fout.write(json_bytes)
