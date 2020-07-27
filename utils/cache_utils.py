from typing import Dict, Any
import math


def _pos_to_str(pos: Dict[str, float]) -> str:
    return "_".join([str(pos["x"]), str(pos["y"]), str(pos["z"])])

def _str_to_pos(s: str) -> Dict[str, float]:
    split = s.split("_")
    return {"x": float(split[0]), "y": float(split[1]), "z": float(split[2])}


def get_distance(cache: Dict[str, Any], pos: Dict[str, float], target: Dict[str, float]) -> float:
    pos = {
        "x": 0.25 * math.ceil(pos["x"] / 0.25),
        "y": pos["y"],
        "z": 0.25 * math.ceil(pos["z"] / 0.25)
    }
    sp = _get_shortest_path_distance_from_cache(cache, pos, target)
    if sp == -1.0:
        pos = {
            "x": 0.25 * math.floor(pos["x"] / 0.25),
            "y": pos["y"],
            "z": 0.25 * math.ceil(pos["z"] / 0.25)
        }
        sp = _get_shortest_path_distance_from_cache(cache, pos, target)
    if sp == -1.0:
        pos = {
            "x": 0.25 * math.ceil(pos["x"] / 0.25),
            "y": pos["y"],
            "z": 0.25 * math.floor(pos["z"] / 0.25)
        }
        sp = _get_shortest_path_distance_from_cache(cache, pos, target)
    if sp == -1.0:
        pos = {
            "x": 0.25 * math.floor(pos["x"] / 0.25),
            "y": pos["y"],
            "z": 0.25 * math.floor(pos["z"] / 0.25)
        }
        sp = _get_shortest_path_distance_from_cache(cache, pos, target)
    if sp == -1.0:
        pos = find_nearest_point_in_cache(cache, pos)
        sp = _get_shortest_path_distance_from_cache(cache, pos, target)
    if sp == -1.0:
        target = find_nearest_point_in_cache(cache, target)
        sp = _get_shortest_path_distance_from_cache(cache, pos, target)
    if sp == -1.0:
        print("Your cache is incomplete!")
        exit()
    return sp


def _get_shortest_path_distance_from_cache(cache: Dict[str, Any], position: Dict[str, float], target: Dict[str, float]) -> float:
    try:
        return cache[_pos_to_str(position)][_pos_to_str(target)]["distance"]
    except:
        return -1.0


def find_nearest_point_in_cache(cache: Dict[str, Any], point: Dict[str, float]) -> Dict[str, float]:
    best_delta = float('inf')
    for p in cache:
        p = _str_to_pos(p)
        delta = abs(point["x"] - p["x"]) + abs(point["y"] - p["y"]) + abs(point["z"] - p["z"])
        if delta < best_delta:
            best_delta = delta
            closest_point = p
    return closest_point
