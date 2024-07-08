import open3d as o3d
import numpy as np
import json


def generate_colors(num_clusters):
    colors = []
    np.random.seed(42)

    for _ in range(num_clusters):
        color = np.random.random(size=3)
        colors.append(color)

    return colors


def load_ground_truth_from_json(file_path):
    with open(file_path) as fp:
        track_data = json.load(fp)

    cone_positions = (
        track_data["blue_cones"] + track_data["yellow_cones"] + track_data["big_cones"]
    )
    for cp in cone_positions:
        cp.append(0)

    cone_positions = o3d.utility.Vector3dVector(cone_positions)

    return o3d.geometry.PointCloud(cone_positions)


def draw(pointclouds):
    o3d.visualization.draw(
        pointclouds,
        show_skybox=False,
        point_size=5,
    )
