import os
import open3d as o3d
import numpy as np
import csv
import copy

from enum import Enum
from tqdm import tqdm


### GENERIC ###


def generate_colors(n):
    colors = []
    np.random.seed(1)

    for _ in range(n):
        color = np.random.random(size=3)
        colors.append(color)

    return colors


def draw(pointclouds):
    o3d.visualization.draw(
        pointclouds,
        show_skybox=False,
        point_size=5,
    )


### FRAMES ###

MIN_POINTS_FOR_DETECTION = 10
FRAME_DIVIDER = 2

NODE_TYPE = Enum("Node_Type", ["POSE", "LANDMARK"])
EDGE_TYPE = Enum("Edge_Type", ["ODOMETRY", "OBSERVATION", "LOOP_CLOSURE"])


def read_transform_from_csv(file_path):

    with open(file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            transform = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "rot_x": float(row["rot_x"]),
                "rot_y": float(row["rot_y"]),
                "rot_z": float(row["rot_z"]),
            }

    return transform


def correct_frame(path_to_transform, frame):

    # transform frame coodinate system to LiDAR coordinate system
    transform = read_transform_from_csv(path_to_transform)
    T = np.eye(4)
    T[:3, :3] = o3d.geometry.PointCloud.get_rotation_matrix_from_axis_angle(
        [
            transform["rot_x"],
            transform["rot_y"],
            transform["rot_z"],
        ]
    )
    T[:3, 3] = [transform["x"], transform["y"], transform["z"]]
    T_inv = np.linalg.inv(T)

    return frame.transform(T_inv)


def load_dataset(dataset_path, noise=False, limit=None):

    limit = limit if limit else int(len(os.listdir(f"{dataset_path}/pointclouds")) / 2)
    print(
        f"Loading Dataset track_{dataset_path.rsplit('_', 1)[1]} with {limit} pointcloud(s). {noise=}"
    )

    pcd_path = lambda i: f"{dataset_path}/pointclouds/cloud_frame_{i}.ply"
    tfm_path = lambda i: f"{dataset_path}/transformations/transformation_{i}.csv"

    if noise:
        pcd_path = lambda i: f"{dataset_path}/pointclouds/cloud_frame_{i}_noise.ply"

    pcds = []

    for i in tqdm(range(0, limit, FRAME_DIVIDER)):

        try:
            frame = o3d.io.read_point_cloud(pcd_path(i))
            frame = correct_frame(tfm_path(i), frame)

            pcds.append(frame)

        except FileNotFoundError as e:
            print(e)
            break

    return pcds


def get_cone_positions(frame):

    # remove ground plane
    _, inliers = frame.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=100
    )
    frame = frame.select_by_index(inliers, invert=True)

    # cluster remaining pointcloud and get cone centers
    labels = np.array(
        frame.cluster_dbscan(
            eps=0.1, min_points=MIN_POINTS_FOR_DETECTION, print_progress=True
        )
    )
    max_label = labels.max()

    clusters = [
        frame.select_by_index(np.where(labels == i)[0]) for i in range(max_label + 1)
    ]

    # derive cone positions from center
    positions = o3d.utility.Vector3dVector(
        [cluster.get_center() for cluster in clusters]
    )
    print()

    # remove z-component
    for position in positions:
        position[2] = 0

    return positions


def preprocess_frame(frame):
    return o3d.geometry.PointCloud(get_cone_positions(frame))


def generate_static_visualization(
    frames,
    pose_graph,
    node_types,
    visualize_poses=False,
    visualize_edges=True,
    show_frames=[],
):
    frames = copy.deepcopy(frames)
    poses = o3d.geometry.TriangleMesh.create_coordinate_frame()
    edges = o3d.geometry.LineSet()
    path = o3d.geometry.PointCloud()

    i = 0
    for idx, node in enumerate(pose_graph.nodes):
        if node_types[idx] == NODE_TYPE.POSE:
            frames[i].transform(node.pose)

            if visualize_poses:
                pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
                pose.transform(node.pose)
                poses = poses + pose

            if visualize_edges:
                point = np.asarray(node.pose[:3, 3])
                edges.points.append(point)

                path.points.append(point)
                path.colors.append([1, 0.65, 0])

                if i > 0:
                    edges.lines.append([i - 1, i])
                    edges.colors.append([0, 0, 0])

                if i == len(frames) - 1:
                    edges.lines.append([i, 0])
                    edges.colors.append([1, 0, 0])

            i += 1

    pointcloud = []

    if len(show_frames):
        for n in show_frames:
            pointcloud.append(frames[n])
    else:
        pointcloud = frames

    if visualize_poses:
        pointcloud.append(poses)

    if visualize_edges:
        pointcloud.append(edges)
        pointcloud.append(path)

    return pointcloud
