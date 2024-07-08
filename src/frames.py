import os
import open3d as o3d
import numpy as np
import csv
from tqdm import tqdm


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


def load_dataset(dataset_path):

    n = int(len(os.listdir(f"{dataset_path}/pointclouds")) / 2)
    print(f"Loading Dataset with {n} pointclouds")  # exclude frames with noise

    pcds = []

    for i in tqdm(range(n)):

        if os.path.exists(f"{dataset_path}/pointclouds/cloud_frame_{i}.ply"):
            try:
                frame = o3d.io.read_point_cloud(
                    f"{dataset_path}/pointclouds/cloud_frame_{i}.ply"
                )

                frame = correct_frame(
                    f"{dataset_path}/transformations/transformation_{i}.csv",
                    frame,
                )

                pcds.append(frame)

            except FileNotFoundError:
                print(f"FileNotFoundError with frame {i}")

        else:
            break

    return pcds


def get_cone_positions(frame):

    # remove ground plane
    _, inliers = frame.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    frame = frame.select_by_index(inliers, invert=True)

    # cluster remaining pointcloud and get cone centers
    labels = np.array(frame.cluster_dbscan(eps=0.1, min_points=4, print_progress=True))
    max_label = labels.max()

    clusters = [
        frame.select_by_index(np.where(labels == i)[0]) for i in range(max_label + 1)
    ]

    # derive cone positions from center
    positions = o3d.utility.Vector3dVector(
        [cluster.get_center() for cluster in clusters]
    )

    return positions, max_label + 1
