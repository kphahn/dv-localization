import os
import csv
import json
import numpy as np
import open3d as o3d

from tqdm import tqdm


def _transform_from_csv(file_path):

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


def _labels_from_csv(file_path):

    cone_coordinates = [
        [],  # blue_cones
        [],  # yellow_cones
        [],  # big_cones
    ]

    with open(file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            coordinate = [row["x"], row["y"], 0]

            match row["label"]:
                case "blue_cone":
                    idx = 0
                case "yellow_cone":
                    idx = 1
                case "big_cone":
                    idx = 2

            cone_coordinates[idx].append(coordinate)

    local_coordinates = {
        "blue_cones": o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cone_coordinates[0])
        ).paint_uniform_color(np.asarray([0, 0, 1])),
        "yellow_cones": o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cone_coordinates[1])
        ).paint_uniform_color(np.asarray([1, 1, 0])),
        "big_cones": o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cone_coordinates[2])
        ).paint_uniform_color(np.asarray([1, 0.3, 0])),
    }

    return local_coordinates


def _labels_from_json(file_path):

    with open(file_path) as fp:
        print(f"reading from file {fp.name}")
        track_data = json.load(fp)

    cone_coordinates = [
        track_data["blue_cones"],
        track_data["yellow_cones"],
        track_data["big_cones"],
    ]

    for coordinates in cone_coordinates:
        for coord in coordinates:
            coord.append(0)

    global_coordinates = {
        "blue_cones": o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cone_coordinates[0])
        ).paint_uniform_color(np.asarray([0, 0, 1])),
        "yellow_cones": o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cone_coordinates[1])
        ).paint_uniform_color(np.asarray([1, 1, 0])),
        "big_cones": o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(cone_coordinates[2])
        ).paint_uniform_color(np.asarray([1, 0.3, 0])),
    }

    return global_coordinates


def _correct_frame(frame, transform):

    # transform frame coodinate system to LiDAR coordinate system
    T = np.eye(4)
    T[:3, :3] = o3d.geometry.PointCloud.get_rotation_matrix_from_axis_angle(
        [
            transform["rot_x"],
            transform["rot_y"],
            transform["rot_z"],
        ]
    )
    T[:3, 3] = [transform["x"], transform["y"], 0]
    T_inv = np.linalg.inv(T)

    return frame.transform(T_inv)


def _load_dataset(
    dataset_path,
    divider=1,
    noise=False,
    limit=None,
    perspective="local",
):

    dataset_name = dataset_path.rstrip("/").rsplit("/", 1)[1]

    limit = limit if limit else int(len(os.listdir(f"{dataset_path}/pointclouds")) / 2)

    print(f"Loading Dataset track_{dataset_name} with {limit} pointcloud(s). {noise=}")

    pcd_path = lambda i: f"{dataset_path}/pointclouds/cloud_frame_{i}.ply"
    tfm_path = lambda i: f"{dataset_path}/transformations/transformation_{i}.csv"
    lab_path = lambda i: f"{dataset_path}/labels/labels_{i}.csv"

    if noise:
        pcd_path = lambda i: f"{dataset_path}/pointclouds/cloud_frame_{i}_noise.ply"

    pointclouds = []
    labels = []

    for idx in tqdm(range(0, limit, divider)):

        try:
            pointcloud = o3d.io.read_point_cloud(pcd_path(idx))

            if perspective == "local":
                transform = _transform_from_csv(tfm_path(idx))
                pointcloud = _correct_frame(pointcloud, transform)

                labels.append(_labels_from_csv(lab_path(idx)))

            pointclouds.append(pointcloud)

        except FileNotFoundError as e:
            print(e)
            break

    return pointclouds, labels


def load_relative_frames(
    dataset_path,
    divider=1,
    noise=False,
    limit=None,
):
    pointclouds, labels = _load_dataset(
        dataset_path,
        divider=divider,
        noise=noise,
        limit=limit,
        perspective="local",
    )

    return pointclouds, labels


def load_ground_truth(dataset_path):

    pointclouds, labels = _load_dataset(
        dataset_path=dataset_path,
        divider=1,
        noise=False,
        limit=None,
        perspective="global",
    )

    dataset_name = dataset_path.rstrip("/").rsplit("/", 1)[1]

    ground_truth_pcd = o3d.geometry.PointCloud()
    for pointcloud in pointclouds:
        _, inliers = pointcloud.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=100
        )

        ground_truth_pcd += pointcloud.select_by_index(inliers, invert=True)

    labels = _labels_from_json(f"{dataset_path}/{dataset_name}.json")

    ground_truth_coords = (
        labels["blue_cones"].paint_uniform_color(np.asarray([0, 0, 1]))
        + labels["yellow_cones"].paint_uniform_color(np.asarray([1, 1, 0]))
        + labels["big_cones"].paint_uniform_color(np.asarray([1, 0.65, 0]))
    )

    rotation_matrix = np.array(
        [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    ground_truth_pcd.transform(rotation_matrix)
    ground_truth_coords.transform(rotation_matrix)

    return ground_truth_pcd, ground_truth_coords
