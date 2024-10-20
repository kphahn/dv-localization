import open3d as o3d
import numpy as np
from argparse import ArgumentParser

import fake_scenes
import pcd
import utils


if __name__ == "__main__":

    # Initialization
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--noise",
        default=False,
        help="Use frames with noise",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to track dataset",
    )
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path
    dataset_name = DATASET_PATH.rstrip("/").rsplit("/", 1)[1]

    # load dataset
    lidar_dataset = fake_scenes.load_relative_frames(
        DATASET_PATH, divider=1, noise=args.noise, end=1
    )

    frame = lidar_dataset[0]

    frame_processed = pcd.preprocess_frame(frame)
    frame_without_ground = pcd.remove_ground(frame, 0)
    clusters = pcd.extract_clusters(frame_without_ground, 10)
    cone_coordinates = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pcd.detect_cones_by_size(clusters))
    )

    cluster = o3d.geometry.PointCloud()
    for c in clusters:
        cluster += c

    utils.draw(
        [
            frame.paint_uniform_color(utils.black),
            frame_processed.paint_uniform_color(utils.black),
            frame_without_ground.paint_uniform_color(utils.black),
            cluster.paint_uniform_color(utils.blue),
            cone_coordinates.paint_uniform_color(utils.red),
        ]
    )


print("Program finished")
