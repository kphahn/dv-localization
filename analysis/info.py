import open3d as o3d
import numpy as np
from argparse import ArgumentParser

import fake_scenes
import utils
import slam
import pcd

import time
import copy

import statistics

# distances = []

# print(statistics.mean(distances))
# print(sum(distances))
# # DEBUG #
# translation = odometry[:3, 3]
# distance = np.linalg.norm(translation)
# distances.append(distance)
# # ##########


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

    lidar_dataset = fake_scenes.load_relative_frames(
        args.dataset_path, divider=1, noise=args.noise
    )

    lidar_dataset.append(lidar_dataset[0])

    previous_frame = lidar_dataset[0]
    odometry = np.eye(4)

    rmse_list = []

    for frame in lidar_dataset:

        frame_processed = pcd.preprocess_frame(frame)
        frame_without_ground = pcd.remove_ground(frame, 0)
        clusters = pcd.extract_clusters(frame_without_ground, 10)
        cone_coordinates = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pcd.detect_cones_by_size(clusters))
        )

        # ESTIMATE MOTION
        odometry, rmse = slam.estimate_odometry(
            cone_coordinates,
            previous_frame,
            odometry,
        )

        rmse_list.append(rmse)
        previous_frame = cone_coordinates

    print(statistics.mean(rmse_list))

    # frame = lidar_dataset[0]

    # frame_cropped = pcd.preprocess_frame(frame)
    # frame_without_ground = pcd.remove_ground(frame_cropped, 0)

    # clusters = pcd.extract_clusters(frame_without_ground, 10)
    # cone_coordinates = o3d.geometry.PointCloud(
    #     o3d.utility.Vector3dVector(pcd.detect_cones_by_size(clusters))
    # )
    # # centers = o3d.utility.Vector3dVector([cluster.get_center() for cluster in clusters])

    # utils.draw([frame, frame_cropped, frame_without_ground, cone_coordinates])

    # colors = utils.generate_colors(len(clusters))

    # for cluster, color in zip(clusters, colors):
    #     # cluster.paint_uniform_color(color)
    #     bb = cluster.get_axis_aligned_bounding_box()
    #     bb.color = np.asarray([1, 0, 0])

    #     cluster_center = cluster.get_center()
    #     cluster_center[2] = 0.1
    #     cluster_center = o3d.geometry.PointCloud(
    #         o3d.utility.Vector3dVector([cluster_center])
    #     )
    #     cluster_highest = o3d.geometry.PointCloud(
    #         o3d.utility.Vector3dVector([pcd.get_xy_of_highest_point(cluster)])
    #     )

    #     distance = np.linalg.norm(cluster_center.points[0] - cluster_highest.points[0])
    #     print(distance)
    #     utils.draw(
    #         [
    #             cluster.paint_uniform_color(np.asarray([0.3, 0.3, 0.3])),
    #             cluster_center.paint_uniform_color(np.asarray([1, 0.3, 0])),
    #             cluster_highest.paint_uniform_color(np.asarray([0, 0.3, 1])),
    #             bb,
    #         ]
    #     )

    # clusters.append(frame_without_ground)

    # utils.draw(clusters)

    # with open(DATASET_PATH + "/" + dataset_name + ".txt", "w") as f:
    #     print("mean=", statistics.mean(distances), file=f)
    #     print("sum=", sum(distances), file=f)
    #     print("rmse=", rmse, file=f)

    #     print("", file=f)
    #     for i, d in enumerate(distances):
    #         print(i, d, file=f)
