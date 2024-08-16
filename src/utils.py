import open3d as o3d
import numpy as np
import copy
import os


def load_dataset(dataset_path):

    directory = os.listdir(dataset_path)
    dataset = []

    for file in directory:
        dataset.append(o3d.io.read_point_cloud(os.path.join(dataset_path, file)))

    return dataset


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


def visualize_pose_graph(
    frames,
    pose_graph,
    visualize_poses=False,
    visualize_edges=True,
    show_frames=[],
):

    frames = copy.deepcopy(frames)
    poses = o3d.geometry.TriangleMesh.create_coordinate_frame()
    edges = o3d.geometry.LineSet()
    path = o3d.geometry.PointCloud()

    for idx, frame in enumerate(frames):

        idx_current_pose = pose_graph.idxs_poses[idx]
        pose_transform = pose_graph.nodes[idx_current_pose].pose

        frame.transform(pose_transform)

        if visualize_poses:
            pose = o3d.geometry.TriangleMesh.create_coordinate_frame()
            pose.transform(pose_transform)
            poses = poses + pose

        if visualize_edges:
            point = np.asarray(pose_transform[:3, 3])
            edges.points.append(point)

            path.points.append(point)
            path.colors.append([1, 0.65, 0])

            if idx > 0:
                edges.lines.append([idx - 1, idx])
                edges.colors.append([0, 0, 0])

            if idx == len(frames) - 1:
                edges.lines.append([idx, 0])
                edges.colors.append([1, 0, 0])

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
