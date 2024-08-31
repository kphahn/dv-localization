import open3d as o3d
import numpy as np
import copy
import os
from sensor_msgs.msg import PointCloud2, PointField, ChannelFloat32


def convert_matrix_to_ros(name, data):

    channel_msg = ChannelFloat32()
    channel_msg.name = name
    channel_msg.values = data.flatten().tolist()

    return channel_msg


def convert_ros_to_matrix(msg, shape):

    flat_array = np.array(msg.values, dtype=np.float32)
    array = flat_array.reshape(shape)

    return array


def convert_o3d_to_ros(self, frame):

    points = np.asarray(frame.points)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    time = self.get_clock().now().to_msg()
    pointcloud_msg = PointCloud2()
    pointcloud_msg.header.stamp = time
    pointcloud_msg.header.frame_id = str(
        self.frame_count
    )  # field misused to identify number of frame in dataset
    pointcloud_msg.height = 1
    pointcloud_msg.width = points.shape[0]
    pointcloud_msg.fields = fields
    pointcloud_msg.is_bigendian = False
    pointcloud_msg.point_step = 12  # 3 * 4 bytes (float32)
    pointcloud_msg.row_step = pointcloud_msg.point_step * points.shape[0]
    pointcloud_msg.is_dense = True
    pointcloud_msg.data = np.asarray(points, np.float32).tobytes()

    return pointcloud_msg


def convert_ros_to_o3d(self, pointcloud_msg):

    # Extract fields from the PointCloud2 message
    dtype_list = []
    for field in pointcloud_msg.fields:
        if field.datatype == PointField.FLOAT32:
            dtype_list.append((field.name, np.float32))
        else:
            raise NotImplementedError(
                f"Field {field.name} with datatype {field.datatype} not implemented"
            )

    # Convert the raw data to a structured NumPy array
    pointcloud_array = np.frombuffer(pointcloud_msg.data, dtype=np.dtype(dtype_list))
    points = np.vstack(
        [
            pointcloud_array["x"],
            pointcloud_array["y"],
            pointcloud_array["z"],
        ]
    ).T

    # Create Open3D point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    return cloud


def load_dataset(dataset_path):

    directory = os.listdir(dataset_path)
    dataset = []

    for file in directory:
        dataset.append(o3d.io.read_point_cloud(os.path.join(dataset_path, file)))

    return dataset


def generate_colors(n, seed=None):
    colors = []
    if seed is not None:
        np.random.seed(seed)

    for _ in range(n):
        # Generate a dark color by limiting the random values to a range (e.g., 0 to 0.5)
        color = np.random.random(size=3) * 0.5
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
