import open3d as o3d
import numpy as np

import utils
import frames

DATASET_PATH = "Datasets/track_s42"
TRACK = "s42"


ground_truth = utils.load_ground_truth_from_json(f"{DATASET_PATH}/track_{TRACK}.json")
lidar_frames = frames.load_dataset(DATASET_PATH)

initial_transform = np.eye(4)
initial_transform[:3, :3] = [
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1],
]  # 90 degrees around z-axis, clockwise

path = o3d.geometry.PointCloud()


for frame in lidar_frames:

    # local registration
    cone_positions, n_cones = frames.get_cone_positions(frame)
    perceived_map = o3d.geometry.PointCloud(cone_positions)

    icp_result = o3d.pipelines.registration.registration_icp(
        perceived_map,
        ground_truth,
        1.50,
        initial_transform,
    )
    matched_map = perceived_map.transform(icp_result.transformation)

    # global registration
    origin_point = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
    origin = o3d.geometry.PointCloud(origin_point)
    origin.transform(icp_result.transformation)
    path = path + origin

    initial_transform = icp_result.transformation

ground_truth.paint_uniform_color([0.5, 0.5, 0.5])
path.paint_uniform_color([1.0, 0.65, 0.0])
utils.draw([ground_truth, path])

o3d.io.write_point_cloud(
    f"{DATASET_PATH}/path_{TRACK}.ply",
    path,
    print_progress=True,
)

print(f"Path saved at {DATASET_PATH}/path_{TRACK}.ply")
print("Program terminated")
