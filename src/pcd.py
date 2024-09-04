import numpy as np
import open3d as o3d
import copy


def preprocess_frame(frame):

    min_bound = np.array([-15, 0, -0.5])
    max_bound = np.array([15, 40, 1])
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    return frame.crop(bounding_box)


def remove_ground(frame, seed):

    # make function deterministic
    o3d.utility.random.seed(seed)

    _, inliers = frame.segment_plane(
        distance_threshold=0.05, ransac_n=5, num_iterations=1000
    )

    return frame.select_by_index(inliers, invert=True)


def extract_clusters(frame, min_detection_points):

    # cluster remaining pointcloud and get cone centers
    labels = np.array(
        frame.cluster_dbscan(
            eps=0.285 / 2,  # half a cone
            min_points=min_detection_points,
        )
    )
    max_label = labels.max()

    clusters = [
        frame.select_by_index(np.where(labels == i)[0]) for i in range(max_label + 1)
    ]

    return clusters


def get_xy_of_highest_point(cluster):

    # detect highest point and set z-component to 0
    points = np.asarray(cluster.points)
    max_z_index = np.argmax(points[:, 2])
    coordinate = copy.deepcopy(points[max_z_index])
    coordinate[2] = 0

    return coordinate


def filter_by_size(cluster):

    bb = cluster.get_axis_aligned_bounding_box()
    height = bb.get_extent()[2]

    if height > 0.1:
        return True

    return False


def reconstruct_cone(
    center,
    cone_width,
    original_pointcloud,
):
    reconstructed_cone = o3d.geometry.PointCloud()
    radius = cone_width / 2
    for point in original_pointcloud.points:
        if np.linalg.norm(point[:2] - center[:2]) <= radius:
            reconstructed_cone.points.append(point)

    return reconstructed_cone


def calculate_expected_points(
    d, cone_height, cone_width, resolution_ver, resolution_hor
):

    E_d = (
        0.5
        * (cone_height / (2 * d * np.tan(resolution_ver / 2)))
        * (cone_width / (2 * d * np.tan(resolution_hor / 2)))
    )

    return E_d


def filter_by_num_points(
    center,
    original_pointcloud,
    tolerance=0.2,
):

    # set cone specification
    cone_height = 0.325
    cone_width = 0.285
    resolution_ver = np.radians(0.3)
    resolution_hor = np.radians(0.2)

    # reconstruct the cylindrical area around the cluster center
    reconstructed_cluster = reconstruct_cone(center, cone_width, original_pointcloud)

    # compare expected and actual points
    distance = np.linalg.norm(center)
    expected_points = calculate_expected_points(
        distance, cone_height, cone_width, resolution_ver, resolution_hor
    )
    actual_points = len(reconstructed_cluster.points)

    # # Diagnostic information
    # print(f"Distance: {distance:.2f}m")
    # print(f"Expected points: {expected_points:.2f}")
    # print(f"Actual points: {actual_points}")
    # print(f"Actual/Expected ratio: {actual_points / expected_points:.2f}")
    # print("---")

    if abs(actual_points - expected_points) / expected_points <= tolerance:
        return True

    return False


def detect_cones_by_size(clusters):
    cones = [cluster for cluster in clusters if filter_by_size(cluster)]
    cone_coordinates = [get_xy_of_highest_point(cone) for cone in cones]

    return cone_coordinates


def detect_cones_by_num_points(clusters, frame):
    cluster_coordinates = [get_xy_of_highest_point(cluster) for cluster in clusters]
    cone_coordinates = [
        coordinate
        for coordinate in cluster_coordinates
        if filter_by_num_points(coordinate, frame)
    ]

    return cone_coordinates
