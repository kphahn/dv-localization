import open3d as o3d
import numpy as np

import utils

pcd = o3d.io.read_point_cloud("Datasets/track_0/pointclouds/cloud_frame_0.ply")

# remove ground from pointcloud
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01, ransac_n=3, num_iterations=1000
)
above_ground_cloud = pcd.select_by_index(inliers, invert=True)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# cluster the remaining pointcloud
labels = np.array(
    above_ground_cloud.cluster_dbscan(eps=0.1, min_points=4, print_progress=True)
)
max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")
clusters = [
    above_ground_cloud.select_by_index(np.where(labels == i)[0])
    for i in range(max_label + 1)
]

# color each cluster differently
num_clusters = len(clusters)
colors = utils.generate_colors(num_clusters)
for i, cluster in enumerate(clusters):
    cluster.paint_uniform_color(colors[i])

# visualize the result
o3d.visualization.draw_geometries(clusters)

### TO DO:
# - Classify cluster as a cone (Cone Detection p.10 AMZ Driverless)
# - Find center of cone
# - Get Position of cone as coordinates
