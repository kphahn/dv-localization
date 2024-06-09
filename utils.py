from pathlib import Path
import open3d as o3d
import numpy as np


def generate_colors(num_clusters):
    colors = []
    np.random.seed(42)

    for _ in range(num_clusters):
        color = np.random.random(size=3)
        colors.append(color)

    return colors


def plot_point_cloud_from_file(file: str, uniform_color: bool = True):
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"Error: {file} does not exist")

    pcd = o3d.io.read_point_cloud(file)
    if uniform_color:
        pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd])
