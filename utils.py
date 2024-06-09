from argparse import ArgumentParser
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import h5py
import open3d as o3d

pio.templates.default = "plotly_dark"

def plot_point_cloud(file: Path, noise: bool = False):
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"Error: {file} does not exist")
    
    suffix = file.suffix

    if suffix == ".hdf5":
        with h5py.File(file) as data:
            x = data[f"location{'_noise' if noise else ''}_x"][()][0].astype(np.float32)
            y = data[f"location{'_noise' if noise else ''}_y"][()][0].astype(np.float32)
            z = data[f"location{'_noise' if noise else ''}_z"][()][0].astype(np.float32)
            intensity = data["intensity"][()][0].astype(np.float32)

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=3, color=intensity, colorscale="Viridis", opacity=0.8
                    ),
                )
            ]
        )

        fig.update_layout(scene=dict(aspectmode="data"))
        fig.show()
    
    if suffix == ".ply":
        file = str(file)
        if noise:
            file = file.rpartition('.')[0] + "_noise" + suffix
        pcd = o3d.io.read_point_cloud(file)

        pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([pcd])

def main():
    parser = ArgumentParser()
    parser.add_argument("file", type=Path, help="Path to the point cloud file")
    parser.add_argument("--noise", action="store_true", help="Plot the noisy point cloud")
    args = parser.parse_args()
    plot_point_cloud(args.file, noise=args.noise)

if __name__ == "__main__":
    main()
