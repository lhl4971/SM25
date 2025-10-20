import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_solution(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    data = pd.read_csv(filename, header=None).values
    M, N = data.shape
    print(f"Data shape: {data.shape}")

    x = np.linspace(0.0, 3.0, M)
    y = np.linspace(0.0, 3.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 2D
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        data.T,
        origin="lower",
        cmap="viridis",
        interpolation="bilinear",
        extent=[0.0, 3.0, 0.0, 3.0]
    )
    plt.colorbar(im, label="u(x,y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    output_filename = os.path.splitext(filename)[0] + '_2D.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"2D heatmap saved as: {output_filename}")
    plt.close()

    # 3D
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data, cmap='viridis', linewidth=0, antialiased=True)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x,y)")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="u(x,y)")
    plt.tight_layout()

    output_filename_3d = os.path.splitext(filename)[0] + '_3D.png'
    plt.savefig(output_filename_3d, dpi=300, bbox_inches='tight')
    print(f"3D surface saved as: {output_filename_3d}")
    plt.close()

plot_solution("solution/solution_M_800_N_1200.csv")