import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_solution(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    data = pd.read_csv(filename, header=None).values
    print(f"✅ Data shape: {data.shape}")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        data.T,
        origin="lower",
        cmap="viridis",
        interpolation="bilinear",
        extent=[0.0, 3.0, 0.0, 3.0]
    )
    plt.colorbar(im, label="u(x,y)")
    plt.title(f"Solution heatmap: {filename}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    
    output_filename = os.path.splitext(filename)[0] + '.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")

    plt.close()

plot_solution("solution/solution_M_800_N_1200.csv")