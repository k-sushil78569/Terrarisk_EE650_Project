import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from terrain_generator import TerrainMap

def generate_paper_figures():
    # 1. Load Monte Carlo metrics
    df = pd.read_csv("benchmark_results.csv")
    
    # 2. Plot Entrapment Rate Comparison Bar Chart
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=df, x="planner", y="entrapped", palette="Reds_r", ci=None)
    plt.title("Entrapment Rate Across Planners (N=50 Monte Carlo Trials)")
    plt.ylabel("Entrapment Rate (Lower is Better)")
    plt.xlabel("Planner Variant")
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("entrapment_rate_comparison.png", dpi=300)
    plt.close()

    # 3. Plot Terrain Ground-truth vs Visual Risk Map
    terrain = TerrainMap(seed=101)
    XX, YY, c_map, vis_map = terrain.generate_grid_maps()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].pcolormesh(XX, YY, c_map, cmap="viridis_r", shading="auto")
    axes[0].set_title("Ground-Truth Soil Cohesion c (Physical Hazard)")
    fig.colorbar(im0, ax=axes[0], label="Cohesion (kPa)")
    
    im1 = axes[1].pcolormesh(XX, YY, vis_map, cmap="magma", shading="auto")
    axes[1].set_title("Perceived Visual Risk Map (Stage 2 LiDAR/Camera)")
    fig.colorbar(im1, ax=axes[1], label="Perceived Risk Score")
    
    plt.tight_layout()
    plt.savefig("terrain_ambiguity_map.png", dpi=300)
    plt.close()
    print("Generated publication plots: 'entrapment_rate_comparison.png' and 'terrain_ambiguity_map.png'")

if __name__ == "__main__":
    generate_paper_figures()