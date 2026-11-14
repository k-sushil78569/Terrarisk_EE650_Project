#!/usr/bin/env python3
"""
TerraRisk: Publication Figures and Results Analysis Generator
============================================================
Parses benchmark_results.csv from the 50-seed Monte Carlo run and generates:
1. High-DPI publication plots (Entrapment rate, Efficiency, Sinkage, Risk)
2. Formatted Markdown and LaTeX metrics summary table
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from terrarisk.terrain import TerrainMap
from terrarisk.appearance_sensor import AppearanceSensor

# Set publication style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
})


def analyze_and_plot():
    # 1. Load Monte Carlo metrics
    df = pd.read_csv("benchmark_results.csv")
    num_seeds = df['seed'].nunique()
    print(f"Loaded benchmark_results.csv with {len(df)} total runs across {num_seeds} seeds.")

    # 2. Compute Aggregated Metrics
    summary = df.groupby("planner").agg(
        Success_Rate=("success", "mean"),
        Entrapment_Rate=("entrapped", "mean"),
        Avg_Efficiency=("efficiency", "mean"),
        Std_Efficiency=("efficiency", "std"),
        Avg_Max_Sinkage_cm=("max_sinkage", lambda x: x.mean() * 100),
        Std_Max_Sinkage_cm=("max_sinkage", lambda x: x.std() * 100),
        Avg_Accum_Risk=("accum_risk", "mean")
    ).reset_index()

    # Sort planners in stage order
    planner_order = [
        "Deterministic_NMPC",
        "Gaussian_SMPC",
        "Stage2_Visual_CVaR",
        "Stage3_Physical_CVaR_SMPC"
    ]
    summary['planner'] = pd.Categorical(summary['planner'], categories=planner_order, ordered=True)
    summary = summary.sort_values('planner')

    print("\n" + "="*80)
    print("                     TERRARISK BENCHMARK METRICS SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80 + "\n")

    # 3. Create Multi-Panel Publication Figure (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300)
    colors = ["#d95f02", "#7570b3", "#e7298a", "#1b9e77"]

    # (A) Entrapment Rate Bar Plot
    sns.barplot(
        data=df, x="planner", y="entrapped", order=planner_order,
        palette=colors, ax=axes[0, 0], errorbar=None
    )
    axes[0, 0].set_title("(a) Entrapment Rate (Lower is Better)")
    axes[0, 0].set_ylabel("Entrapment Rate")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_xticklabels(["Deterministic\nNMPC", "Gaussian\nSMPC", "Stage 2\nVisual CVaR", "Stage 3\nPhysical CVaR"])
    axes[0, 0].set_ylim(0, max(0.4, df.groupby("planner")["entrapped"].mean().max() * 1.25))
    for p in axes[0, 0].patches:
        height = p.get_height()
        if height > 0:
            axes[0, 0].annotate(f'{height*100:.1f}%',
                                (p.get_x() + p.get_width() / 2., height / 2),
                                ha='center', va='center', color='white', fontweight='bold')

    # (B) Path Efficiency Box/Violin Plot
    sns.boxplot(
        data=df[df['success'] == True], x="planner", y="efficiency", order=planner_order,
        palette=colors, ax=axes[0, 1], width=0.5
    )
    axes[0, 1].set_title("(b) Path Efficiency η = d_direct / L_path (Successful Runs)")
    axes[0, 1].set_ylabel("Efficiency Ratio")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_xticklabels(["Deterministic\nNMPC", "Gaussian\nSMPC", "Stage 2\nVisual CVaR", "Stage 3\nPhysical CVaR"])

    # (C) Maximum Sinkage Distribution (Boxplot + z_crit threshold)
    sns.boxplot(
        data=df, x="planner", y=df["max_sinkage"] * 100, order=planner_order,
        palette=colors, ax=axes[1, 0], width=0.5
    )
    axes[1, 0].axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, label='Entrapment Threshold z_crit (10cm)')
    axes[1, 0].set_title("(c) Maximum Sinkage Distribution")
    axes[1, 0].set_ylabel("Max Sinkage (cm)")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_xticklabels(["Deterministic\nNMPC", "Gaussian\nSMPC", "Stage 2\nVisual CVaR", "Stage 3\nPhysical CVaR"])
    axes[1, 0].legend(loc='upper left')

    # (D) Accumulated Tail Risk
    sns.barplot(
        data=df, x="planner", y="accum_risk", order=planner_order,
        palette=colors, ax=axes[1, 1], errorbar=None
    )
    axes[1, 1].set_title("(d) Accumulated Tail Risk Over Horizon")
    axes[1, 1].set_ylabel("Accumulated Tail Hazard Cost")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_xticklabels(["Deterministic\nNMPC", "Gaussian\nSMPC", "Stage 2\nVisual CVaR", "Stage 3\nPhysical CVaR"])

    plt.tight_layout()
    plt.savefig("stage3_final_benchmark.png", dpi=300)
    plt.close()
    print("Saved multi-panel publication plot: 'stage3_final_benchmark.png'")

    # 4. Generate Terrain Ambiguity Map (Physical vs Perceived)
    sensor = AppearanceSensor(rng_seed=101)
    xs = np.linspace(0, 50, 100)
    ys = np.linspace(0, 50, 100)
    XX, YY = np.meshgrid(xs, ys)
    c_map = np.zeros_like(XX)
    vis_map = np.zeros_like(XX)

    terrain = TerrainMap(seed=101)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            px, py = XX[i, j], YY[i, j]
            soil = terrain.get_soil_parameters(px, py, stochastic=False)
            c_map[i, j] = soil.c
            _, prior_soil, unc = sensor.classify_terrain(px, py, soil)
            vis_map[i, j] = unc

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    im0 = axes[0].pcolormesh(XX, YY, c_map, cmap="viridis_r", shading="auto")
    axes[0].set_title("Ground-Truth Soil Cohesion c (kPa) [Proprioceptive Hazard]")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    fig.colorbar(im0, ax=axes[0], label="Cohesion c (kPa)")

    im1 = axes[1].pcolormesh(XX, YY, vis_map, cmap="magma", shading="auto")
    axes[1].set_title("Perceived Visual Risk Map (Stage 2 LiDAR/Camera)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    fig.colorbar(im1, ax=axes[1], label="Perceived Risk Score")

    plt.tight_layout()
    plt.savefig("terrain_ambiguity_map.png", dpi=300)
    plt.close()
    print("Saved terrain ambiguity comparison map: 'terrain_ambiguity_map.png'")


if __name__ == "__main__":
    analyze_and_plot()
