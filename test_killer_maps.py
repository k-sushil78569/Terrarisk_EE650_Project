#!/usr/bin/env python3
"""
TerraRisk: Killer Map Stress Test
==================================
Runs all 4 controllers on 3 maps specifically engineered to expose the
fundamental weaknesses of Deterministic NMPC and Gaussian SMPC.

Analytically guaranteed (see killer_maps.py docstrings):
  - NMPC and Gaussian SMPC: ~93% entrapment per 5 steps in patch center
  - Stage 3 Physical CVaR: EKF triggers at patch EDGE, reroutes to bypass

Outputs:
  - killer_maps_results.csv        (quantitative metrics)
  - killer_maps_benchmark.png      (3-panel visualization, one per map)
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from terrarisk.killer_maps import (
    create_death_funnel_map,
    create_diagonal_minefield_map,
    create_false_safe_shortcut_map,
)
from terrarisk.dynamics import RoverModel
from terrarisk.appearance_sensor import AppearanceSensor
from terrarisk.deterministic_nmpc import DeterministicNMPCPlanner
from terrarisk.appearance_mpc import AppearanceMPCPlanner
from terrarisk.cvar_smpc import CVaRSMPCPlanner
from terrarisk.stage3_cvar_planner import Stage3PhysicallyGroundedCVaRPlanner
from terrarisk.simulator import Simulator


# ── Killer map scenario definitions ───────────────────────────────────────────
MAP_CONFIGS = [
    {
        "name": "Map 1: Death Funnel",
        "fn": create_death_funnel_map,
        "start": np.array([10.0, 50.0, 0.0, 0.5]),
        "goal":  np.array([90.0, 50.0]),
        "why_trapped": "NMPC/Gaussian: straight corridor → central z_nom=7.55cm → 93% stochastic entrapment\nStage 3: EKF triggers at patch edge → safe strip navigation",
    },
    {
        "name": "Map 2: Diagonal Minefield",
        "fn": create_diagonal_minefield_map,
        "start": np.array([10.0, 10.0, np.pi/4, 0.5]),
        "goal":  np.array([90.0, 90.0]),
        "why_trapped": "NMPC/Gaussian: shortest diagonal → 3 consecutive deceptive mines → near-certain entrapment\nStage 3: anomaly memory from Mine A diverts to safe arc around all three",
    },
    {
        "name": "Map 3: False-Safe Shortcut",
        "fn": create_false_safe_shortcut_map,
        "start": np.array([10.0, 50.0, 0.0, 0.5]),
        "goal":  np.array([90.0, 50.0]),
        "why_trapped": "NMPC/Gaussian: cost gradient always prefers short direct route (10*z_nom=0.75 < detour cost)\nStage 3: first contact anomaly flips decision to safe arc — 40% longer but alive",
    },
]

PLANNER_STYLES = {
    "Deterministic_NMPC":         {"color": "#e41a1c", "ls": "--",  "lw": 2.2,  "label": "Deterministic NMPC (Baseline)"},
    "Gaussian_SMPC":               {"color": "#984ea3", "ls": ":",   "lw": 2.2,  "label": "Gaussian SMPC (α=0.50)"},
    "Stage2_Visual_CVaR":          {"color": "#ff7f00", "ls": "-.",  "lw": 2.2,  "label": "Stage 2: Visual CVaR (Moon [9])"},
    "Stage3_Physical_CVaR_SMPC":   {"color": "#1b9e77", "ls": "-",   "lw": 3.0,  "label": "Stage 3: Physical CVaR — Ours ✓"},
}

PLANNER_ORDER = list(PLANNER_STYLES.keys())


def build_planner(name, rover, terrain, sensor, dt=0.25):
    if name == "Deterministic_NMPC":
        return DeterministicNMPCPlanner(rover, terrain, horizon=14, num_samples=120, dt=dt)
    elif name == "Gaussian_SMPC":
        return CVaRSMPCPlanner(rover, terrain, horizon=14, num_samples=120, alpha=0.50, dt=dt)
    elif name == "Stage2_Visual_CVaR":
        return AppearanceMPCPlanner(rover, terrain, sensor, horizon=14, num_samples=130, dt=dt)
    elif name == "Stage3_Physical_CVaR_SMPC":
        return Stage3PhysicallyGroundedCVaRPlanner(rover, terrain, sensor, horizon=16, num_samples=160, alpha=0.90, dt=dt)
    raise ValueError(name)


def run_all():
    rover = RoverModel()
    dt = 0.25
    max_steps = 400
    results = []
    traj_store = {}  # (map_idx, planner_name) → result dict

    print("\n" + "="*80)
    print("   TERRARISK KILLER MAP STRESS TEST — exposing NMPC/Gaussian SMPC failures")
    print("   3 Analytically Engineered Maps × 4 Controllers = 12 Trials")
    print("="*80)

    for m_idx, cfg in enumerate(MAP_CONFIGS):
        terrain = cfg["fn"]()
        sensor  = AppearanceSensor(rng_seed=99 + m_idx, terrain=terrain)
        start   = cfg["start"]
        goal    = cfg["goal"]
        direct  = float(np.linalg.norm(goal - start[:2]))

        print(f"\n>>> [{m_idx+1}/3] {cfg['name']}  (100m × 100m)")
        print(f"    {cfg['why_trapped'].splitlines()[0]}")
        print("-"*80)

        for p_name in PLANNER_ORDER:
            t0 = time.time()
            sim     = Simulator(terrain, rover, dt=dt)
            planner = build_planner(p_name, rover, terrain, sensor, dt=dt)
            res     = sim.run_trial(planner, start, goal, max_steps=max_steps)
            elapsed = time.time() - t0

            pl = float(res["path_length"])
            eff = (direct / pl) if (not res["entrapped"] and pl > 1.0) else 0.0
            z_max = res["max_sinkage"] * 100.0
            sinkages = res["sinkages"]
            accum = float(np.mean(np.maximum(0, sinkages - 0.5*rover.z_crit)**2)) if len(sinkages) else 0.0

            if res["reached_goal"]:    status = "✓ GOAL REACHED"
            elif res["entrapped"]:     status = "✗ ENTRAPPED  ← CONTROLLER FAILED"
            else:                      status = "⏱ TIMEOUT"

            print(f"  {p_name:28s} | {status:30s} | z_max={z_max:5.2f}cm | steps={res['steps']:3d}")

            results.append(dict(map=cfg["name"], map_idx=m_idx+1, planner=p_name,
                                reached_goal=res["reached_goal"], entrapped=res["entrapped"],
                                steps=res["steps"], path_length_m=pl, efficiency=eff,
                                max_sinkage_cm=z_max, accum_risk=accum, wall_time_s=elapsed))
            traj_store[(m_idx, p_name)] = res

    df = pd.DataFrame(results)
    df.to_csv("killer_maps_results.csv", index=False)

    # ── Aggregate summary ──────────────────────────────────────────────────────
    summary = df.groupby("planner").agg(
        Success_Rate   =("reached_goal", "mean"),
        Entrapment_Rate=("entrapped",    "mean"),
        Avg_Efficiency =("efficiency",   "mean"),
        Avg_Sinkage_cm =("max_sinkage_cm","mean"),
    ).reset_index()
    print("\n" + "="*80)
    print("   KILLER MAP OVERALL SUMMARY (3 maps)")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)

    # ── Visualise ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), dpi=200)
    fig.suptitle("TerraRisk Killer Maps — Controller Failure Analysis\n"
                 "✗ = ENTRAPPED   ✓ = GOAL REACHED   ◉ = False-Safe Deceptive Zone",
                 fontsize=13, fontweight="bold")

    for m_idx, cfg in enumerate(MAP_CONFIGS):
        ax  = axes[m_idx]
        terrain = cfg["fn"]()
        start   = cfg["start"]
        goal    = cfg["goal"]

        # Ground-truth sinkage heatmap
        R = 80
        xs = np.linspace(0, terrain.x_max, R)
        ys = np.linspace(0, terrain.y_max, R)
        XX, YY = np.meshgrid(xs, ys)
        ZZ = np.zeros_like(XX)
        for i in range(R):
            for j in range(R):
                s = terrain.get_soil_parameters(XX[i,j], YY[i,j], stochastic=False)
                z, _, _ = rover.compute_sinkage_and_slip(1.0, s)
                ZZ[i,j] = z * 100.0

        # Custom colormap: safe=green → dangerous=red with z_crit line visible
        cf = ax.contourf(XX, YY, ZZ, levels=np.linspace(0, 12, 20), cmap="RdYlGn_r", alpha=0.80, vmin=0, vmax=12)
        ax.contour(XX, YY, ZZ, levels=[8.0], colors=["black"], linewidths=[2.0],
                   linestyles=["--"])  # z_crit boundary

        cbar = plt.colorbar(cf, ax=ax, fraction=0.038, pad=0.04)
        cbar.set_label("Nominal Sinkage (cm)", fontsize=8)
        cbar.ax.axhline(8.0, color='black', linewidth=2, linestyle='--')
        cbar.ax.text(1.1, 8.0, "z_crit", transform=cbar.ax.get_yaxis_transform(),
                     fontsize=7, va='center', color='black')

        # Mark deceptive (ambiguous) patches with dotted purple circles + label
        for p in terrain.patches:
            px, py, r, sev, ambig, _ = terrain._parse_patch(p)
            if ambig:
                circ = plt.Circle((px, py), r, fill=False, edgecolor='purple',
                                   linewidth=2.2, linestyle='--')
                ax.add_patch(circ)
                ax.text(px, py + r + 1.5, "FALSE-SAFE\nSINKHOLE", color='purple',
                        fontsize=6.5, ha='center', fontweight='bold')

        # Plot trajectories
        for p_name in PLANNER_ORDER:
            res   = traj_store[(m_idx, p_name)]
            traj  = res["trajectory"]
            style = PLANNER_STYLES[p_name]

            ax.plot(traj[:,0], traj[:,1], color=style["color"], linestyle=style["ls"],
                    linewidth=style["lw"], alpha=0.9, label=style["label"])

            # Entrapment marker
            if res["entrapped"]:
                ax.plot(traj[-1,0], traj[-1,1], marker='X', color=style["color"],
                        markersize=14, markeredgecolor='black', markeredgewidth=1.5,
                        zorder=10)
                ax.annotate("TRAPPED!", xy=(traj[-1,0], traj[-1,1]),
                            xytext=(traj[-1,0]+4, traj[-1,1]+4),
                            fontsize=6.5, color=style["color"], fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=style["color"], lw=1.2))

        # Start / Goal markers
        ax.plot(start[0], start[1], 'o', color='lime', ms=10, markeredgecolor='black', zorder=11)
        ax.plot(goal[0],  goal[1],  '*', color='gold', ms=15, markeredgecolor='black', zorder=11)
        ax.text(start[0]+2, start[1]-4, "START", fontsize=8, color='lime', fontweight='bold')
        ax.text(goal[0]+2,  goal[1]-4,  "GOAL",  fontsize=8, color='gold', fontweight='bold')

        ax.set_xlim(0, terrain.x_max)
        ax.set_ylim(0, terrain.y_max)
        ax.set_title(f"{cfg['name']}\n(dashed line = z_crit boundary)", fontsize=10, fontweight='bold')
        ax.set_xlabel("X (m)", fontsize=9)
        ax.set_ylabel("Y (m)", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

        if m_idx == 0:
            ax.legend(loc='upper left', fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig("killer_maps_benchmark.png", dpi=200, bbox_inches='tight')
    print("\nVisualization saved → killer_maps_benchmark.png")
    return df


if __name__ == "__main__":
    run_all()
