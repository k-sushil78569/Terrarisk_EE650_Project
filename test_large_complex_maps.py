#!/usr/bin/env python3
"""
TerraRisk: Large & Complex Multi-Map Benchmark Test Harness
===========================================================
Executes closed-loop navigation benchmarking across multiple large, highly complex
terrains (100m x 100m) with varying topology, hazard distributions, and deceptive
'false-safe' regolith pockets.

Evaluates all 4 controller architectures:
  1. Deterministic NMPC (Nominal baseline)
  2. Gaussian SMPC (Chance-constrained baseline)
  3. Stage 2 Visual CVaR (Appearance-only MPC, Moon et al. [9])
  4. Stage 3 Physical CVaR-SMPC (Ours: in situ EKF + terramechanics risk tail-bounding)

Generates:
  - large_complex_maps_results.csv (quantitative metrics across all trials)
  - large_complex_maps_benchmark.png (4-panel publication visualization)
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from terrarisk.terrain import (
    TerrainMap,
    create_dune_field_map,
    create_crater_basin_map,
    create_canyon_chokepoint_map,
    create_procedural_complex_map
)
from terrarisk.dynamics import RoverModel
from terrarisk.appearance_sensor import AppearanceSensor
from terrarisk.deterministic_nmpc import DeterministicNMPCPlanner
from terrarisk.appearance_mpc import AppearanceMPCPlanner
from terrarisk.cvar_smpc import CVaRSMPCPlanner
from terrarisk.stage3_cvar_planner import Stage3PhysicallyGroundedCVaRPlanner
from terrarisk.simulator import Simulator


def get_map_configs():
    """Returns definitions of large, complex test terrains (100m x 100m)."""
    return [
        {
            "name": "Map 1: Planetary Dune Field",
            "map_fn": lambda: create_dune_field_map(seed=42),
            "start": np.array([10.0, 10.0, np.pi / 4, 0.5]),
            "goal": np.array([88.0, 88.0]),
            "description": "Undulating dunes with 4 deceptive drift sand traps on diagonal shortcuts."
        },
        {
            "name": "Map 2: Cratered Regolith Basin",
            "map_fn": lambda: create_crater_basin_map(seed=101),
            "start": np.array([10.0, 15.0, np.pi / 4, 0.5]),
            "goal": np.array([88.0, 85.0]),
            "description": "15 crater rim & basin hazards with deep, highly compressible regolith dust."
        },
        {
            "name": "Map 3: Canyon Chokepoint & Trap",
            "map_fn": lambda: create_canyon_chokepoint_map(seed=202),
            "start": np.array([10.0, 50.0, 0.0, 0.5]),
            "goal": np.array([90.0, 50.0]),
            "description": "Flanked corridors; central gap is a false-safe sinkhole requiring detour."
        },
        {
            "name": "Map 4: Procedural Chaos Field",
            "map_fn": lambda: create_procedural_complex_map(seed=777, num_patches=18),
            "start": np.array([10.0, 10.0, np.pi / 4, 0.5]),
            "goal": np.array([88.0, 88.0]),
            "description": "18 procedurally generated stochastic patches with 55% visual ambiguity."
        }
    ]


def build_planner(planner_name: str, rover: RoverModel, terrain: TerrainMap, sensor: AppearanceSensor, dt: float = 0.25):
    """Instantiates controller with tuning appropriate for 100m x 100m environments."""
    if planner_name == "Deterministic_NMPC":
        return DeterministicNMPCPlanner(rover, terrain, horizon=14, num_samples=100, dt=dt)
    elif planner_name == "Gaussian_SMPC":
        # Mean-centric chance-constrained SMPC proxy (alpha=0.50)
        return CVaRSMPCPlanner(rover, terrain, horizon=14, num_samples=100, alpha=0.50, dt=dt)
    elif planner_name == "Stage2_Visual_CVaR":
        return AppearanceMPCPlanner(rover, terrain, sensor, horizon=14, num_samples=110, dt=dt)
    elif planner_name == "Stage3_Physical_CVaR_SMPC":
        return Stage3PhysicallyGroundedCVaRPlanner(rover, terrain, sensor, horizon=15, num_samples=130, alpha=0.90, dt=dt)
    else:
        raise ValueError(f"Unknown planner: {planner_name}")


def run_benchmark():
    rover = RoverModel()
    map_configs = get_map_configs()
    planner_names = [
        "Deterministic_NMPC",
        "Gaussian_SMPC",
        "Stage2_Visual_CVaR",
        "Stage3_Physical_CVaR_SMPC"
    ]
    
    dt = 0.25
    max_steps = 360
    
    results = []
    run_trajectories = {}  # (map_idx, planner_name) -> result dict
    
    total_runs = len(map_configs) * len(planner_names)
    run_idx = 0
    t_start_total = time.time()
    
    print("\n" + "=" * 80)
    print("      TERRARISK LARGE & COMPLEX MULTI-MAP BENCHMARK (100m x 100m)")
    print(f"      {len(map_configs)} Diverse Maps  ×  {len(planner_names)} Controllers  =  {total_runs} Total Trials")
    print("=" * 80)
    
    for m_idx, m_cfg in enumerate(map_configs):
        terrain = m_cfg["map_fn"]()
        sensor = AppearanceSensor(rng_seed=42 + m_idx, terrain=terrain)
        start_state = m_cfg["start"]
        goal_pos = m_cfg["goal"]
        direct_dist = float(np.linalg.norm(goal_pos - start_state[:2]))
        
        print(f"\n>>> [Map {m_idx + 1}/4]: {m_cfg['name']} (Size: {terrain.x_max:.0f}m x {terrain.y_max:.0f}m)")
        print(f"    Features: {m_cfg['description']}")
        print(f"    Start: ({start_state[0]:.1f}, {start_state[1]:.1f}) -> Goal: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f}) [Direct Dist: {direct_dist:.1f}m]")
        print("-" * 80)
        
        for p_name in planner_names:
            run_idx += 1
            t0 = time.time()
            
            # Fresh planner and simulation instance per trial
            sim = Simulator(terrain, rover, dt=dt)
            planner = build_planner(p_name, rover, terrain, sensor, dt=dt)
            
            res = sim.run_trial(planner, start_state, goal_pos, max_steps=max_steps)
            elapsed = time.time() - t0
            
            path_len = float(res["path_length"])
            efficiency = (direct_dist / path_len) if (not res["entrapped"] and path_len > 1.0) else 0.0
            max_sink_cm = float(res["max_sinkage"] * 100.0)
            
            sinkages = res["sinkages"]
            accum_risk = float(np.mean(np.maximum(0, sinkages - 0.5 * rover.z_crit) ** 2)) if len(sinkages) > 0 else 0.0
            
            status_str = "✓ GOAL" if res["reached_goal"] else ("✗ ENTRAPPED" if res["entrapped"] else "⏱ TIMEOUT")
            
            print(f"  [{run_idx:2d}/{total_runs}] {p_name:26s} | {status_str:12s} | "
                  f"Sinkage: {max_sink_cm:5.2f}cm | Steps: {res['steps']:3d} | "
                  f"Path: {path_len:6.1f}m | Time: {elapsed:5.1f}s")
            
            results.append({
                "map": m_cfg["name"],
                "map_idx": m_idx + 1,
                "planner": p_name,
                "reached_goal": res["reached_goal"],
                "entrapped": res["entrapped"],
                "steps": res["steps"],
                "path_length_m": path_len,
                "efficiency": efficiency,
                "max_sinkage_cm": max_sink_cm,
                "mean_sinkage_cm": float(res["mean_sinkage"] * 100.0),
                "accum_risk": accum_risk,
                "wall_time_s": elapsed
            })
            
            run_trajectories[(m_idx, p_name)] = res
            
    total_time = (time.time() - t_start_total) / 60.0
    df = pd.DataFrame(results)
    csv_file = "large_complex_maps_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSaved benchmark metrics to {csv_file} (Total execution: {total_time:.2f} mins)")
    
    # =========================================================================
    # Aggregated Summary Table
    # =========================================================================
    summary = df.groupby("planner").agg(
        Success_Rate=("reached_goal", "mean"),
        Entrapment_Rate=("entrapped", "mean"),
        Avg_Efficiency=("efficiency", "mean"),
        Avg_Max_Sinkage_cm=("max_sinkage_cm", "mean"),
        Avg_Risk=("accum_risk", "mean"),
        Avg_Steps=("steps", "mean")
    ).reset_index()
    
    print("\n" + "=" * 80)
    print("                  LARGE COMPLEX MAPS - OVERALL BENCHMARK SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80 + "\n")
    
    # =========================================================================
    # Generate 4-Panel Visualization Plot
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 15), dpi=300)
    axes = axes.flatten()
    
    planner_styles = {
        "Deterministic_NMPC": {"color": "#e41a1c", "ls": "--", "lw": 2.0, "label": "Deterministic NMPC"},
        "Gaussian_SMPC": {"color": "#984ea3", "ls": ":", "lw": 2.2, "label": "Gaussian SMPC"},
        "Stage2_Visual_CVaR": {"color": "#ff7f00", "ls": "-.", "lw": 2.4, "label": "Stage 2: Visual CVaR (Moon [9])"},
        "Stage3_Physical_CVaR_SMPC": {"color": "#1b9e77", "ls": "-", "lw": 2.8, "label": "Stage 3: Physical CVaR (Ours)"}
    }
    
    for m_idx, m_cfg in enumerate(map_configs):
        ax = axes[m_idx]
        terrain = m_cfg["map_fn"]()
        start_state = m_cfg["start"]
        goal_pos = m_cfg["goal"]
        
        # High-res contour grid of sinkage hazard
        res_grid = 80
        xs = np.linspace(0, terrain.x_max, res_grid)
        ys = np.linspace(0, terrain.y_max, res_grid)
        XX, YY = np.meshgrid(xs, ys)
        ZZ = np.zeros_like(XX)
        
        for i in range(res_grid):
            for j in range(res_grid):
                s = terrain.get_soil_parameters(XX[i, j], YY[i, j], stochastic=False)
                z, _, _ = rover.compute_sinkage_and_slip(1.0, s)
                ZZ[i, j] = z * 100.0  # cm
                
        cf = ax.contourf(XX, YY, ZZ, levels=14, cmap="YlOrRd", alpha=0.85)
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Ground-Truth Sinkage Hazard (cm)", fontsize=9)
        
        # Highlight ambiguous false-safe patches
        for p in terrain.patches:
            px, py, radius, _, ambiguous, _ = terrain._parse_patch(p)
            if ambiguous:
                circle = plt.Circle((px, py), radius, color='purple', fill=False, linestyle='--', linewidth=1.8)
                ax.add_patch(circle)
                ax.text(px, py, "False-Safe\nRegolith", color='purple', fontsize=8, weight='bold', ha='center', va='center')
                
        # Overlay Trajectories
        for p_name in planner_names:
            t_res = run_trajectories[(m_idx, p_name)]
            traj = t_res["trajectory"]
            style = planner_styles[p_name]
            
            ax.plot(traj[:, 0], traj[:, 1], color=style["color"], linestyle=style["ls"],
                    linewidth=style["lw"], label=style["label"])
            
            # If entrapped, mark termination with red X
            if t_res["entrapped"]:
                ax.plot(traj[-1, 0], traj[-1, 1], marker='X', color='red', markersize=10, markeredgecolor='black')
                
        # Mark Start and Goal
        ax.plot(start_state[0], start_state[1], marker='o', color='lime', markersize=9, markeredgecolor='black', label='Start')
        ax.plot(goal_pos[0], goal_pos[1], marker='*', color='gold', markersize=14, markeredgecolor='black', label='Goal')
        
        ax.set_title(f"{m_cfg['name']}\n({terrain.x_max:.0f}m × {terrain.y_max:.0f}m)", fontsize=11, weight='bold')
        ax.set_xlabel("X (meters)", fontsize=10)
        ax.set_ylabel("Y (meters)", fontsize=10)
        ax.set_xlim(0, terrain.x_max)
        ax.set_ylim(0, terrain.y_max)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if m_idx == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
            
    plt.tight_layout()
    output_png = "large_complex_maps_benchmark.png"
    plt.savefig(output_png, dpi=300)
    print(f"Publication-quality visualization saved to: {output_png}\n")


if __name__ == "__main__":
    run_benchmark()
