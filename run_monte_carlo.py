#!/usr/bin/env python3
"""
TerraRisk: Corrected Monte Carlo Benchmark Runner
==================================================
Uses actual planner classes and full closed-loop Simulator for differentiated results.
Replaces the simplified point-mass heuristic in terrarisk/run_monte_carlo.py.

Usage:
    python run_monte_carlo.py              # Full 50-seed benchmark
    python run_monte_carlo.py --quick 5    # Quick 5-seed smoke test
"""
import argparse
import time
import numpy as np
import pandas as pd

from terrarisk.terrain import TerrainMap
from terrarisk.dynamics import RoverModel
from terrarisk.appearance_sensor import AppearanceSensor
from terrarisk.deterministic_nmpc import DeterministicNMPCPlanner
from terrarisk.appearance_mpc import AppearanceMPCPlanner
from terrarisk.cvar_smpc import CVaRSMPCPlanner
from terrarisk.stage3_cvar_planner import Stage3PhysicallyGroundedCVaRPlanner
from terrarisk.simulator import Simulator

PLANNERS = [
    "Deterministic_NMPC",
    "Gaussian_SMPC",
    "Stage2_Visual_CVaR",
    "Stage3_Physical_CVaR_SMPC",
]


def create_planner(name: str, rover: RoverModel, terrain: TerrainMap,
                   sensor: AppearanceSensor):
    """Instantiates the correct planner with benchmark-appropriate hyper-parameters."""
    if name == "Deterministic_NMPC":
        return DeterministicNMPCPlanner(rover, terrain, horizon=10, num_samples=80)
    elif name == "Gaussian_SMPC":
        # Gaussian SMPC proxy: CVaR-SMPC with low alpha (mean-centric, not tail-aware)
        # α = 0.50 ⇒ CVaR ≈ expected value of top-50 % sinkage (mild risk weighting)
        return CVaRSMPCPlanner(rover, terrain, horizon=10, num_samples=80, alpha=0.50)
    elif name == "Stage2_Visual_CVaR":
        return AppearanceMPCPlanner(rover, terrain, sensor, horizon=12, num_samples=100)
    elif name == "Stage3_Physical_CVaR_SMPC":
        return Stage3PhysicallyGroundedCVaRPlanner(
            rover, terrain, sensor, horizon=12, num_samples=120, alpha=0.90
        )
    else:
        raise ValueError(f"Unknown planner: {name}")


def run_single_trial(seed: int, planner_name: str, max_steps: int = 200) -> dict:
    """Executes one closed-loop navigation trial and returns metrics."""
    # Seed control for reproducibility
    np.random.seed(seed * 100 + PLANNERS.index(planner_name))

    terrain = TerrainMap(x_max=50.0, y_max=50.0, seed=seed)
    rover = RoverModel()
    sensor = AppearanceSensor(rng_seed=seed)
    sim = Simulator(terrain, rover, dt=0.2)

    start_state = np.array([5.0, 5.0, np.pi / 4, 0.5])
    goal_pos = np.array([42.0, 42.0])

    planner = create_planner(planner_name, rover, terrain, sensor)
    result = sim.run_trial(planner, start_state, goal_pos, max_steps=max_steps)

    # Derived metrics
    direct_dist = np.linalg.norm(goal_pos - start_state[:2])
    path_len = result["path_length"]
    efficiency = (direct_dist / path_len) if (not result["entrapped"] and path_len > 1.0) else 0.0

    sinkages = result["sinkages"]
    z_crit = rover.z_crit
    accum_risk = float(np.mean(np.maximum(0, sinkages - 0.5 * z_crit) ** 2)) if len(sinkages) > 0 else 0.0

    return {
        "seed": seed,
        "planner": planner_name,
        "success": result["reached_goal"],
        "entrapped": result["entrapped"],
        "path_length": float(path_len),
        "efficiency": float(efficiency),
        "max_sinkage": float(result["max_sinkage"]),
        "mean_sinkage": float(result["mean_sinkage"]),
        "steps": result["steps"],
        "accum_risk": accum_risk,
    }


def run_benchmark(num_seeds: int = 50, max_steps: int = 200):
    """Executes the full Monte Carlo benchmark matrix."""
    results = []
    total = num_seeds * len(PLANNERS)
    t0 = time.time()

    print(f"\n{'='*72}")
    print(f"  TerraRisk Monte Carlo Benchmark")
    print(f"  {num_seeds} seeds × {len(PLANNERS)} planners = {total} trials")
    print(f"  Max steps per trial: {max_steps}")
    print(f"{'='*72}\n")

    for seed in range(num_seeds):
        for pi, planner_name in enumerate(PLANNERS):
            trial_num = seed * len(PLANNERS) + pi + 1

            trial_result = run_single_trial(seed, planner_name, max_steps)
            results.append(trial_result)

            elapsed = time.time() - t0
            eta = (elapsed / trial_num) * (total - trial_num) if trial_num > 0 else 0

            if trial_result["success"]:
                status = "✓ GOAL"
            elif trial_result["entrapped"]:
                status = "✗ ENTRAPPED"
            else:
                status = "— timeout"

            print(
                f"  [{trial_num:3d}/{total}]  seed={seed:2d}  "
                f"{planner_name:30s}  {status:14s}  "
                f"z_max={trial_result['max_sinkage']*100:5.2f}cm  "
                f"steps={trial_result['steps']:3d}  "
                f"(ETA: {eta/60:.1f} min)"
            )

    # Save raw results
    df = pd.DataFrame(results)
    csv_path = "benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    # Aggregated summary
    summary = df.groupby("planner").agg(
        Success_Rate=("success", "mean"),
        Entrapment_Rate=("entrapped", "mean"),
        Avg_Efficiency=("efficiency", "mean"),
        Avg_Max_Sinkage_cm=("max_sinkage", lambda x: x.mean() * 100),
        Avg_Accum_Risk=("accum_risk", "mean"),
    ).reset_index()

    wall_time = (time.time() - t0) / 60.0

    print(f"\n{'='*72}")
    print("  FINAL BENCHMARK METRICS")
    print(f"{'='*72}")
    print(summary.to_string(index=False))
    print(f"{'='*72}")
    print(f"  Total wall-clock time: {wall_time:.1f} minutes")
    print(f"  Results saved to: {csv_path}")
    print(f"{'='*72}\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TerraRisk Monte Carlo Benchmark")
    parser.add_argument("--quick", type=int, default=None,
                        help="Quick smoke test with N seeds (default: full 50)")
    args = parser.parse_args()

    num_seeds = args.quick if args.quick else 50
    run_benchmark(num_seeds=num_seeds)
