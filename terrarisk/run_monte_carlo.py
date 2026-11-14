import json
import numpy as np
import pandas as pd
from terrain_generator import TerrainMap

PLANNERS = ["Deterministic_NMPC", "Gaussian_SMPC", "Stage2_Visual_CVaR", "Stage3_Physical_CVaR_SMPC"]

def simulate_trial(seed: int, planner_type: str) -> dict:
    """
    Simulates a single trajectory run for a given seed and planner architecture.
    """
    terrain = TerrainMap(x_max=50.0, y_max=50.0, resolution=0.5, seed=seed)
    rng = np.random.default_rng(seed)
    
    # State initialization
    pos = np.array([0.0, 0.0])
    goal = np.array([45.0, 45.0])
    max_steps = 150
    dt = 0.5
    
    path_length = 0.0
    entrapped = False
    max_sinkage = 0.0
    
    for step in range(max_steps):
        dist_to_goal = np.linalg.norm(goal - pos)
        if dist_to_goal < 2.0:
            break  # Goal reached
            
        # True physical soil at current rover position
        true_soil = terrain.get_ground_truth_soil(pos[0], pos[1], stochastic=True)
        # Visual/Camera classification at current rover position
        vis_risk = terrain.get_visual_appearance_risk(pos[0], pos[1])
        
        # Direct heading toward goal with planner-specific adjustments
        dir_vector = (goal - pos) / dist_to_goal
        
        # Controller decision dynamics
        if planner_type == "Deterministic_NMPC":
            # Direct path; blind to risk
            step_dir = dir_vector
            
        elif planner_type == "Gaussian_SMPC":
            # Mild avoidance based on mean parameter variances
            perturb = rng.normal(0, 0.1, size=2)
            step_dir = dir_vector + perturb
            
        elif planner_type == "Stage2_Visual_CVaR":
            # Avoids high visual risk regions (tricked by deceptive soft patches)
            detour = np.array([-dir_vector[1], dir_vector[0]]) * vis_risk * 2.0
            step_dir = dir_vector - detour
            
        elif planner_type == "Stage3_Physical_CVaR_SMPC":
            # Proprioceptive feedback: detects low cohesion 'c' in real-time
            physical_hazard = max(0.0, (2.0 - true_soil.c) / 2.0)
            detour = np.array([-dir_vector[1], dir_vector[0]]) * physical_hazard * 3.5
            step_dir = dir_vector - detour
            
        # Normalize motion vector
        step_dir = step_dir / np.linalg.norm(step_dir)
        step_dist = 0.4
        pos += step_dir * step_dist
        path_length += step_dist
        
        # Terramechanics entrapment condition: low cohesion leads to sinkage
        sinkage = 0.12 / max(0.1, true_soil.c)
        max_sinkage = max(max_sinkage, sinkage)
        
        if sinkage > 0.35: # Entrapment threshold (35 cm)
            entrapped = True
            break

    direct_dist = np.linalg.norm(goal - np.array([0.0, 0.0]))
    efficiency = direct_dist / max(path_length, 1.0) if not entrapped else 0.0
    success = (not entrapped) and (np.linalg.norm(goal - pos) < 2.0)

    return {
        "seed": seed,
        "planner": planner_type,
        "success": success,
        "entrapped": entrapped,
        "path_length": float(path_length),
        "efficiency": float(efficiency),
        "max_sinkage": float(max_sinkage)
    }

def run_benchmark_suite(num_trials=50):
    """Executes full Monte Carlo test matrix across seeds and planners."""
    results = []
    print(f"Starting Monte Carlo Benchmark ({num_trials} seeds x {len(PLANNERS)} planners)...")
    
    for seed in range(num_trials):
        for planner in PLANNERS:
            res = simulate_trial(seed=seed, planner_type=planner)
            results.append(res)
            
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    
    # Aggregated Summary Table
    summary = df.groupby("planner").agg(
        Success_Rate=("success", "mean"),
        Entrapment_Rate=("entrapped", "mean"),
        Avg_Path_Efficiency=("efficiency", "mean"),
        Avg_Max_Sinkage=("max_sinkage", "mean")
    ).reset_index()
    
    print("\n================ FINAL BENCHMARK METRICS ================")
    print(summary.to_string(index=False))
    print("========================================================")

if __name__ == "__main__":
    run_benchmark_suite(num_trials=50)