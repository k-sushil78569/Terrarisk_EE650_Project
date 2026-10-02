import numpy as np
import matplotlib.pyplot as plt
from terrarisk.terrain import TerrainMap
from terrarisk.dynamics import RoverModel
from terrarisk.appearance_sensor import AppearanceSensor
from terrarisk.deterministic_nmpc import DeterministicNMPCPlanner
from terrarisk.appearance_mpc import AppearanceMPCPlanner
from terrarisk.stage3_cvar_planner import Stage3PhysicallyGroundedCVaRPlanner
from terrarisk.simulator import Simulator
from pychrono_sim.chrono_wrapper import PyChronoSimRunner

def main():
    print("=== TerraRisk Stage 3 Comprehensive Benchmark ===")
    print("Testing Stage 1 (NMPC) vs. Stage 2 (Appearance-Only MPC) vs. Stage 3 (Physically-Grounded CVaR-SMPC)\n")
    
    terrain = TerrainMap(x_max=50.0, y_max=50.0, seed=42)
    rover = RoverModel()
    sensor = AppearanceSensor(rng_seed=42)
    sim = Simulator(terrain, rover, dt=0.2)
    
    # Initialize PyChrono high-fidelity multibody engine wrapper
    chrono_sim = PyChronoSimRunner(terrain)
    
    start_state = np.array([5.0, 5.0, np.pi/4, 0.5])
    goal_pos = np.array([42.0, 42.0])
    
    nmpc_planner = DeterministicNMPCPlanner(rover, terrain, horizon=20, num_samples=250)
    app_planner = AppearanceMPCPlanner(rover, terrain, sensor, horizon=20, num_samples=250)
    stage3_planner = Stage3PhysicallyGroundedCVaRPlanner(rover, terrain, sensor, horizon=20, num_samples=300)
    
    print("[Stage 1] Running Deterministic NMPC...")
    res_nmpc = sim.run_trial(nmpc_planner, start_state, goal_pos, max_steps=250)
    print(f"  -> Reached Goal: {res_nmpc['reached_goal']}, Entrapped: {res_nmpc['entrapped']}, Steps: {res_nmpc['steps']}, Path: {res_nmpc['path_length']:.2f}m, Max Sinkage: {res_nmpc['max_sinkage']*100:.2f}cm")
    
    print("\n[Stage 2] Running Appearance-Only MPC (Moon et al. [9])...")
    res_app = sim.run_trial(app_planner, start_state, goal_pos, max_steps=250)
    print(f"  -> Reached Goal: {res_app['reached_goal']}, Entrapped: {res_app['entrapped']}, Steps: {res_app['steps']}, Path: {res_app['path_length']:.2f}m, Max Sinkage: {res_app['max_sinkage']*100:.2f}cm")
    
    print("\n[Stage 3] Running In-Situ Physically-Grounded CVaR-SMPC (Ours)...")
    res_s3 = sim.run_trial(stage3_planner, start_state, goal_pos, max_steps=250)
    print(f"  -> Reached Goal: {res_s3['reached_goal']}, Entrapped: {res_s3['entrapped']}, Steps: {res_s3['steps']}, Path: {res_s3['path_length']:.2f}m, Max Sinkage: {res_s3['max_sinkage']*100:.2f}cm")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    X, Y = np.meshgrid(np.linspace(0, 50, 100), np.linspace(0, 50, 100))
    Z_grid = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            soil = terrain.get_soil_parameters(X[i,j], Y[i,j], stochastic=False)
            z, _, _ = rover.compute_sinkage_and_slip(1.0, soil)
            Z_grid[i,j] = z * 100.0
            
    im = axes[0].contourf(X, Y, Z_grid, levels=15, cmap='YlOrRd')
    plt.colorbar(im, ax=axes[0], label='Ground Truth Sinkage Hazard (cm)')
    
    axes[0].plot(res_nmpc['trajectory'][:,0], res_nmpc['trajectory'][:,1], 'r--', linewidth=2, label='Stage 1: NMPC')
    axes[0].plot(res_app['trajectory'][:,0], res_app['trajectory'][:,1], 'm-.', linewidth=2.5, label='Stage 2: Appearance MPC (Moon [9])')
    axes[0].plot(res_s3['trajectory'][:,0], res_s3['trajectory'][:,1], 'b-', linewidth=2.5, label='Stage 3: Physical CVaR-SMPC (Ours)')
    axes[0].plot(start_state[0], start_state[1], 'go', markersize=8, label='Start')
    axes[0].plot(goal_pos[0], goal_pos[1], 'r*', markersize=12, label='Goal')
    
    axes[0].set_title('Stage 3 Final Benchmark: Trajectory Comparison')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True)
    
    # Sinkage comparison over time
    axes[1].plot(res_nmpc['sinkages'] * 100.0, 'r--', label='Stage 1 NMPC')
    axes[1].plot(res_app['sinkages'] * 100.0, 'm-.', label='Stage 2 Appearance MPC')
    axes[1].plot(res_s3['sinkages'] * 100.0, 'b-', label='Stage 3 In-Situ Physical CVaR')
    axes[1].axhline(y=rover.z_crit * 100.0, color='black', linestyle=':', label='Critical Entrapment Threshold (8cm)')
    axes[1].set_title('Sinkage Profiles & Tail Risk Bounding')
    axes[1].set_xlabel('Simulation Step')
    axes[1].set_ylabel('Sinkage Depth (cm)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True)
    
    plt.tight_layout()
    output_png = 'stage3_final_benchmark.png'
    plt.savefig(output_png, dpi=300)
    print(f"\nStage 3 benchmark plot saved successfully to {output_png}")

if __name__ == '__main__':
    main()
