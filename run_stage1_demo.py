import numpy as np
import matplotlib.pyplot as plt
from terrarisk.terrain import TerrainMap
from terrarisk.dynamics import RoverModel
from terrarisk.cvar_smpc import CVaRSMPCPlanner
from terrarisk.deterministic_nmpc import DeterministicNMPCPlanner
from terrarisk.simulator import Simulator

def main():
    print("=== TerraRisk Stage 1 Simulation: CVaR-SMPC vs. Deterministic NMPC ===")
    
    # Initialize terrain, rover, and simulation environment
    terrain = TerrainMap(x_max=50.0, y_max=50.0, seed=100)
    rover = RoverModel()
    sim = Simulator(terrain, rover, dt=0.2)
    
    start_state = np.array([5.0, 5.0, np.pi/4, 0.5])
    goal_pos = np.array([42.0, 42.0])
    
    # Instantiate planners
    cvar_planner = CVaRSMPCPlanner(rover, terrain, horizon=15, num_samples=120, alpha=0.95)
    nmpc_planner = DeterministicNMPCPlanner(rover, terrain, horizon=15, num_samples=120)
    
    print("\nRunning Deterministic NMPC Trial...")
    nmpc_res = sim.run_trial(nmpc_planner, start_state, goal_pos, max_steps=250)
    print(f"Deterministic NMPC -> Reached Goal: {nmpc_res['reached_goal']}, Entrapped: {nmpc_res['entrapped']}, Steps: {nmpc_res['steps']}, Path Length: {nmpc_res['path_length']:.2f}m, Max Sinkage: {nmpc_res['max_sinkage']*100:.2f}cm")
    
    print("\nRunning Risk-Sensitive CVaR-SMPC Trial...")
    cvar_res = sim.run_trial(cvar_planner, start_state, goal_pos, max_steps=250)
    print(f"CVaR-SMPC          -> Reached Goal: {cvar_res['reached_goal']}, Entrapped: {cvar_res['entrapped']}, Steps: {cvar_res['steps']}, Path Length: {cvar_res['path_length']:.2f}m, Max Sinkage: {cvar_res['max_sinkage']*100:.2f}cm")
    
    # Plotting results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grid heatmap of soft patches
    X, Y = np.meshgrid(np.linspace(0, 50, 100), np.linspace(0, 50, 100))
    Z_grid = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            soil = terrain.get_soil_parameters(X[i,j], Y[i,j], stochastic=False)
            z, _, _ = rover.compute_sinkage_and_slip(1.0, soil)
            Z_grid[i,j] = z * 100.0 # cm
            
    im = axes[0].contourf(X, Y, Z_grid, levels=15, cmap='YlOrRd')
    plt.colorbar(im, ax=axes[0], label='Nominal Sinkage (cm)')
    
    # Plot trajectories
    nmpc_traj = nmpc_res['trajectory']
    cvar_traj = cvar_res['trajectory']
    
    axes[0].plot(nmpc_traj[:,0], nmpc_traj[:,1], 'r--', linewidth=2.5, label='Deterministic NMPC')
    axes[0].plot(cvar_traj[:,0], cvar_traj[:,1], 'b-', linewidth=2.5, label='CVaR-SMPC (Ours)')
    axes[0].plot(start_state[0], start_state[1], 'go', markersize=8, label='Start')
    axes[0].plot(goal_pos[0], goal_pos[1], 'r*', markersize=12, label='Goal')
    
    axes[0].set_title('Stage 1: Trajectory Comparison on Stochastic Terrain Map')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True)
    
    # Sinkage depth profile comparison
    axes[1].plot(nmpc_res['sinkages'] * 100.0, 'r--', label='Deterministic NMPC Sinkage')
    axes[1].plot(cvar_res['sinkages'] * 100.0, 'b-', label='CVaR-SMPC Sinkage')
    axes[1].axhline(y=rover.z_crit * 100.0, color='black', linestyle=':', label='Critical Entrapment Threshold')
    axes[1].set_title('Wheel Sinkage Profile over Time')
    axes[1].set_xlabel('Simulation Step')
    axes[1].set_ylabel('Sinkage Depth (cm)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    output_png = 'stage1_trajectory_comparison.png'
    plt.savefig(output_png, dpi=300)
    print(f"\nPlot saved successfully to {output_png}")

if __name__ == '__main__':
    main()
