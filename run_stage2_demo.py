import numpy as np
import matplotlib.pyplot as plt
from terrarisk.terrain import TerrainMap
from terrarisk.dynamics import RoverModel
from terrarisk.appearance_sensor import AppearanceSensor
from terrarisk.appearance_mpc import AppearanceMPCPlanner
from terrarisk.cvar_smpc import CVaRSMPCPlanner
from terrarisk.deterministic_nmpc import DeterministicNMPCPlanner
from terrarisk.simulator import Simulator

def main():
    print("=== TerraRisk Phase 2 Benchmark: Appearance-Only MPC vs. Physically-Grounded CVaR-SMPC ===")
    
    terrain = TerrainMap(x_max=50.0, y_max=50.0, seed=42)
    rover = RoverModel()
    sensor = AppearanceSensor(rng_seed=42)
    sim = Simulator(terrain, rover, dt=0.2)
    
    start_state = np.array([5.0, 5.0, np.pi/4, 0.5])
    goal_pos = np.array([42.0, 42.0])
    
    # Planners
    nmpc_planner = DeterministicNMPCPlanner(rover, terrain, horizon=20, num_samples=250)
    app_planner = AppearanceMPCPlanner(rover, terrain, sensor, horizon=20, num_samples=250)
    cvar_planner = CVaRSMPCPlanner(rover, terrain, horizon=20, num_samples=300, alpha=0.90)
    
    print("\n[1/3] Running Deterministic NMPC Trial (Phase 1 Baseline)...")
    res_nmpc = sim.run_trial(nmpc_planner, start_state, goal_pos, max_steps=250)
    print(f"Deterministic NMPC -> Reached Goal: {res_nmpc['reached_goal']}, Entrapped: {res_nmpc['entrapped']}, Steps: {res_nmpc['steps']}, Path Length: {res_nmpc['path_length']:.2f}m, Max Sinkage: {res_nmpc['max_sinkage']*100:.2f}cm")
    
    print("\n[2/3] Running Appearance-Only MPC Trial (Phase 2 Baseline - Moon et al. [9])...")
    res_app = sim.run_trial(app_planner, start_state, goal_pos, max_steps=250)
    print(f"Appearance-Only MPC -> Reached Goal: {res_app['reached_goal']}, Entrapped: {res_app['entrapped']}, Steps: {res_app['steps']}, Path Length: {res_app['path_length']:.2f}m, Max Sinkage: {res_app['max_sinkage']*100:.2f}cm")
    
    print("\n[3/3] Running Physically-Grounded CVaR-SMPC Trial (Ours)...")
    res_cvar = sim.run_trial(cvar_planner, start_state, goal_pos, max_steps=250)
    print(f"CVaR-SMPC (Ours)    -> Reached Goal: {res_cvar['reached_goal']}, Entrapped: {res_cvar['entrapped']}, Steps: {res_cvar['steps']}, Path Length: {res_cvar['path_length']:.2f}m, Max Sinkage: {res_cvar['max_sinkage']*100:.2f}cm")
    
    # Plotting benchmarking comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    X, Y = np.meshgrid(np.linspace(0, 50, 100), np.linspace(0, 50, 100))
    Z_grid = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            soil = terrain.get_soil_parameters(X[i,j], Y[i,j], stochastic=False)
            z, _, _ = rover.compute_sinkage_and_slip(1.0, soil)
            Z_grid[i,j] = z * 100.0
            
    im = axes[0].contourf(X, Y, Z_grid, levels=15, cmap='YlOrRd')
    plt.colorbar(im, ax=axes[0], label='Nominal Soil Sinkage Hazard (cm)')
    
    # Annotate visually ambiguous patch
    axes[0].text(28.0, 31.0, 'Visually Ambiguous\nSoft Patch', color='darkred', weight='bold', fontsize=9, ha='center')
    
    axes[0].plot(res_nmpc['trajectory'][:,0], res_nmpc['trajectory'][:,1], 'r--', linewidth=2, label='Deterministic NMPC')
    axes[0].plot(res_app['trajectory'][:,0], res_app['trajectory'][:,1], 'm-.', linewidth=2.5, label='Appearance-Only MPC (Moon [9])')
    axes[0].plot(res_cvar['trajectory'][:,0], res_cvar['trajectory'][:,1], 'b-', linewidth=2.5, label='CVaR-SMPC (Ours)')
    axes[0].plot(start_state[0], start_state[1], 'go', markersize=8, label='Start')
    axes[0].plot(goal_pos[0], goal_pos[1], 'r*', markersize=12, label='Goal')
    
    axes[0].set_title('Phase 2 Benchmark: Trajectories on Ambiguous Terrain')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True)
    
    # Sinkage Profiles
    axes[1].plot(res_nmpc['sinkages'] * 100.0, 'r--', label='Deterministic NMPC')
    axes[1].plot(res_app['sinkages'] * 100.0, 'm-.', label='Appearance-Only MPC (Moon [9])')
    axes[1].plot(res_cvar['sinkages'] * 100.0, 'b-', label='CVaR-SMPC (Ours)')
    axes[1].axhline(y=rover.z_crit * 100.0, color='black', linestyle=':', label='Entrapment Threshold (8cm)')
    axes[1].set_title('Sinkage Profiles & Entrapment Failures')
    axes[1].set_xlabel('Simulation Step')
    axes[1].set_ylabel('Sinkage Depth (cm)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True)
    
    plt.tight_layout()
    output_png = 'stage2_appearance_vs_physical.png'
    plt.savefig(output_png, dpi=300)
    print(f"\nPhase 2 Benchmark plot saved successfully to {output_png}")

if __name__ == '__main__':
    main()
