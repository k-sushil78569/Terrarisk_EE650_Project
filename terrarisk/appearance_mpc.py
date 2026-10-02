import numpy as np
from .dynamics import RoverModel
from .terrain import TerrainMap
from .appearance_sensor import AppearanceSensor

class AppearanceMPCPlanner:
    """
    Stage 2 Baseline MPC Planner (reproducing Moon et al. [9] sensing bounds).
    Relies ONLY on LiDAR/Camera visual terrain classification and elevation/roughness maps.
    """
    def __init__(self, rover: RoverModel, terrain: TerrainMap, sensor: AppearanceSensor, horizon=20, num_samples=250, dt=0.2):
        self.rover = rover
        self.terrain = terrain
        self.sensor = sensor
        self.N = horizon
        self.M = num_samples
        self.dt = dt
        
        self.noise_std = np.array([0.5, 0.6])
        self.temperature = 0.05
        
    def plan(self, current_state: np.ndarray, goal_pos: np.ndarray, prev_controls: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        Plans path using prior soil estimates derived strictly from visual appearance classification.
        """
        if prev_controls is None:
            U = np.zeros((self.N, 2))
        else:
            U = np.vstack([prev_controls[1:], np.zeros((1, 2))])
            
        costs = np.zeros(self.M)
        delta_U = np.random.normal(0, self.noise_std, size=(self.M, self.N, 2))
        
        for k in range(self.M):
            state = current_state.copy()
            traj_cost = 0.0
            
            for t in range(self.N):
                u_t = np.clip(U[t] + delta_U[k, t], [-1.2, -1.2], [1.2, 1.2])
                state = self.rover.step_kinematics(state, u_t, self.dt)
                
                # Fetch actual ground truth soil for evaluating sensor input
                gt_soil = self.terrain.get_soil_parameters(state[0], state[1], stochastic=False)
                v_class, prior_soil, uncertainty = self.sensor.classify_terrain(state[0], state[1], gt_soil)
                
                # Predict sinkage strictly based on visual appearance prior
                z_vis, _, _ = self.rover.compute_sinkage_and_slip(state[3], prior_soil)
                
                dist_to_goal = np.hypot(state[0] - goal_pos[0], state[1] - goal_pos[1])
                
                # Visual cost map penalty
                visual_hazard_penalty = 0.0
                if "Loose" in v_class:
                    visual_hazard_penalty = 50.0  # Over-conservative avoidance of visually dark patches
                elif "Ambiguous" in v_class:
                    visual_hazard_penalty = 5.0   # Under-reacts to visually ambiguous soft patch!
                    
                traj_cost += dist_to_goal + 20.0 * z_vis + visual_hazard_penalty + 0.1 * (u_t[0]**2 + u_t[1]**2)
                
            costs[k] = traj_cost
            
        min_cost = np.min(costs)
        weights = np.exp(-1.0 / self.temperature * (costs - min_cost))
        weights /= (np.sum(weights) + 1e-8)
        
        U_opt = U + np.sum(weights[:, None, None] * delta_U, axis=0)
        
        return U_opt, {"expected_cost": np.mean(costs)}
