import numpy as np
from .dynamics import RoverModel
from .terrain import TerrainMap
from .appearance_sensor import AppearanceSensor
from .online_estimator import OnlineSoilEstimator

class Stage3PhysicallyGroundedCVaRPlanner:
    """
    Stage 3 CVaR-SMPC Planner featuring in situ proprioceptive soil parameter estimation.
    Fuses wheel slip, sinkage depth, motor torque, and IMU data to continuously update Bekker-Wong soil estimates.
    """
    def __init__(self, rover: RoverModel, terrain: TerrainMap, sensor: AppearanceSensor, horizon=22, num_samples=350, alpha=0.90, dt=0.2):
        self.rover = rover
        self.terrain = terrain
        self.sensor = sensor
        self.estimator = OnlineSoilEstimator(rover_mass=rover.mass)
        self.N = horizon
        self.M = num_samples
        self.alpha = alpha
        self.dt = dt
        
        self.noise_std = np.array([0.5, 0.6])
        self.temperature = 0.05
        self.current_estimated_soil = None
        
    def compute_cvar(self, sinkage_samples: np.ndarray) -> float:
        sorted_samples = np.sort(sinkage_samples)
        var_idx = int(np.ceil(self.alpha * len(sorted_samples))) - 1
        var_idx = max(0, min(len(sorted_samples) - 1, var_idx))
        return float(np.mean(sorted_samples[var_idx:]))

    def update_online_sensing(self, current_state: np.ndarray):
        """
        Reads physical observations (sinkage, slip, torque) and updates estimator.
        """
        gt_soil = self.terrain.get_soil_parameters(current_state[0], current_state[1], stochastic=True)
        z_actual, slip_actual, torque_actual = self.rover.compute_sinkage_and_slip(current_state[3], gt_soil)
        
        self.current_estimated_soil = self.estimator.update(z_actual, slip_actual, torque_actual, current_state[3])

    def plan(self, current_state: np.ndarray, goal_pos: np.ndarray, prev_controls: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        Plans path using online fused physical estimate + CVaR tail risk bound.
        """
        self.update_online_sensing(current_state)
        
        if prev_controls is None:
            U = np.zeros((self.N, 2))
        else:
            U = np.vstack([prev_controls[1:], np.zeros((1, 2))])
            
        costs = np.zeros(self.M)
        max_sinkages = np.zeros(self.M)
        
        delta_U = np.random.normal(0, self.noise_std, size=(self.M, self.N, 2))
        
        for k in range(self.M):
            state = current_state.copy()
            traj_cost = 0.0
            sample_sinkages = []
            
            for t in range(self.N):
                u_t = np.clip(U[t] + delta_U[k, t], [-1.2, -1.2], [1.2, 1.2])
                state = self.rover.step_kinematics(state, u_t, self.dt)
                
                # Fetch online estimated soil parameters for dynamic prediction
                z, slip, _ = self.rover.compute_sinkage_and_slip(state[3], self.current_estimated_soil)
                sample_sinkages.append(z)
                
                dist_to_goal = np.hypot(state[0] - goal_pos[0], state[1] - goal_pos[1])
                traj_cost += dist_to_goal + 0.1 * (u_t[0]**2 + u_t[1]**2)
                
            max_z = np.max(sample_sinkages)
            max_sinkages[k] = max_z
            costs[k] = traj_cost
            
        cvar_risk = self.compute_cvar(max_sinkages)
        
        # Severe penalty on predicted sinkage exceeding safe operating envelope
        cvar_threshold = self.rover.z_crit * 0.50
        tail_penalty = 15000.0 * np.maximum(0.0, max_sinkages - cvar_threshold)**2
        total_costs = costs + tail_penalty
        
        min_cost = np.min(total_costs)
        weights = np.exp(-1.0 / self.temperature * (total_costs - min_cost))
        weights /= (np.sum(weights) + 1e-8)
        
        U_opt = U + np.sum(weights[:, None, None] * delta_U, axis=0)
        
        info = {
            "cvar_risk": cvar_risk,
            "estimated_cohesion": self.current_estimated_soil.c,
            "estimated_friction_deg": np.degrees(self.current_estimated_soil.phi)
        }
        
        return U_opt, info
