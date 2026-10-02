import numpy as np
from .dynamics import RoverModel
from .terrain import TerrainMap

class CVaRSMPCPlanner:
    """
    Risk-Sensitive Model Predictive Path Integral (MPPI) Planner with CVaR tail-risk constraints.
    """
    def __init__(self, rover: RoverModel, terrain: TerrainMap, horizon=20, num_samples=300, alpha=0.90, dt=0.2):
        self.rover = rover
        self.terrain = terrain
        self.N = horizon
        self.M = num_samples
        self.alpha = alpha            # Risk level (0.90 for worst 10% tail risk)
        self.dt = dt
        
        self.noise_std = np.array([0.5, 0.6])
        self.temperature = 0.05
        
    def compute_cvar(self, sinkage_samples: np.ndarray) -> float:
        """
        Computes CVaR_alpha over rollout samples of maximum trajectory sinkage.
        """
        sorted_samples = np.sort(sinkage_samples)
        var_idx = int(np.ceil(self.alpha * len(sorted_samples))) - 1
        var_idx = max(0, min(len(sorted_samples) - 1, var_idx))
        return float(np.mean(sorted_samples[var_idx:]))

    def plan(self, current_state: np.ndarray, goal_pos: np.ndarray, prev_controls: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        Generates optimal risk-aware control action sequence [a, omega].
        """
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
                
                # Fetch stochastic soil parameters at candidate pose (x, y)
                soil = self.terrain.get_soil_parameters(state[0], state[1], stochastic=True)
                z, slip, _ = self.rover.compute_sinkage_and_slip(state[3], soil)
                sample_sinkages.append(z)
                
                dist_to_goal = np.hypot(state[0] - goal_pos[0], state[1] - goal_pos[1])
                traj_cost += dist_to_goal + 0.1 * (u_t[0]**2 + u_t[1]**2)
                
            max_z = np.max(sample_sinkages)
            max_sinkages[k] = max_z
            costs[k] = traj_cost
            
        # Compute CVaR of max sinkage
        cvar_risk = self.compute_cvar(max_sinkages)
        
        # Heavy penalty on rollouts exceeding 60% of critical sinkage
        cvar_threshold = self.rover.z_crit * 0.55
        tail_penalty = 10000.0 * np.maximum(0.0, max_sinkages - cvar_threshold)**2
        total_costs = costs + tail_penalty
        
        min_cost = np.min(total_costs)
        weights = np.exp(-1.0 / self.temperature * (total_costs - min_cost))
        weights /= (np.sum(weights) + 1e-8)
        
        U_opt = U + np.sum(weights[:, None, None] * delta_U, axis=0)
        
        info = {
            "cvar_risk": cvar_risk,
            "max_sinkage": np.max(max_sinkages),
            "expected_cost": np.mean(total_costs)
        }
        
        return U_opt, info
