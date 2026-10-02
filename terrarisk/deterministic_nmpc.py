import numpy as np
from .dynamics import RoverModel
from .terrain import TerrainMap

class DeterministicNMPCPlanner:
    """
    Baseline Deterministic NMPC Planner evaluating dynamics under nominal (mean) terrain parameters.
    """
    def __init__(self, rover: RoverModel, terrain: TerrainMap, horizon=15, num_samples=150, dt=0.2):
        self.rover = rover
        self.terrain = terrain
        self.N = horizon
        self.M = num_samples
        self.dt = dt
        self.noise_std = np.array([0.5, 0.4])
        self.temperature = 0.1
        
    def plan(self, current_state: np.ndarray, goal_pos: np.ndarray, prev_controls: np.ndarray = None) -> tuple[np.ndarray, dict]:
        """
        Generates deterministic optimal control sequence without considering stochastic tail risk.
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
                u_t = np.clip(U[t] + delta_U[k, t], [-1.5, -1.0], [1.5, 1.0])
                state = self.rover.step_kinematics(state, u_t, self.dt)
                
                # Fetch nominal (non-stochastic) soil parameters
                soil = self.terrain.get_soil_parameters(state[0], state[1], stochastic=False)
                z, _, _ = self.rover.compute_sinkage_and_slip(state[3], soil)
                
                dist_to_goal = np.hypot(state[0] - goal_pos[0], state[1] - goal_pos[1])
                # Deterministic cost only considers nominal sinkage expectation
                traj_cost += dist_to_goal + 10.0 * z + 0.1 * (u_t[0]**2 + u_t[1]**2)
                
            costs[k] = traj_cost
            
        min_cost = np.min(costs)
        weights = np.exp(-1.0 / self.temperature * (costs - min_cost))
        weights /= (np.sum(weights) + 1e-8)
        
        U_opt = U + np.sum(weights[:, None, None] * delta_U, axis=0)
        
        return U_opt, {"expected_cost": np.mean(costs)}
