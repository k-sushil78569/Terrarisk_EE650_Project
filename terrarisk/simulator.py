import numpy as np
from .dynamics import RoverModel
from .terrain import TerrainMap

class Simulator:
    """
    Simulation environment for executing navigation trials and gathering performance metrics.
    """
    def __init__(self, terrain: TerrainMap, rover: RoverModel, dt=0.2):
        self.terrain = terrain
        self.rover = rover
        self.dt = dt
        
    def run_trial(self, planner, start_state: np.ndarray, goal_pos: np.ndarray, max_steps=300) -> dict:
        """
        Executes a single navigation trial.
        """
        state = start_state.copy()
        trajectory = [state.copy()]
        sinkages = []
        slips = []
        controls = []
        
        entrapped = False
        reached_goal = False
        prev_u = None
        
        for step in range(max_steps):
            # Plan next control action
            U_opt, info = planner.plan(state, goal_pos, prev_controls=prev_u)
            action = U_opt[0]
            controls.append(action)
            prev_u = U_opt
            
            # Step physical kinematics
            state = self.rover.step_kinematics(state, action, self.dt)
            trajectory.append(state.copy())
            
            # Interact with actual stochastic ground truth terrain
            soil = self.terrain.get_soil_parameters(state[0], state[1], stochastic=True)
            z, slip, _ = self.rover.compute_sinkage_and_slip(state[3], soil)
            sinkages.append(z)
            slips.append(slip)
            
            # Check entrapment condition
            if self.rover.is_entrapped(z, slip):
                entrapped = True
                break
                
            # Check goal condition
            dist_to_goal = np.hypot(state[0] - goal_pos[0], state[1] - goal_pos[1])
            if dist_to_goal < 1.0:
                reached_goal = True
                break
                
        traj_arr = np.array(trajectory)
        path_length = np.sum(np.hypot(np.diff(traj_arr[:, 0]), np.diff(traj_arr[:, 1]))) if len(traj_arr) > 1 else 0.0
        
        return {
            "trajectory": traj_arr,
            "sinkages": np.array(sinkages),
            "slips": np.array(slips),
            "controls": np.array(controls),
            "entrapped": entrapped,
            "reached_goal": reached_goal,
            "steps": len(sinkages),
            "path_length": path_length,
            "max_sinkage": np.max(sinkages) if len(sinkages) > 0 else 0.0,
            "mean_sinkage": np.mean(sinkages) if len(sinkages) > 0 else 0.0
        }
