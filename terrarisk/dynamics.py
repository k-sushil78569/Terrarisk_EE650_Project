import numpy as np
from .terrain import SoilParameters

class RoverModel:
    """
    2D Kinematic Rover Model coupled with Bekker-Wong slip-sinkage mechanics.
    """
    def __init__(self, mass=150.0, wheel_radius=0.25, wheel_width=0.15, num_wheels=4):
        self.mass = mass                # Total mass (kg)
        self.g = 9.81                   # Gravity (m/s^2)
        self.weight = mass * self.g     # Weight (N)
        self.r = wheel_radius           # Wheel radius (m)
        self.b = wheel_width            # Wheel width (m)
        self.N_wheels = num_wheels      # Number of wheels
        
        # Load per wheel (N)
        self.W_wheel = self.weight / self.N_wheels
        
        # Critical thresholds
        self.z_crit = 0.08              # Critical sinkage depth for entrapment (m) = 8 cm
        self.slip_crit = 0.65           # Critical wheel slip ratio
        
    def step_kinematics(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """
        Updates 2D pose [x, y, theta, v].
        action = [a, omega] (linear acceleration, angular velocity).
        """
        x, y, theta, v = state
        a, omega = action
        
        # Update velocities and positions
        v_next = max(0.0, v + a * dt)
        theta_next = (theta + omega * dt + np.pi) % (2 * np.pi) - np.pi
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        
        return np.array([x_next, y_next, theta_next, v_next])

    def compute_sinkage_and_slip(self, v: float, soil: SoilParameters) -> tuple[float, float, float]:
        """
        Computes Bekker-Wong sinkage depth z (m), wheel slip ratio s, and drive torque T (Nm).
        """
        K_eq = (soil.kc / self.b + soil.kphi) * 1000.0 # Convert kN to N
        
        # Sinkage depth z (m)
        z = ((self.W_wheel * (soil.n + 1.0)) / (self.b * K_eq)) ** (1.0 / (soil.n + 1.0))
        
        p_avg = self.W_wheel / (self.b * max(0.005, z))
        tau = (soil.c * 1000.0) + p_avg * np.tan(soil.phi)
        
        tractive_demand = self.W_wheel * 0.2
        max_tractive_force = self.b * max(0.005, z) * tau * self.N_wheels
        
        slip_ratio = min(0.98, tractive_demand / max(1.0, max_tractive_force))
        torque = (self.W_wheel * z + max_tractive_force * self.r) / self.N_wheels
        
        return z, slip_ratio, torque

    def is_entrapped(self, sinkage: float, slip: float) -> bool:
        """Determines if the rover is in an unrecoverable entrapped state."""
        return (sinkage >= self.z_crit) or (slip >= self.slip_crit)
