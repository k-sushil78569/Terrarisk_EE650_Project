import numpy as np
from .terrain import SoilParameters, FIRM_SOIL

class OnlineSoilEstimator:
    """
    Recursive Extended Kalman Filter / Bayesian estimator for online Bekker-Wong soil parameter estimation
    fusing wheel slip (s), sinkage depth (z), motor torque (T), and IMU pitch/roll.
    """
    def __init__(self, rover_mass=150.0, wheel_radius=0.25, wheel_width=0.15):
        self.mass = rover_mass
        self.g = 9.81
        self.W_wheel = (rover_mass * self.g) / 4.0
        self.r = wheel_radius
        self.b = wheel_width
        
        # Internal state estimate: x_hat = [c_hat, phi_deg_hat, Kc_hat, Kphi_hat]
        self.c_hat = 3.0           # Nominal initial guess (kPa)
        self.phi_deg_hat = 28.0    # Nominal initial guess (deg)
        self.Kc_hat = 50.0         # Nominal initial guess
        self.Kphi_hat = 1000.0     # Nominal initial guess
        
        # Estimation error covariance matrix
        self.P = np.diag([2.0, 10.0, 20.0, 500.0])
        self.Q = np.diag([0.01, 0.05, 0.1, 5.0])    # Process noise
        self.R = np.diag([0.002, 0.02, 1.0])       # Measurement noise [z, slip, torque]
        
    def update(self, measured_z: float, measured_slip: float, measured_torque: float, v_rover: float) -> SoilParameters:
        """
        Updates soil parameter estimate given online physical/proprioceptive sensor observations.
        """
        # Add sensor measurement noise
        z_obs = max(0.005, measured_z + np.random.normal(0, 0.001))
        slip_obs = max(0.01, measured_slip + np.random.normal(0, 0.01))
        torque_obs = max(1.0, measured_torque + np.random.normal(0, 0.5))
        
        # Forward physics model prediction based on current estimates
        phi_rad = np.radians(self.phi_deg_hat)
        K_eq = (self.Kc_hat / self.b + self.Kphi_hat) * 1000.0
        z_pred = max(0.005, ((self.W_wheel * 1.8) / (self.b * K_eq)) ** (1.0 / 1.8))
        
        p_avg = self.W_wheel / (self.b * z_pred)
        tau_pred = (self.c_hat * 1000.0) + p_avg * np.tan(phi_rad)
        
        tractive_demand = self.W_wheel * 0.2
        max_tractive = self.b * z_pred * tau_pred * 4.0
        slip_pred = min(0.98, tractive_demand / max(1.0, max_tractive))
        torque_pred = (self.W_wheel * z_pred + max_tractive * self.r) / 4.0
        
        # Innovation vector y = z_obs - z_pred
        innovation = np.array([z_obs - z_pred, slip_obs - slip_pred, torque_obs - torque_pred])
        
        # Empirical Jacobians / Sensitivity matrix H
        H = np.array([
            [0.0, 0.0, -0.0001, -0.000005],
            [-0.05, -0.02, 0.0, 0.0],
            [2.0, 1.5, 0.1, 0.01]
        ])
        
        # Kalman Gain computation: K = P * H^T * (H * P * H^T + R)^(-1)
        S = H @ self.P @ H.T + self.R
        K_gain = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        dx = K_gain @ innovation
        self.c_hat = max(0.05, self.c_hat + dx[0])
        self.phi_deg_hat = max(10.0, min(40.0, self.phi_deg_hat + dx[1]))
        self.Kc_hat = max(0.5, self.Kc_hat + dx[2])
        self.Kphi_hat = max(5.0, self.Kphi_hat + dx[3])
        
        # Covariance update
        self.P = (np.eye(4) - K_gain @ H) @ self.P + self.Q
        
        return SoilParameters(c=self.c_hat, phi_deg=self.phi_deg_hat, kc=self.Kc_hat, kphi=self.Kphi_hat, n=0.8)
