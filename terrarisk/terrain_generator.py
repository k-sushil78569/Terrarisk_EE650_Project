import numpy as np
from dataclasses import dataclass

@dataclass
class SoilParameters:
    """Bekker-Wong soil mechanics parameters."""
    c: float        # Soil cohesion (kPa)
    phi_deg: float  # Internal friction angle (deg)
    kc: float       # Cohesive deformation modulus (kN/m^(n+1))
    kphi: float     # Frictional deformation modulus (kN/m^(n+2))
    n: float        # Exponent of sinkage
    
    @property
    def phi_rad(self) -> float:
        return np.radians(self.phi_deg)

# Baseline soil reference profiles
FIRM_SOIL = SoilParameters(c=5.0, phi_deg=35.0, kc=100.0, kphi=2000.0, n=1.1)
SOFT_REGOLITH = SoilParameters(c=0.45, phi_deg=20.0, kc=8.0, kphi=100.0, n=0.65)

class TerrainMap:
    """
    Synthesizes continuous terramechanics properties, visual risk maps,
    and geometric elevation fields for Monte Carlo navigation benchmarks.
    """
    def __init__(self, x_max=50.0, y_max=50.0, resolution=0.5, num_patches=4, seed=42):
        self.x_max = x_max
        self.y_max = y_max
        self.resolution = resolution
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.nx = int(x_max / resolution) + 1
        self.ny = int(y_max / resolution) + 1
        self.x_coords = np.linspace(0, x_max, self.nx)
        self.y_coords = np.linspace(0, y_max, self.ny)
        
        # Procedural patch generation with random coordinates & severity
        self.patches = []
        for _ in range(num_patches):
            px = self.rng.uniform(10.0, x_max - 10.0)
            py = self.rng.uniform(10.0, y_max - 10.0)
            radius = self.rng.uniform(4.0, 7.0)
            severity = self.rng.uniform(0.7, 0.95)
            # High probability of being visually deceptive (Stage 2 vs Stage 3 gap)
            is_visually_ambiguous = self.rng.choice([True, False], p=[0.6, 0.4])
            self.patches.append({
                'pos': (px, py),
                'radius': radius,
                'severity': severity,
                'ambiguous': is_visually_ambiguous
            })

    def get_ground_truth_soil(self, x: float, y: float, stochastic=True) -> SoilParameters:
        """Computes true physical soil parameters at (x, y)."""
        weight = 0.0
        for p in self.patches:
            px, py = p['pos']
            dist = np.hypot(x - px, y - py)
            if dist < p['radius']:
                w = p['severity'] * (1.0 - (dist / p['radius'])**2)
                weight = max(weight, w)
                
        c_nom = (1.0 - weight) * FIRM_SOIL.c + weight * SOFT_REGOLITH.c
        phi_nom = (1.0 - weight) * FIRM_SOIL.phi_deg + weight * SOFT_REGOLITH.phi_deg
        kc_nom = (1.0 - weight) * FIRM_SOIL.kc + weight * SOFT_REGOLITH.kc
        kphi_nom = (1.0 - weight) * FIRM_SOIL.kphi + weight * SOFT_REGOLITH.kphi
        n_nom = (1.0 - weight) * FIRM_SOIL.n + weight * SOFT_REGOLITH.n
        
        if stochastic and weight > 0.05:
            scale = 0.45 * weight
            c_val = max(0.08, c_nom * self.rng.lognormal(mean=0.0, sigma=scale))
            kc_val = max(0.8, kc_nom * self.rng.lognormal(mean=0.0, sigma=scale))
            kphi_val = max(8.0, kphi_nom * self.rng.lognormal(mean=0.0, sigma=scale))
        else:
            c_val, kc_val, kphi_val = c_nom, kc_nom, kphi_nom
            
        return SoilParameters(c=c_val, phi_deg=phi_nom, kc=kc_val, kphi=kphi_val, n=n_nom)

    def get_visual_appearance_risk(self, x: float, y: float) -> float:
        """
        Returns perceived risk score [0, 1] from simulated LiDAR/Camera (Stage 2 input).
        Deceptive patches report artificially low risk visually.
        """
        vis_risk = 0.0
        for p in self.patches:
            px, py = p['pos']
            dist = np.hypot(x - px, y - py)
            if dist < p['radius']:
                w = p['severity'] * (1.0 - (dist / p['radius'])**2)
                # Visually ambiguous patches suppress visual risk score
                perceived_w = w * 0.15 if p['ambiguous'] else w
                vis_risk = max(vis_risk, perceived_w)
        return float(np.clip(vis_risk, 0.0, 1.0))

    def generate_grid_maps(self):
        """Pre-computes 2D spatial matrices for fast plotting & sampling."""
        XX, YY = np.meshgrid(self.x_coords, self.y_coords)
        c_map = np.zeros_like(XX)
        vis_map = np.zeros_like(XX)
        
        for i in range(self.ny):
            for j in range(self.nx):
                x, y = XX[i, j], YY[i, j]
                soil = self.get_ground_truth_soil(x, y, stochastic=False)
                c_map[i, j] = soil.c
                vis_map[i, j] = self.get_visual_appearance_risk(x, y)
                
        return XX, YY, c_map, vis_map