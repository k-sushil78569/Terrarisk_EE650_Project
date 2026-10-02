import numpy as np

class SoilParameters:
    """Bekker-Wong soil mechanics parameters."""
    def __init__(self, c: float, phi_deg: float, kc: float, kphi: float, n: float):
        self.c = c                  # Soil cohesion (kPa)
        self.phi = np.radians(phi_deg) # Internal friction angle (rad)
        self.kc = kc                # Cohesive modulus of deformation (kN/m^(n+1))
        self.kphi = kphi            # Frictional modulus of deformation (kN/m^(n+2))
        self.n = n                  # Exponent of sinkage

FIRM_SOIL = SoilParameters(c=5.0, phi_deg=35.0, kc=100.0, kphi=2000.0, n=1.1)
SOFT_REGOLITH = SoilParameters(c=0.45, phi_deg=20.0, kc=8.0, kphi=100.0, n=0.65)

class TerrainMap:
    """
    2D spatial terrain grid with stochastic high-variance soft soil patches.
    """
    def __init__(self, x_max=50.0, y_max=50.0, resolution=0.5, seed=42):
        self.x_max = x_max
        self.y_max = y_max
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)
        
        # Soft patch centers directly blocking the nominal path
        self.patches = [
            (20.0, 20.0, 5.5, 0.85),
            (28.0, 28.0, 5.5, 0.90), # Visually Ambiguous Patch
            (35.0, 35.0, 5.0, 0.80)
        ]
        
    def get_soil_parameters(self, x: float, y: float, stochastic=True) -> SoilParameters:
        """
        Computes soil parameters at position (x, y).
        If stochastic=True, samples from heavy-tailed distribution in soft zones.
        """
        weight = 0.0
        for px, py, radius, severity in self.patches:
            dist = np.hypot(x - px, y - py)
            if dist < radius:
                w = severity * (1.0 - (dist / radius)**2)
                if w > weight:
                    weight = w
                    
        c_nom = (1.0 - weight) * FIRM_SOIL.c + weight * SOFT_REGOLITH.c
        phi_deg_nom = (1.0 - weight) * 35.0 + weight * 20.0
        kc_nom = (1.0 - weight) * FIRM_SOIL.kc + weight * SOFT_REGOLITH.kc
        kphi_nom = (1.0 - weight) * FIRM_SOIL.kphi + weight * SOFT_REGOLITH.kphi
        n_nom = (1.0 - weight) * FIRM_SOIL.n + weight * SOFT_REGOLITH.n
        
        if stochastic and weight > 0.05:
            scale = 0.45 * weight
            c_val = max(0.08, c_nom * self.rng.lognormal(mean=0.0, sigma=scale))
            kc_val = max(0.8, kc_nom * self.rng.lognormal(mean=0.0, sigma=scale))
            kphi_val = max(8.0, kphi_nom * self.rng.lognormal(mean=0.0, sigma=scale))
        else:
            c_val = c_nom
            kc_val = kc_nom
            kphi_val = kphi_nom
            
        return SoilParameters(c=c_val, phi_deg=phi_deg_nom, kc=kc_val, kphi=kphi_val, n=n_nom)
