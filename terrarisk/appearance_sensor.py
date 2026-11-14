import numpy as np
from .terrain import SoilParameters, FIRM_SOIL, SOFT_REGOLITH

class AppearanceSensor:
    """
    Simulates exteroceptive LiDAR and Camera terrain sensing.
    Maps visual appearance (color, elevation, surface texture) to coarse terrain classes
    with prior probability bounds (reflecting the appearance-only limitation).
    """
    def __init__(self, rng_seed=42, terrain=None):
        self.rng = np.random.default_rng(rng_seed)
        self.terrain = terrain
        
    def classify_terrain(self, x: float, y: float, ground_truth_soil: SoilParameters = None, terrain=None) -> tuple[str, SoilParameters, float]:
        """
        Returns (visual_class, expected_prior_soil, variance_uncertainty).
        LiDAR/Camera cannot see subterranean cohesion; visually identical patches look the same.
        """
        active_terrain = terrain if terrain is not None else self.terrain
        if active_terrain is not None and hasattr(active_terrain, "get_visual_classification"):
            return active_terrain.get_visual_classification(x, y)

        # Fallback to legacy static coordinates for backwards compatibility with legacy tests
        dist_obvious = np.hypot(x - 22.0, y - 22.0)
        dist_ambiguous = np.hypot(x - 28.0, y - 28.0)
        
        if dist_obvious < 6.0:
            visual_class = "Loose Dark Sand"
            prior_soil = SoilParameters(c=1.0, phi_deg=20.0, kc=15.0, kphi=300.0, n=0.8)
            uncertainty = 0.5
        elif dist_ambiguous < 6.0:
            visual_class = "Uniform Light Sand (Ambiguous)"
            prior_soil = SoilParameters(c=4.5, phi_deg=33.0, kc=80.0, kphi=1500.0, n=1.0)
            uncertainty = 0.35
        else:
            visual_class = "Firm Bedrock/Sand"
            prior_soil = FIRM_SOIL
            uncertainty = 0.1
            
        return visual_class, prior_soil, uncertainty
