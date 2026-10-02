import numpy as np
from .terrain import SoilParameters, FIRM_SOIL, SOFT_REGOLITH

class AppearanceSensor:
    """
    Simulates exteroceptive LiDAR and Camera terrain sensing.
    Maps visual appearance (color, elevation, surface texture) to coarse terrain classes
    with wide prior probability bounds (reflecting the appearance-only limitation).
    """
    def __init__(self, rng_seed=42):
        self.rng = np.random.default_rng(rng_seed)
        
    def classify_terrain(self, x: float, y: float, ground_truth_soil: SoilParameters) -> tuple[str, SoilParameters, float]:
        """
        Returns (visual_class, expected_prior_soil, variance_uncertainty).
        LiDAR/Camera cannot see subterranean cohesion; visually identical patches look the same.
        """
        # Distinguish visually obvious vs visually ambiguous patches
        # Soft regolith patch near (22, 22) is visually obvious (darker/rougher)
        # Soft regolith patch near (28, 28) is VISUALLY AMBIGUOUS (looks identical to firm sand)
        
        dist_obvious = np.hypot(x - 22.0, y - 22.0)
        dist_ambiguous = np.hypot(x - 28.0, y - 28.0)
        
        if dist_obvious < 6.0:
            # Visually obvious loose sand/mud
            visual_class = "Loose Dark Sand"
            # High prior uncertainty on soil cohesion
            prior_soil = SoilParameters(c=1.0, phi_deg=20.0, kc=15.0, kphi=300.0, n=0.8)
            uncertainty = 0.5
        elif dist_ambiguous < 6.0:
            # VISUALLY AMBIGUOUS: Looks like Firm Sand, but mechanically soft!
            visual_class = "Uniform Light Sand (Ambiguous)"
            # Visual sensor misclassifies or assumes nominal firm parameters with moderate variance
            prior_soil = SoilParameters(c=4.5, phi_deg=33.0, kc=80.0, kphi=1500.0, n=1.0)
            uncertainty = 0.35
        else:
            visual_class = "Firm Bedrock/Sand"
            prior_soil = FIRM_SOIL
            uncertainty = 0.1
            
        return visual_class, prior_soil, uncertainty
