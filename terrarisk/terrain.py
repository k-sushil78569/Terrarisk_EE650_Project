import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class SoilParameters:
    """Bekker-Wong soil mechanics parameters."""
    c: float                  # Soil cohesion (kPa)
    phi_deg: float            # Internal friction angle (deg)
    kc: float                 # Cohesive modulus of deformation (kN/m^(n+1))
    kphi: float               # Frictional modulus of deformation (kN/m^(n+2))
    n: float                  # Exponent of sinkage

    def __post_init__(self):
        self.phi = np.radians(self.phi_deg)

# Baseline soil reference profiles
FIRM_SOIL = SoilParameters(c=5.0, phi_deg=35.0, kc=100.0, kphi=2000.0, n=1.1)
SOFT_REGOLITH = SoilParameters(c=0.45, phi_deg=20.0, kc=8.0, kphi=100.0, n=0.65)
DEEP_DUST = SoilParameters(c=0.20, phi_deg=17.0, kc=4.0, kphi=50.0, n=0.55)
COMPACTED_GRAVEL = SoilParameters(c=8.0, phi_deg=38.0, kc=180.0, kphi=3500.0, n=1.2)


@dataclass
class TerrainPatch:
    """Represents a discrete terramechanics hazard patch."""
    px: float
    py: float
    radius: float
    severity: float = 0.85
    ambiguous: bool = False
    name: str = "Soft Regolith"
    custom_soil: Optional[SoilParameters] = None


class TerrainMap:
    """
    2D spatial terrain grid with stochastic high-variance soft soil patches.
    Supports arbitrary dimensions, multi-patch complexity, and exteroceptive visual ambiguity.
    """
    def __init__(self, x_max=50.0, y_max=50.0, resolution=0.5, seed=42, patches=None, name="Standard Terrain"):
        self.x_max = x_max
        self.y_max = y_max
        self.resolution = resolution
        self.seed = seed
        self.name = name
        self.rng = np.random.default_rng(seed)
        
        if patches is not None:
            self.patches = patches
        else:
            # Default backward-compatible 3-patch configuration for 50x50 map
            self.patches = [
                TerrainPatch(20.0, 20.0, 5.5, severity=0.85, ambiguous=False, name="Obvious Loose Sand"),
                TerrainPatch(28.0, 28.0, 5.5, severity=0.90, ambiguous=True, name="Visually Ambiguous Soft Regolith"),
                TerrainPatch(35.0, 35.0, 5.0, severity=0.80, ambiguous=False, name="Obvious Mud/Sand")
            ]
            
    def _parse_patch(self, p) -> tuple[float, float, float, float, bool, Optional[SoilParameters]]:
        if isinstance(p, TerrainPatch):
            return p.px, p.py, p.radius, p.severity, p.ambiguous, p.custom_soil
        elif isinstance(p, dict):
            px, py = p.get('pos', (p.get('px', 0.0), p.get('py', 0.0)))
            return px, py, p['radius'], p.get('severity', 0.85), p.get('ambiguous', False), p.get('custom_soil', None)
        else:
            # Tuple: (px, py, radius, severity, [ambiguous], [custom_soil])
            px, py, r, sev = p[0], p[1], p[2], p[3]
            ambig = p[4] if len(p) > 4 else (True if (px, py) == (28.0, 28.0) else False)
            c_soil = p[5] if len(p) > 5 else None
            return px, py, r, sev, ambig, c_soil

    def get_soil_parameters(self, x: float, y: float, stochastic=True) -> SoilParameters:
        """
        Computes ground-truth physical soil parameters at position (x, y).
        If stochastic=True, samples from heavy-tailed log-normal distribution in soft zones.
        """
        weight = 0.0
        active_custom_soil = None

        for p in self.patches:
            px, py, radius, severity, _, custom_soil = self._parse_patch(p)
            dist = np.hypot(x - px, y - py)
            if dist < radius:
                w = severity * (1.0 - (dist / radius)**2)
                if w > weight:
                    weight = w
                    active_custom_soil = custom_soil
                    
        target_soft = active_custom_soil if active_custom_soil is not None else SOFT_REGOLITH

        c_nom = (1.0 - weight) * FIRM_SOIL.c + weight * target_soft.c
        phi_deg_nom = (1.0 - weight) * FIRM_SOIL.phi_deg + weight * target_soft.phi_deg
        kc_nom = (1.0 - weight) * FIRM_SOIL.kc + weight * target_soft.kc
        kphi_nom = (1.0 - weight) * FIRM_SOIL.kphi + weight * target_soft.kphi
        n_nom = (1.0 - weight) * FIRM_SOIL.n + weight * target_soft.n
        
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

    def get_visual_classification(self, x: float, y: float) -> tuple[str, SoilParameters, float]:
        """
        Simulates exteroceptive Camera / LiDAR classification at (x, y).
        If a patch is ambiguous, returns a false-safe firm prior.
        """
        for p in self.patches:
            px, py, radius, severity, ambiguous, _ = self._parse_patch(p)
            dist = np.hypot(x - px, y - py)
            if dist < radius:
                if ambiguous:
                    # Deceptive soft patch looks visually like light flat sand
                    return (
                        "Uniform Light Sand (Ambiguous)",
                        SoilParameters(c=4.5, phi_deg=33.0, kc=80.0, kphi=1500.0, n=1.0),
                        0.35
                    )
                else:
                    # Obvious rough/dark loose sand
                    return (
                        "Loose Dark Sand / Regolith",
                        SoilParameters(c=1.0, phi_deg=20.0, kc=15.0, kphi=300.0, n=0.8),
                        0.50
                    )
                    
        return "Firm Bedrock / Sand", FIRM_SOIL, 0.10


# =========================================================================
# Preset Large & Complex Map Generators (100m x 100m)
# =========================================================================

def create_dune_field_map(seed=42) -> TerrainMap:
    """
    Map 1: Planetary Dune Field (100m x 100m).
    Features large undulating sand dunes and 4 visually deceptive powdery drift sand traps
    directly blocking the direct diagonal transit route.
    """
    patches = [
        # Large obvious dune ridges
        TerrainPatch(25.0, 20.0, radius=9.0, severity=0.85, ambiguous=False, name="Dune Ridge A"),
        TerrainPatch(20.0, 45.0, radius=10.0, severity=0.85, ambiguous=False, name="Dune Ridge B"),
        TerrainPatch(45.0, 75.0, radius=11.0, severity=0.88, ambiguous=False, name="Dune Ridge C"),
        TerrainPatch(70.0, 30.0, radius=12.0, severity=0.85, ambiguous=False, name="Dune Ridge D"),
        TerrainPatch(80.0, 65.0, radius=10.0, severity=0.90, ambiguous=False, name="Dune Ridge E"),
        TerrainPatch(50.0, 15.0, radius=8.0, severity=0.82, ambiguous=False, name="Dune Ridge F"),
        TerrainPatch(15.0, 80.0, radius=9.0, severity=0.85, ambiguous=False, name="Dune Ridge G"),
        TerrainPatch(85.0, 85.0, radius=7.0, severity=0.80, ambiguous=False, name="Goal Approach Dune"),

        # DECEPTIVE 'FALSE-SAFE' DRIFT SAND TRAPS (Look like smooth flat firm sand)
        TerrainPatch(38.0, 38.0, radius=7.5, severity=0.95, ambiguous=True, name="Deceptive Drift Trap 1", custom_soil=DEEP_DUST),
        TerrainPatch(52.0, 52.0, radius=8.0, severity=0.95, ambiguous=True, name="Deceptive Drift Trap 2", custom_soil=DEEP_DUST),
        TerrainPatch(62.0, 42.0, radius=7.0, severity=0.92, ambiguous=True, name="Deceptive Drift Trap 3", custom_soil=SOFT_REGOLITH),
        TerrainPatch(40.0, 65.0, radius=7.5, severity=0.92, ambiguous=True, name="Deceptive Drift Trap 4", custom_soil=SOFT_REGOLITH),
    ]
    return TerrainMap(x_max=100.0, y_max=100.0, resolution=0.5, seed=seed, patches=patches, name="Planetary Dune Field")


def create_crater_basin_map(seed=101) -> TerrainMap:
    """
    Map 2: Cratered Regolith Basin (100m x 100m).
    Impact crater rims with deep, highly compressible dust pockets in the basin centers.
    """
    patches = [
        # Impact crater centers with fine powder
        TerrainPatch(30.0, 30.0, radius=10.0, severity=0.95, ambiguous=True, name="Crater Basin Alpha", custom_soil=DEEP_DUST),
        TerrainPatch(65.0, 70.0, radius=12.0, severity=0.95, ambiguous=True, name="Crater Basin Beta", custom_soil=DEEP_DUST),
        TerrainPatch(75.0, 25.0, radius=11.0, severity=0.88, ambiguous=False, name="Crater Basin Gamma"),
        TerrainPatch(25.0, 75.0, radius=10.0, severity=0.85, ambiguous=False, name="Crater Basin Delta"),
        TerrainPatch(50.0, 50.0, radius=8.5, severity=0.92, ambiguous=True, name="Central Ejecta Sinkhole", custom_soil=DEEP_DUST),

        # Peripheral boulder/regolith clusters
        TerrainPatch(15.0, 40.0, radius=7.0, severity=0.80, ambiguous=False),
        TerrainPatch(45.0, 20.0, radius=7.5, severity=0.82, ambiguous=False),
        TerrainPatch(55.0, 85.0, radius=8.0, severity=0.85, ambiguous=False),
        TerrainPatch(85.0, 50.0, radius=8.5, severity=0.80, ambiguous=False),
        TerrainPatch(40.0, 80.0, radius=7.0, severity=0.85, ambiguous=False),
        TerrainPatch(70.0, 45.0, radius=7.5, severity=0.90, ambiguous=True, name="Deceptive Ejecta Pocket"),
        TerrainPatch(60.0, 35.0, radius=6.5, severity=0.88, ambiguous=False),
        TerrainPatch(35.0, 60.0, radius=7.0, severity=0.90, ambiguous=True, name="Shallow Powder Trench"),
    ]
    return TerrainMap(x_max=100.0, y_max=100.0, resolution=0.5, seed=seed, patches=patches, name="Cratered Regolith Basin")


def create_canyon_chokepoint_map(seed=202) -> TerrainMap:
    """
    Map 3: Canyon Chokepoint & Labyrinth (100m x 100m).
    Flanked corridors forcing the rover through bottlenecks. The obvious direct gap
    contains an invisible loose sinkhole, requiring dynamic detection and detour into the north canyon.
    """
    patches = [
        # South canyon impassable mud/regolith barrier
        TerrainPatch(20.0, 35.0, radius=9.0, severity=0.90, ambiguous=False),
        TerrainPatch(35.0, 32.0, radius=9.0, severity=0.90, ambiguous=False),
        TerrainPatch(50.0, 30.0, radius=9.5, severity=0.90, ambiguous=False),
        TerrainPatch(65.0, 30.0, radius=9.0, severity=0.90, ambiguous=False),
        TerrainPatch(80.0, 32.0, radius=9.0, severity=0.90, ambiguous=False),

        # North canyon barrier
        TerrainPatch(20.0, 65.0, radius=9.0, severity=0.90, ambiguous=False),
        TerrainPatch(35.0, 68.0, radius=9.0, severity=0.90, ambiguous=False),
        TerrainPatch(50.0, 70.0, radius=9.5, severity=0.90, ambiguous=False),
        TerrainPatch(65.0, 70.0, radius=9.0, severity=0.90, ambiguous=False),
        TerrainPatch(80.0, 68.0, radius=9.0, severity=0.90, ambiguous=False),

        # The central chokepoint tempting straight passage:
        TerrainPatch(48.0, 50.0, radius=7.5, severity=0.96, ambiguous=True, name="Chokepoint False-Safe Trap", custom_soil=DEEP_DUST),
        TerrainPatch(60.0, 52.0, radius=7.0, severity=0.94, ambiguous=True, name="Exit False-Safe Trap", custom_soil=DEEP_DUST),
        
        # Intermediate patches
        TerrainPatch(15.0, 15.0, radius=7.0, severity=0.80, ambiguous=False),
        TerrainPatch(85.0, 85.0, radius=7.0, severity=0.80, ambiguous=False),
    ]
    return TerrainMap(x_max=100.0, y_max=100.0, resolution=0.5, seed=seed, patches=patches, name="Canyon Chokepoint Labyrinth")


def create_procedural_complex_map(seed=555, num_patches=18) -> TerrainMap:
    """
    Map 4: Procedural High-Complexity Chaos (100m x 100m).
    Fully stochastic arrangement of 18 patches with varying radius, severity, and 50% visual ambiguity.
    """
    rng = np.random.default_rng(seed)
    patches = []
    for i in range(num_patches):
        px = rng.uniform(15.0, 85.0)
        py = rng.uniform(15.0, 85.0)
        radius = rng.uniform(5.0, 10.5)
        severity = rng.uniform(0.78, 0.96)
        ambiguous = bool(rng.choice([True, False], p=[0.55, 0.45]))
        soil = DEEP_DUST if (ambiguous and severity > 0.90) else SOFT_REGOLITH
        patches.append(TerrainPatch(
            px=px, py=py, radius=radius, severity=severity, ambiguous=ambiguous,
            name=f"Patch_{i+1}_{'Ambiguous' if ambiguous else 'Obvious'}",
            custom_soil=soil
        ))
    return TerrainMap(x_max=100.0, y_max=100.0, resolution=0.5, seed=seed, patches=patches, name=f"Procedural Chaos (Seed {seed})")
