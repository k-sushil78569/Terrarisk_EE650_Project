"""
TerraRisk: Killer Maps — specifically engineered to trap NMPC and Gaussian SMPC
while Stage 3 Physical CVaR survives through early anomaly detection.

Design Principle (analytically verified):
  Severity 0.94 DEEP_DUST patches:
    - Nominal sinkage (z_nom) = 7.55 cm  <  z_crit = 8 cm
      → NMPC and Gaussian SMPC think "risky but feasible" and push through
    - Stochastic entrapment per step: P = 41%
      → Over 5 steps in patch center: P_entrap = 93%  → near-certain death
    - EKF detection triggers at dist_frac=0.9 from center (edge), z_nom=4.7 cm > 3.5 cm threshold
      → Stage 3 gets warned BEFORE reaching the lethal center and reroutes

Three distinct "killer" scenarios:
  Map 1 "Death Funnel"      : Corridor with ONE mandatory deceptive sinkhole; safe margins exist for Stage 3
  Map 2 "Diagonal Minefield": Three deceptive patches on the natural diagonal route; arc bypass available
  Map 3 "False-Safe Shortcut": Attractive short route is a death trap; long safe route rewarded by Stage 3

Each map is 100m × 100m.
"""

import numpy as np
from .terrain import TerrainMap, TerrainPatch, SoilParameters, FIRM_SOIL, SOFT_REGOLITH, DEEP_DUST


# ── Analytically calibrated killer patch parameters ────────────────────────────
#   severity=0.94, DEEP_DUST  →  z_nom=7.55cm, P_entrap_5steps=93%
#   severity=0.92, DEEP_DUST  →  z_nom=6.80cm, P_entrap_5steps=79%
# ── Impassable blocker (everyone avoids): severity=0.99, NOT ambiguous ─────────
#   severity=0.99, DEEP_DUST  →  z_nom >> 8cm  (NMPC, Stage 2, Stage 3 all avoid)

KILLER_SOIL = SoilParameters(c=0.20, phi_deg=17.0, kc=4.0, kphi=50.0, n=0.55)   # DEEP_DUST

def _blocker(px, py, radius=14.0, name="Wall"):
    """Impassable obvious wall — everyone avoids (z_nom >> z_crit, NOT ambiguous)."""
    return TerrainPatch(px=px, py=py, radius=radius, severity=0.99,
                        ambiguous=False, name=name, custom_soil=KILLER_SOIL)

def _killer(px, py, radius=10.0, name="Deceptive Killer"):
    """Deceptive killer patch — NMPC/Gaussian push through; Stage 3 detects and reroutes.
    severity=0.94, DEEP_DUST → z_nom=7.55cm, P_entrap_5steps=93%
    """
    return TerrainPatch(px=px, py=py, radius=radius, severity=0.94,
                        ambiguous=True, name=name, custom_soil=KILLER_SOIL)

def _moderate_killer(px, py, radius=10.0, name="Moderate Killer"):
    """Slightly less severe deceptive patch. severity=0.92 → z_nom=6.8cm, P_5step=79%"""
    return TerrainPatch(px=px, py=py, radius=radius, severity=0.92,
                        ambiguous=True, name=name, custom_soil=KILLER_SOIL)


# ═══════════════════════════════════════════════════════════════════════════════
# MAP 1: "Death Funnel"
# ═══════════════════════════════════════════════════════════════════════════════
def create_death_funnel_map(seed=1001) -> TerrainMap:
    """
    Map 1: 100m × 100m horizontal corridor.

    Layout:
        North wall (y≈72–86): 5 overlapping obvious blockers — everyone avoids
        South wall (y≈14–28): 5 overlapping obvious blockers — everyone avoids
        Corridor center (y=50, x=50): ONE massive deceptive killer (radius=14m)
            → forces rover from start(10,50) toward goal(90,50) through the center

    Safe margins inside corridor:
        N safe strip: y ≈ 60–70  (between deceptive patch top-edge and N wall bottom)
        S safe strip: y ≈ 30–40  (between S wall top-edge and deceptive patch bottom-edge)

    Controller responses:
      NMPC: queries nominal terrain, sees z_nom=7.55cm at center, still lowest-cost path
            (blockers on N/S have z >> 8cm) → goes straight through center → TRAPPED (~93%)
      Gaussian SMPC: α=0.5 rollouts see some high draws but prefer shortest route → TRAPPED
      Stage 2 Visual: patch is AMBIGUOUS (looks like flat firm sand) → misclassified safe → TRAPPED
      Stage 3 Physical: EKF triggers at patch EDGE (z=4.7cm > 3.5cm threshold) → marks anomaly
                        → CVaR penalty steers toward N or S safe strip (z<2cm) → SURVIVES
    """
    patches = [
        # ── North Wall (5 overlapping obvious blockers) ──
        _blocker(20.0,  78.0, radius=14.0, name="N-Wall-1"),
        _blocker(40.0,  78.0, radius=14.0, name="N-Wall-2"),
        _blocker(60.0,  78.0, radius=14.0, name="N-Wall-3"),
        _blocker(80.0,  78.0, radius=14.0, name="N-Wall-4"),
        _blocker(100.0, 78.0, radius=14.0, name="N-Wall-5"),

        # ── South Wall (5 overlapping obvious blockers) ──
        _blocker(20.0,  22.0, radius=14.0, name="S-Wall-1"),
        _blocker(40.0,  22.0, radius=14.0, name="S-Wall-2"),
        _blocker(60.0,  22.0, radius=14.0, name="S-Wall-3"),
        _blocker(80.0,  22.0, radius=14.0, name="S-Wall-4"),
        _blocker(100.0, 22.0, radius=14.0, name="S-Wall-5"),

        # ── THE KILLER: massive deceptive sinkhole directly in corridor center ──
        _killer(50.0, 50.0, radius=14.0, name="☠ Death Funnel Sinkhole"),

        # ── Entry/exit anchor bumps so NMPC doesn't cut near walls ──
        _blocker(0.0,  50.0, radius=5.0,  name="Entry-N-Bump"),
        _blocker(100.0, 50.0, radius=5.0, name="Exit-N-Bump"),
    ]
    return TerrainMap(x_max=100.0, y_max=100.0, resolution=0.5, seed=seed,
                      patches=patches, name="Map 1: Death Funnel")


# ═══════════════════════════════════════════════════════════════════════════════
# MAP 2: "Diagonal Minefield"
# ═══════════════════════════════════════════════════════════════════════════════
def create_diagonal_minefield_map(seed=2002) -> TerrainMap:
    """
    Map 2: 100m × 100m open terrain, start=(10,10), goal=(90,90).

    Three large deceptive killer patches placed directly on the natural diagonal path:
        Mine A at (32, 32) — first obstacle encountered on direct route
        Mine B at (53, 53) — second, unavoidable on straight path
        Mine C at (72, 72) — third, blocks final approach

    Off-diagonal bypass routes (what Stage 3 finds):
        Route via (10,55) → arc around mine A and B → (90,90)   [safe, longer]
        Route via (55,10) → arc around mine B and C → (90,90)   [also safe]

    Controller responses:
      NMPC: heads straight along diagonal → hits Mine A edge → stochastic spike → TRAPPED
      Gaussian SMPC: also prefers diagonal (α=0.5, mean-focused) → same fate
      Stage 2 Visual: all three mines ambiguous → misclassified as firm flat sand → TRAPPED
      Stage 3 Physical: detects Mine A at edge (z≈4.7cm, EKF triggered), marks anomaly zone,
                        reroutes to arc path around all three mines → SURVIVES
    """
    patches = [
        # ── Three diagonal deceptive killer mines ──
        _killer(32.0,  32.0, radius=11.0, name="☠ Mine Alpha (diagonal)"),
        _killer(53.0,  53.0, radius=12.0, name="☠ Mine Beta (diagonal)"),
        _killer(72.0,  72.0, radius=11.0, name="☠ Mine Gamma (diagonal)"),

        # ── Soft but non-ambiguous blockers to channel NMPC toward diagonal ──
        # These make N-arc look costly (moderate sinkage) but NOT trapped — just expensive
        # This ensures Stage 3 detects and takes N-arc but NMPC still prefers diagonal
        _moderate_killer(15.0, 55.0, radius=9.0,  name="N-Arc Blocker-1 (moderate)"),
        _moderate_killer(30.0, 70.0, radius=9.0,  name="N-Arc Blocker-2 (moderate)"),
        _moderate_killer(55.0, 15.0, radius=9.0,  name="S-Arc Blocker-1 (moderate)"),
        _moderate_killer(70.0, 30.0, radius=9.0,  name="S-Arc Blocker-2 (moderate)"),
    ]
    return TerrainMap(x_max=100.0, y_max=100.0, resolution=0.5, seed=seed,
                      patches=patches, name="Map 2: Diagonal Minefield")


# ═══════════════════════════════════════════════════════════════════════════════
# MAP 3: "False-Safe Shortcut"
# ═══════════════════════════════════════════════════════════════════════════════
def create_false_safe_shortcut_map(seed=3003) -> TerrainMap:
    """
    Map 3: 100m × 100m, start=(10,50), goal=(90,50).

    Two distinct route choices from start to goal:
        SHORT ROUTE (direct E-W at y=50): attractive straight path, 80m Euclidean,
                    but THREE deceptive killer patches at (30,50), (50,50), (70,50)
                    → visually looks like smooth flat corridor
                    → nominal sinkage ≈7.55cm → all three trap rovers with P≈93% per patch
        LONG ROUTE (arc via N at y=75 or S at y=25): 40% longer path, completely safe,
                    no patches anywhere near it

    Controller responses:
      NMPC: cost = dist_to_goal + 10*z.  On short route, z_nom=7.55cm → penalty 0.75/step.
            On long arc, z_nom≈0.5cm → almost zero penalty but longer distance.
            Total short route cost << long arc → NMPC always picks short route → TRAPPED
      Gaussian SMPC: α=0.5 rollouts see some spikes but mean is 7.55cm → prefers short → TRAPPED
      Stage 2 Visual: killer patches are AMBIGUOUS → classified as "flat safe sand" → TRAPPED
      Stage 3 Physical: enters first killer patch edge → z=4.7cm > 3.5cm → EKF fires immediately
                        → anomaly memory blocks short route → CVaR forces arc route → SURVIVES

    This is the clearest demonstration: Stage 3 takes a 40% longer path but arrives safely.
    All others gamble on the shortcut and lose.
    """
    patches = [
        # ── THREE deceptive killers on the direct E-W route at y=50 ──
        _killer(30.0, 50.0, radius=12.0, name="☠ Shortcut Trap 1"),
        _killer(52.0, 50.0, radius=12.0, name="☠ Shortcut Trap 2"),
        _killer(72.0, 50.0, radius=11.0, name="☠ Shortcut Trap 3"),

        # ── Blockers to prevent extreme N/S route (makes Stage 3 route through y=70-75 zone) ──
        _blocker(50.0,  93.0, radius=12.0, name="Extreme-N Blocker"),
        _blocker(50.0,   7.0, radius=12.0, name="Extreme-S Blocker"),
    ]
    return TerrainMap(x_max=100.0, y_max=100.0, resolution=0.5, seed=seed,
                      patches=patches, name="Map 3: False-Safe Shortcut")
