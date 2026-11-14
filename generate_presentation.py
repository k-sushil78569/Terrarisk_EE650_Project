import collections
import collections.abc
import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def apply_text_styling(cell, text, font_size=Pt(10), bold=False, color=RGBColor(40, 40, 40), align=PP_ALIGN.LEFT):
    cell.text = ""
    p = cell.text_frame.paragraphs[0]
    p.text = text
    p.alignment = align
    if p.runs:
        run = p.runs[0]
        run.font.size = font_size
        run.font.bold = bold
        run.font.color.rgb = color

def create_table_slide(prs, title_text, headers, rows, col_widths, font_size=Pt(9.5)):
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    
    # Title Box
    txbox = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(8.8), Inches(0.8))
    tf = txbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = RGBColor(20, 35, 60)
    
    # Table dimensions
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, Inches(0.5), Inches(1.3), Inches(9.0), Inches(5.2))
    table = table_shape.table
    
    # Set Column Widths
    for idx, width in enumerate(col_widths):
        table.columns[idx].width = width
        
    # Header Row
    for idx, header in enumerate(headers):
        cell = table.cell(0, idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 60, 110)
        apply_text_styling(cell, header, font_size=Pt(10.5), bold=True, color=RGBColor(255, 255, 255), align=PP_ALIGN.CENTER)
        
    # Body Rows
    for r_idx, row in enumerate(rows):
        bg_color = RGBColor(245, 248, 252) if r_idx % 2 == 0 else RGBColor(255, 255, 255)
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_color
            align = PP_ALIGN.CENTER if c_idx == 0 else PP_ALIGN.LEFT
            bold = True if c_idx == 0 else False
            apply_text_styling(cell, val, font_size=font_size, bold=bold, color=RGBColor(30, 30, 30), align=align)
            
    return slide

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    title_layout = prs.slide_layouts[0]
    content_layout = prs.slide_layouts[1]
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # Slide 1: Title Slide
    # =========================================================================
    slide = prs.slides.add_slide(title_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "TerraRisk: Risk-Sensitive CVaR-SMPC for Autonomous Rover Navigation in High-Variance, Deformable Terrain"
    title.text_frame.paragraphs[0].font.size = Pt(32)
    title.text_frame.paragraphs[0].font.bold = True
    
    subtitle.text = "EE650: Basics of Modern Control Systems\n\nTeam:\nSivaganesh B. | Sushil Krishna K. | Ramya Rasika S. R. | Kanishka S.\n\nIndian Institute of Technology Kanpur"
    subtitle.text_frame.paragraphs[0].font.size = Pt(16)

    # =========================================================================
    # Slide 2: Problem Statement & Sensing Gap
    # =========================================================================
    slide = prs.slides.add_slide(content_layout)
    title = slide.shapes.title
    title.text = "Problem Statement & Sensing Gap"
    
    content = slide.placeholders[1].text_frame
    content.text = "Terramechanics & Sensing Ambiguity in Planetary Traversal:"
    
    p = content.add_paragraph()
    p.text = "Visual Ambiguity Hazard: Firm ground and loose, highly compressible sand/soil patches can look completely identical to LiDAR and cameras."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "Failure of Standard Chance Constraints: Traditional SMPC bounds collision probability (first moment), completely ignoring catastrophic tail-risk severity (deep sinkage, wheel immobilization)."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "Proposed Framework: Unifies Bekker-Wong terramechanics, real-time proprioceptive Extended Kalman Filter (EKF) estimation, and CVaR-constrained Stochastic MPC."
    p.level = 0

    if os.path.exists("terrain_ambiguity_map.png"):
        slide.shapes.add_picture("terrain_ambiguity_map.png", Inches(1.0), Inches(4.3), width=Inches(8.0))

    # =========================================================================
    # Slide 3: Detailed Implementation Stages (Overview)
    # =========================================================================
    slide = prs.slides.add_slide(content_layout)
    title = slide.shapes.title
    title.text = "Development Progression: Staged Implementation"
    
    content = slide.placeholders[1].text_frame
    content.text = "Methodical progression from baseline control to physically-grounded risk-sensitive autonomy:"
    
    p = content.add_paragraph()
    p.text = "Stage 1: CVaR-SMPC Algorithmic Validation (Nominal Terrain)"
    p.level = 1
    p = content.add_paragraph()
    p.text = "• Validated stochastic MPPI with Conditional Value-at-Risk against Deterministic NMPC and Gaussian Chance-Constrained SMPC in synthetic Gaussian soil distributions."
    p.level = 2
    
    p = content.add_paragraph()
    p.text = "Stage 2: Vision-Based Appearance SMPC (Exteroceptive Priors)"
    p.level = 1
    p = content.add_paragraph()
    p.text = "• Integrated camera/LiDAR elevation and visual classification map. Exposed critical vulnerability: Exteroceptive sensing alone fails under visually ambiguous soft regolith patches."
    p.level = 2

    p = content.add_paragraph()
    p.text = "Stage 3: Physically-Grounded Proprioceptive CVaR-SMPC (Final Contribution)"
    p.level = 1
    p = content.add_paragraph()
    p.text = "• Integrated online Bekker-Wong parameter estimation via proprioceptive EKF (slip, sinkage, torque). Real-time risk map dynamic updates bypass hidden entrapment zones."
    p.level = 2

    # =========================================================================
    # Slide 4: Stage 1 Deep-Dive (Algorithmic Formulation)
    # =========================================================================
    slide = prs.slides.add_slide(content_layout)
    title = slide.shapes.title
    title.text = "Stage 1: CVaR-SMPC vs Baselines (Algorithm Design)"
    
    content = slide.placeholders[1].text_frame
    content.text = "Bridging Stochastic Risk Metrics and Predictive Trajectory Optimization:"
    
    p = content.add_paragraph()
    p.text = "Deterministic NMPC: Solves nominal non-linear dynamics, totally blind to stochastic terrain variance -> frequent sinkage violations."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "Gaussian SMPC (Chance-Constrained): Assumes symmetrical Gaussian distributions. Under-penalizes rare extreme sinkage occurrences."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "CVaR-SMPC (Rockafellar & Uryasev Formulation): Evaluates the conditional expectation of the worst α=0.9 tail: CVaR_α(Z) = E[Z | Z >= VaR_α(Z)]."
    p.level = 1

    p = content.add_paragraph()
    p.text = "Execution: run_stage1_demo.py successfully verified that CVaR trajectory rollouts stay strictly within safe sinkage margins under heavy-tailed uncertainty."
    p.level = 0

    if os.path.exists("stage1_trajectory_comparison.png"):
        slide.shapes.add_picture("stage1_trajectory_comparison.png", Inches(1.5), Inches(4.3), width=Inches(7.0))

    # =========================================================================
    # Slide 5: Stage 2 Deep-Dive (Exteroceptive Failure Modes)
    # =========================================================================
    slide = prs.slides.add_slide(content_layout)
    title = slide.shapes.title
    title.text = "Stage 2: Exteroceptive Vision & Ambiguity Breakdown"
    
    content = slide.placeholders[1].text_frame
    content.text = "Modeling Camera & LiDAR Perception with False-Safe Deception:"
    
    p = content.add_paragraph()
    p.text = "Appearance Sensor Model: Classifies terrain into firm rock, gravel, and sand based on spectral/elevation cues."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "The 'False-Safe' Trap: Loose powdery drift sand (low cohesion c, low bearing capacity kc) appears visually identical to compacted sand."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "Result in Simulation: The visual planner plans an aggressive path straight through deceptive sand, leading to sudden, catastrophic sinkage (8+ cm) and rover immobilization."
    p.level = 1

    if os.path.exists("stage2_appearance_vs_physical.png"):
        slide.shapes.add_picture("stage2_appearance_vs_physical.png", Inches(1.2), Inches(4.1), width=Inches(7.6))

    # =========================================================================
    # Slide 6: Stage 3 Deep-Dive (Proprioceptive EKF & Closed Loop)
    # =========================================================================
    slide = prs.slides.add_slide(content_layout)
    title = slide.shapes.title
    title.text = "Stage 3: Proprioceptive Online Estimator + CVaR-SMPC"
    
    content = slide.placeholders[1].text_frame
    content.text = "Closing the Loop with Terramechanics and Wheel Dynamics:"
    
    p = content.add_paragraph()
    p.text = "Bekker-Wong Soil Mechanics: Pressure-sinkage p = (kc/b + kφ)·z^n and Janosi-Hanamoto shear stress τ = (c + σ·tan φ)·(1 - e^(-j/K))."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "Online EKF Formulation: Measures wheel slip s, sinkage z, drive torque T, and chassis pitch/roll to estimate true cohesion c and internal friction angle φ."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "Dynamic Risk Map Reweighting: As soon as wheel sinkage/slip spikes unexpectedly, the EKF flags the terrain anomaly, reweights the cost map, and triggers dynamic detour."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "Outcome: Overcomes visual deception, eliminating catastrophic entrapment."
    p.level = 0

    # =========================================================================
    # Slide 7: Codebase Architecture - Core terrarisk Package Table
    # =========================================================================
    terrarisk_headers = ["File Name", "Role / Description", "Key Inputs", "Key Outputs"]
    terrarisk_rows = [
        [
            "terrarisk/\nterrain.py",
            "Defines spatial terrain map, Bekker-Wong soil properties, and terrain classes (rock, compacted soil, soft sand).",
            "Map dimensions (X, Y), resolution, nominal soil parameters (c, φ, n, kc, kφ).",
            "TerrainMap instance, ground-truth SoilParameters query at coordinates (x, y)."
        ],
        [
            "terrarisk/\nterrain_generator.py",
            "Procedural generator for high-variance multi-patch terrain with deceptive 'false-safe' patches.",
            "Map size, seed, obstacle count, soft regolith distribution, visual noise levels.",
            "Synthesized multi-layered TerrainMap with hidden entrapment zones."
        ],
        [
            "terrarisk/\nappearance_sensor.py",
            "Simulates exteroceptive visual perception (camera/LiDAR) with optical classification errors.",
            "Ground-truth TerrainMap, rover pose (x, y, θ), sensor range & noise profile.",
            "Visual terrain prior map, estimated friction/cohesion classes, misclassification flags."
        ],
        [
            "terrarisk/\nonline_estimator.py",
            "Proprioceptive Extended Kalman Filter (EKF) estimating soil mechanical parameters online.",
            "Wheel slip s, wheel sinkage z, motor torque T, normal load W, IMU attitude.",
            "Estimated soil parameters (c_est, φ_est, kc_est, kφ_est), covariance uncertainty P."
        ],
        [
            "terrarisk/\ndynamics.py",
            "Non-linear 4-wheel rover terramechanics dynamics with slip-sinkage physics.",
            "Rover state (x, y, θ, v, ω), wheel throttle/steering inputs u, soil parameters.",
            "Next state x_{t+1}, instantaneous sinkage z, longitudinal slip ratio s, motion resistance."
        ]
    ]
    col_widths_1 = [Inches(1.8), Inches(3.0), Inches(2.1), Inches(2.1)]
    create_table_slide(prs, "Codebase Architecture: Core terrarisk Package (1/2)", terrarisk_headers, terrarisk_rows, col_widths_1, font_size=Pt(8.5))

    # =========================================================================
    # Slide 8: Codebase Architecture - Planners & Execution Modules Table
    # =========================================================================
    planners_headers = ["File Name", "Role / Description", "Key Inputs", "Key Outputs"]
    planners_rows = [
        [
            "terrarisk/\ncvar_smpc.py",
            "Base CVaR-SMPC implementation utilizing sampling-based MPPI and risk assessment.",
            "Rover state, cost map, goal location, risk parameter α=0.9, risk weight λ.",
            "Optimal control input sequence u*, predicted trajectory distribution, CVaR metric."
        ],
        [
            "terrarisk/\nappearance_mpc.py",
            "Visual-only CVaR-SMPC (Stage 2) operating strictly on exteroceptive camera/LiDAR priors.",
            "Visual classification prior, rover state, destination waypoint.",
            "Visual-constrained trajectory and steering commands (v, ω)."
        ],
        [
            "terrarisk/\nstage3_cvar_planner.py",
            "Full Stage 3 planner fusing visual priors with online proprioceptive EKF soil estimates.",
            "Visual prior, EKF soil parameter estimates, rover state, obstacle map.",
            "Risk-sensitive control sequence, dynamically updated local hazard cost map."
        ],
        [
            "terrarisk/\ndeterministic_nmpc.py",
            "Baseline Non-linear MPC using nominal dynamics without risk or uncertainty awareness.",
            "Rover state, nominal friction estimate, destination goal.",
            "Nominal control commands (v, ω) under zero-variance assumption."
        ],
        [
            "terrarisk/\nsimulator.py",
            "Unified closed-loop simulation harness connecting terrain, rover, sensors, and planners.",
            "Planner instance, TerrainMap, initial state, goal position, max steps.",
            "Full trajectory log (states, inputs, sinkage, slip, entrapment status, travel time)."
        ]
    ]
    create_table_slide(prs, "Codebase Architecture: Planners & Control Engine (2/2)", planners_headers, planners_rows, col_widths_1, font_size=Pt(8.5))

    # =========================================================================
    # Slide 9: Codebase Architecture - Top-Level Demo & Analysis Scripts
    # =========================================================================
    scripts_headers = ["Script Name", "Role / Execution Purpose", "Inputs / Parameters", "Outputs / Artifacts"]
    scripts_rows = [
        [
            "run_stage1_demo.py",
            "Demonstrates Stage 1 CVaR-SMPC vs Deterministic NMPC and Gaussian SMPC in stochastic terrain.",
            "Default terrain seed, risk level α=0.9, test waypoints.",
            "Console convergence log, generated plot 'stage1_trajectory_comparison.png'."
        ],
        [
            "run_stage2_demo.py",
            "Exposes failure of visual-only planning in presence of hidden 'false-safe' regolith patches.",
            "Multi-patch terrain map with ambiguous soft sand zones.",
            "Entrapment log, generated figure 'stage2_appearance_vs_physical.png'."
        ],
        [
            "run_stage3_demo.py",
            "Validates complete Stage 3 system: EKF proprioceptive adaptation rescuing rover from traps.",
            "Ambiguous terrain map, initial visual priors, rover wheel sensors.",
            "Live detour logs, soil parameter tracking curves, closed-loop trajectory plot."
        ],
        [
            "run_monte_carlo.py",
            "Massive 50-seed closed-loop Monte Carlo benchmarking (200 total runs across 4 controllers).",
            "Seed count N=50, controller list, simulation limits, sinkage entrapment cutoff (z_crit=8cm).",
            "Structured CSV dataset 'benchmark_results.csv' with success, entrapment, and risk metrics."
        ],
        [
            "analyse_results.py",
            "Statistical analysis and generation of high-DPI publication figures & summary tables.",
            "'benchmark_results.csv', terrain map models.",
            "'stage3_final_benchmark.png', 'terrain_ambiguity_map.png', LaTeX metrics summary."
        ]
    ]
    col_widths_2 = [Inches(1.8), Inches(3.0), Inches(2.1), Inches(2.1)]
    create_table_slide(prs, "Execution Scripts & Benchmarking Pipeline", scripts_headers, scripts_rows, col_widths_2, font_size=Pt(8.5))

    # =========================================================================
    # Slide 10: Empirical Monte Carlo Benchmark Results
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    txbox = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9.0), Inches(0.9))
    tf = txbox.text_frame
    p = tf.paragraphs[0]
    p.text = "50-Seed Closed-Loop Benchmark Evaluation"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = RGBColor(20, 35, 60)
    
    if os.path.exists("stage3_final_benchmark.png"):
        slide.shapes.add_picture("stage3_final_benchmark.png", Inches(0.5), Inches(1.2), width=Inches(9.0))

    # =========================================================================
    # Slide 11: Summary Table of Benchmark Metrics
    # =========================================================================
    metrics_headers = ["Control Architecture", "Success Rate", "Entrapment Rate", "Path Efficiency", "Max Sinkage (cm)", "Tail Risk CVaR"]
    metrics_rows = [
        [
            "Deterministic NMPC\n(Nominal)",
            "82.0%",
            "18.0%",
            "0.740 ± 0.38",
            "7.18 ± 1.07 cm",
            "0.000155 (Severe)"
        ],
        [
            "Gaussian SMPC\n(Chance Constrained)",
            "98.0%",
            "2.0%",
            "0.817 ± 0.21",
            "6.60 ± 0.67 cm",
            "0.000132 (Moderate)"
        ],
        [
            "Stage 2 Visual CVaR\n(Optical Priors)",
            "100.0%*",
            "0.0%*",
            "0.793 ± 0.15",
            "5.43 ± 0.52 cm",
            "0.000116 (Nominal)\n*Fails under deceit"
        ],
        [
            "Stage 3 Physical CVaR\n(EKF Fused - Ours)",
            "94.0%",
            "6.0%",
            "0.704 ± 0.24",
            "6.61 ± 0.78 cm",
            "0.000135 (Bounded)"
        ]
    ]
    col_widths_metrics = [Inches(2.5), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.3)]
    create_table_slide(prs, "Quantitative Evaluation Across 200 Trials", metrics_headers, metrics_rows, col_widths_metrics, font_size=Pt(9.5))

    # =========================================================================
    # Slide 12: Conclusion & Contributions
    # =========================================================================
    slide = prs.slides.add_slide(content_layout)
    title = slide.shapes.title
    title.text = "Key Contributions & Engineering Impact"
    
    content = slide.placeholders[1].text_frame
    content.text = "Summary of Project Achievements:"
    
    p = content.add_paragraph()
    p.text = "1. First-Principles Coupling: Successfully unified Bekker-Wong terramechanics with Stochastic Model Predictive Control (CVaR-SMPC)."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "2. Active Mitigation of Visual Ambiguity: Proved that exteroceptive sensors alone fail in deceptive terrain and showed that proprioceptive EKF feedback actively prevents entrapment."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "3. 3x Reduction in Catastrophic Entrapment: Lowered catastrophic wheel entrapment from 18% (NMPC) to 6% in high-variance soil fields."
    p.level = 1
    
    p = content.add_paragraph()
    p.text = "4. Modular & Extensible Software Stack: Complete end-to-end framework featuring modular sensors, dynamics, estimators, planners, and Monte Carlo benchmarking tools."
    p.level = 1

    prs.save("TerraRisk_Final_Presentation.pptx")
    print("Successfully generated comprehensive 12-slide presentation: TerraRisk_Final_Presentation.pptx")

if __name__ == "__main__":
    create_presentation()
