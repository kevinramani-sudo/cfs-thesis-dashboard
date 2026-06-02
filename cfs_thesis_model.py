"""
═══════════════════════════════════════════════════════════════════════════════
CFS MULTI-OBJECTIVE OPTIMIZATION FRAMEWORK —
Cold-Formed Steel Residential Framing — NSGA-II Implementation
═══════════════════════════════════════════════════════════════════════════════
Market references (2025 NZ):
  - Steel supply-fabricate: NZ$4.50–7.00/kg (SCNZ 2024, fabricator quotes)
  - Site labor wage: NZ$35/hr (ERI 2026); on-costs 37% → NZ$48/hr all-in
  - CFS A1-A3 carbon: 2.10 kgCO2e/kg (imported EAF CFS, AS/NZS context; ICE DB v3.0)
  - Transport: NZ$3.00/tonne-km (NZTA freight benchmark; Auckland-Waikato ~121 km)
  - A5 installation: 3.20 kgCO2e/m² (BRANZ LCA data; includes diesel, waste, equipment)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deap import base, creator, tools
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import random
import warnings
import json
import os
from pathlib import Path
from datetime import datetime
from itertools import product as iproduct
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FOLDERS
# ─────────────────────────────────────────────────────────────────────────────

_THIS_FILE = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
_PROJECT_ROOT_DEFAULT = (
    _THIS_FILE.parent.parent
    if _THIS_FILE.parent.name.lower() == "scripts"
    else _THIS_FILE.parent
)

PROJECT_DIR = Path(
    os.environ.get("CFS_THESIS_PROJECT_DIR", str(_PROJECT_ROOT_DEFAULT))
).resolve()

RESULTS_DIR = PROJECT_DIR / "results_fixed"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
DASHBOARD_DIR = PROJECT_DIR / "dashboard"
DASHBOARD_DATA_DIR = DASHBOARD_DIR / "data"

for folder in (RESULTS_DIR, FIG_DIR, TABLE_DIR, DASHBOARD_DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

print(f"[PROJECT ROOT]   {PROJECT_DIR}")
print(f"[OUTPUT ROOT]    {RESULTS_DIR}")
print(f"[FIGURES]        {FIG_DIR}")
print(f"[TABLES]         {TABLE_DIR}")
print(f"[DASHBOARD DATA] {DASHBOARD_DATA_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# RUN CONTROL
# ─────────────────────────────────────────────────────────────────────────────

INIT_METHOD = "lhs"   # "lhs" or "random"

MAIN_SEED = 42
MULTI_SEED_SEEDS = (11, 22, 33, 44, 55)

RUN_MODE = "final"    # "debug" for testing; "final" for thesis outputs 

if RUN_MODE == "debug":
    POP_SIZE = 60
    GENERATIONS = 80
    HV_SAMPLES = 30000
    HV_EVERY = 20
    GRID_PF_STEP = 0.10

    RUN_MONTE_CARLO = False
    MC_RUNS = 50

    RUN_MULTI_SEED = False

    RUN_LHS_GLOBAL_SENSITIVITY = True
    LHS_DOE_SAMPLES = 500

elif RUN_MODE == "final":
    POP_SIZE = 200
    GENERATIONS = 300
    HV_SAMPLES = 500000
    HV_EVERY = 10
    GRID_PF_STEP = 0.05

    RUN_MONTE_CARLO = True
    MC_RUNS = 1000

    RUN_MULTI_SEED = True

    RUN_LHS_GLOBAL_SENSITIVITY = True
    LHS_DOE_SAMPLES = 2000

else:
    raise ValueError("RUN_MODE must be either 'debug' or 'final'.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PROJECT BASELINE PARAMETERS (REFACTORED — physical variables)
# ─────────────────────────────────────────────────────────────────────────────
class CFSProjectBaseline:
    """
    S2 reference case: single-storey residential, 98.3 m² footprint, Waikato NZ.

    DESIGN VARIABLES (refactored to physical engineering quantities):
      
            x = [R_p, R_t, OC, P_f]
      panel_scheme  : index ∈ {0,1,2,3} → set of standard panel widths (mm)
      truss_scheme  : index ∈ {0,1,2}   → set of standard truss spans (m)
      opening_complexity_index : continuous opening/detailing complexity factor
      P_f           : prefabrication level ∈ [0.50, 0.90]
    """
    # ── PROJECT GEOMETRY ─────────────────────────────────────────────────────
    footprint_area_m2 = 98.292
    envelope_area_m2  = 217.2555
    num_wall_panels   = 48
    num_roof_trusses  = 58
    W_base            = 2195.264
    # ── REFERENCE SCENARIO TIMES (S2 medium-prefabrication scenario)
    T_factory_truss_base       = 24.20
    T_factory_panel_base       = 11.70
    T_factory_setup_base      = 3.50    # hrs — jig load + CNC program setup (S2)
    T_delivery_logistics_base = 2.00    # hrs — truck scheduling + unload window
    T_site_mobilisation_base  = 3.00    # hrs — site setup, crane arrival, PPE, layout
    T_installation_truss_base  = 18.18
    T_installation_panel_base  = 57.06
    
    T_factory_production_base = T_factory_truss_base + T_factory_panel_base
    T_factory_base = T_factory_production_base
    T_installation_base = T_installation_truss_base + T_installation_panel_base

    # MODEL-CALCULATED S2 REFERENCE VALUES 
    raw_steel_material_nzd_per_kg = 2.40
    factory_processing_nzd_per_kg = 2.80
    site_labor_rate_nzd_per_hr    = 48.0
       

    C_setup_per_panel_type = 25.0
    C_setup_per_truss_type = 35.0
    C_jig_base             = 800.0

    transport_cost_nzd_per_tonne_km = 3.00

    delivery_logistics_rate_nzd_per_hr = site_labor_rate_nzd_per_hr

    site_mobilisation_rate_nzd_per_hr = site_labor_rate_nzd_per_hr

    lifting_equipment_cost_nzd = 0.0
    steel_waste_rate_ref = 0.05
    steel_waste_disposal_nzd_per_tonne = 250.0

    overhead_fraction_ref = 0.10

    steel_carbon_factor_a1_a3_kgco2e_per_kg = 2.10
    a4_transport_carbon_kgco2e_per_kg       = 0.016335
    a4_default_transport_distance_km        = 121.0
    a5_installation_carbon_kgco2e_per_m2    = 3.20
      
    freight_kgco2e_per_tonne_km = 0.135

 
    nz_grid_kgco2e_per_kwh = 0.08

    factory_average_power_kw = 5.0
    site_average_power_kw = 2.0

    
    include_factory_energy_carbon = True

    include_site_energy_carbon_separately = False

   
    include_waste_carbon = True
    Time_S2_reference = (
        T_factory_production_base
        + T_factory_setup_base
        + T_delivery_logistics_base
        + T_installation_base
        + T_site_mobilisation_base
    )
        
    # ════════════════════════════════════════════════════════════════════════
    # PHYSICAL DESIGN VARIABLES
    # ════════════════════════════════════════════════════════════════════════
    
    # ── PANEL WIDTH SCHEMES (mm) ─────────────────────────────────────────────
    # Each scheme is a committed set of standard panel widths
    # Source: FRAMECAD design library; Howick standard panel sizes
    panel_width_schemes = {
        0: [600, 900, 1200, 1500, 1800, 2100],   # S2 baseline (4 active classes)
        1: [900, 1200, 1500, 1800],              # Moderate rationalisation
        2: [1200, 1500, 1800],                   # Strong rationalisation
        3: [1200, 1800],                         # Aggressive rationalisation
    }
    # Effective number of panel classes for each scheme
    panel_scheme_to_classes = {0: 4, 1: 4, 2: 3, 3: 2}
    panel_scheme_values     = [0, 1, 2, 3]

    # ── CONTINUOUS RATIONALISATION VARIABLES ────────────────────────────────────
    # These replace panel_scheme and truss_scheme as optimiser variables.
    # They do not add new design decisions; they re-express scheme selection as
    # continuous rationalisation intensity.

    R_p_range = (0.0, 1.0)   # panel rationalisation intensity
    R_t_range = (0.0, 1.0)   # truss rationalisation intensity

    # ── TRUSS SPAN SCHEMES (m) ───────────────────────────────────────────────
    # Each scheme is a committed set of truss span families
    truss_span_schemes = {
        0: [3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7,
            6.0, 6.3, 6.6, 6.9, 7.2, 7.5, 7.8],   # 17 spans (S2)
        1: [3.0, 3.3, 3.6, 3.9, 4.2, 4.5, 5.1, 5.4, 5.7,
            6.0, 6.3, 6.6, 6.9, 7.2, 7.5, 7.8],   # 16 spans
        2: [3.0, 3.6, 4.2, 4.5, 4.8, 5.1, 5.4, 6.0,
            6.3, 6.6, 6.9, 7.2, 7.5, 7.8],        # 14 spans     
    }
    truss_scheme_to_types = {0: 17, 1: 16, 2: 14}
    truss_scheme_values   = [0, 1, 2]

    # ── OPENING / DETAILING COMPLEXITY INDEX ─────────────────────────────────
    # ACTIONABLE DESIGN VARIABLE:
    # This does NOT change the number of doors/windows.
    # The number of openings is fixed at N_openings_reference.
    #
    # The variable represents design-for-manufacture/detailing rationalisation:
    #   lower OC  = more standardised lintels, jamb studs, service penetrations,
    #               repeated opening modules, simpler CNC cutting and site fitting
    #   higher OC = more custom opening sizes, non-repeated trimmers, local
    #               reinforcement, extra coordination, and higher installation difficulty
    #
    # Therefore, OC is a controllable detailing/design-standardisation variable,
    # not an architectural variable that removes openings.

    opening_complexity_range = (0.80, 1.30)
    opening_complexity_ref   = 1.00

    N_openings_reference = 12

    opening_complexity_action_levels = {
        0.80: "High standardisation of opening details and repeated lintel/jamb modules",
        1.00: "S2 reference detailing complexity",
        1.15: "Moderate non-standard opening/detailing complexity",
        1.30: "High custom detailing, trimming, coordination and installation complexity",
    }
    # ── PREFABRICATION LEVEL ─────────────────────────────────────────────────
    P_f_range = (0.50, 0.90)

    # ── BASELINE CLASSIFICATION ──────────────────────────────────────────────
    baseline_panel_class_count = 4
    baseline_truss_type_count  = 17

    prefab_scenarios = {
        "S1_low_prefab": 0.50,
        "S2_medium_prefab": 0.72,
        "S3_high_prefab": 0.90,
    }

    # ── REFERENCE DESIGN VECTOR ──────────────────────────────────────────────
    # x_ref = [R_p, R_t, opening_complexity_index, P_f]
    x_ref = [0.0, 0.0, 1.00, 0.72]

    # ── MODEL-CALCULATED S2 COST REFERENCE ──────────────────────────────────
    # This is CFS framing process cost only, not whole-house construction cost.

    P_norm_ref = (x_ref[3] - P_f_range[0]) / (P_f_range[1] - P_f_range[0])
    factory_processing_multiplier_ref = 1.0 + 0.45 * P_norm_ref

    C_ref_steel_material = W_base * raw_steel_material_nzd_per_kg

    C_ref_factory_processing = (
        W_base
        * factory_processing_nzd_per_kg
        * factory_processing_multiplier_ref
    )

    C_ref_site_installation_labour = (
        T_installation_base
        * site_labor_rate_nzd_per_hr
    )

    C_ref_factory_setup = (
        baseline_panel_class_count * C_setup_per_panel_type
        + baseline_truss_type_count * C_setup_per_truss_type
        + C_jig_base
    )

    C_ref_transport = (
        (W_base / 1000.0)
        * a4_default_transport_distance_km
        * transport_cost_nzd_per_tonne_km
    )

    C_ref_delivery_logistics = (
        T_delivery_logistics_base
        * delivery_logistics_rate_nzd_per_hr
    )

    C_ref_site_mobilisation = (
        T_site_mobilisation_base
        * site_mobilisation_rate_nzd_per_hr
    )

    C_ref_waste_handling = (
        (W_base * steel_waste_rate_ref / 1000.0)
        * steel_waste_disposal_nzd_per_tonne
    )

    C_ref_lifting_equipment = lifting_equipment_cost_nzd

    C_ref_direct_subtotal = (
        C_ref_steel_material
        + C_ref_factory_processing
        + C_ref_site_installation_labour
        + C_ref_factory_setup
        + C_ref_transport
        + C_ref_delivery_logistics
        + C_ref_site_mobilisation
        + C_ref_waste_handling
        + C_ref_lifting_equipment
    )

    C_ref_overhead = overhead_fraction_ref * C_ref_direct_subtotal

    C_S2_reference = C_ref_direct_subtotal + C_ref_overhead

    # ── MODEL-CALCULATED S2 CARBON REFERENCE ────────────────────────────────
    # This is CFS framing carbon only, not whole-building embodied carbon.

    CO2_ref_A1_A3_steel = (
        W_base
        * steel_carbon_factor_a1_a3_kgco2e_per_kg
    )

    CO2_ref_A4_transport = (
        (W_base / 1000.0)
        * a4_default_transport_distance_km
        * freight_kgco2e_per_tonne_km
    )

    CO2_ref_factory_energy = (
        T_factory_production_base
        * factory_average_power_kw
        * nz_grid_kgco2e_per_kwh
        if include_factory_energy_carbon else 0.0
    )

    CO2_ref_site_energy = (
        T_installation_base
        * site_average_power_kw
        * nz_grid_kgco2e_per_kwh
        if include_site_energy_carbon_separately else 0.0
    )

    CO2_ref_A5_installation = (
        envelope_area_m2
        * a5_installation_carbon_kgco2e_per_m2
    )

    CO2_ref_waste = (
        W_base
        * steel_waste_rate_ref
        * steel_carbon_factor_a1_a3_kgco2e_per_kg
        if include_waste_carbon else 0.0
    )

    CO2_S2_reference = (
        CO2_ref_A1_A3_steel
        + CO2_ref_A4_transport
        + CO2_ref_factory_energy
        + CO2_ref_site_energy
        + CO2_ref_A5_installation
        + CO2_ref_waste
    )

    C_baseline    = C_S2_reference
    CO2_baseline  = CO2_S2_reference
    Time_baseline = Time_S2_reference

    S2_reference_data_status = "model-calculated CFS framing reference, not measured commercial data"

    S2_cost_scope = (
        "CFS framing process cost: steel material + factory processing + site installation labour "
        "+ setup + transport + delivery/logistics + mobilisation + waste handling + overhead."
    )

    S2_carbon_scope = (
        "CFS framing carbon: A1-A3 steel + A4 transport + factory energy + A5 site installation "
        "+ waste allowance. Whole-building carbon is outside scope."
    )

    S2_time_scope = (
        "Total construction process hours: factory production + setup + delivery/logistics "
        "+ site installation + site mobilisation. Not calendar duration."
    )
    # ── VALIDATED / DEFENSIBLE PARAMETER RANGES FOR UNCERTAINTY ANALYSIS ─────

    raw_steel_material_nzd_per_kg_range = (2.00, 2.65)
    factory_processing_nzd_per_kg_range = (2.20, 3.80)
    combined_supply_processing_nzd_per_kg_range = (4.50, 7.00)

    site_labor_rate_nzd_per_hr_range = (42.00, 58.00)
    transport_cost_nzd_per_tonne_km_range = (2.40, 4.20)

    steel_carbon_factor_a1_a3_range = (1.50, 2.89)
    a5_installation_carbon_range = (1.50, 4.50)
    freight_kgco2e_per_tonne_km_range = (0.107, 0.155)

    C_setup_per_panel_type_range = (15.00, 35.00)
    C_setup_per_truss_type_range = (25.00, 50.00)

    k_prefab_time_saving_range = (0.40, 0.65)
    k_prefab_waste_reduction_range = (0.10, 0.25)
    k_prefab_a5_carbon_saving_range = (0.30, 0.60)

    k_logistics_cost_range = (0.06, 0.18)
    k_logistics_time_range = (0.08, 0.23)
    k_logistics_carbon_range = (0.02, 0.07)

    k_opening_cost_range = (0.08, 0.25)
    k_opening_carbon_range = (0.05, 0.14)
    k_opening_time_range = (0.12, 0.35)

    lambda_opening_prefab_reduction_range = (0.30, 0.60)

    waikato_weather_downtime_fraction_range = (0.05, 0.15)
    weather_shielding_factor_range = (0.40, 0.75)
    # ── INTERACTION COEFFICIENTS ─────────────────────────────────────────────
    # Central values are selected within literature/plausibility ranges and tested
    # through Monte Carlo uncertainty analysis.

    k_prefab_time_saving      = 0.620
    k_prefab_waste_reduction  = 0.150

    # Conservative central value; high values are tested through uncertainty analysis.
    k_prefab_a5_carbon_saving = 0.450

    k_logistics_cost          = 0.110
    k_logistics_time          = 0.150
    k_logistics_carbon        = 0.040

    k_opening_cost            = 0.160
    k_opening_carbon          = 0.090

    # Conservative central value; the uncertainty range tests higher opening-time effects.
    k_opening_time            = 0.160

    lambda_opening_prefab_reduction = 0.45


    # Factory bottleneck threshold (non-linear cost above this)
    P_f_factory_saturation = 0.80

    # Physical minimum site time
    T_site_minimum_hrs = 28.0
    # ─────────────────────────────────────────────────────────────────────
    # BALANCED RESPONSE COEFFICIENTS
    # These prevent artificial boundary domination by giving each variable
    # both benefit and penalty mechanisms.
    # ─────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────
    # TUNED BALANCED RESPONSE COEFFICIENTS 
    # ─────────────────────────────────────────────────────────────────────
    # Rationalisation knee points
    R_p_knee = 0.65
    R_t_knee = 0.65
    # Panel rationalisation effects
    k_Rp_setup_saving = 0.22
    k_Rp_site_saving = 0.030
    k_Rp_waste_saving = 0.012
    k_Rp_factory_penalty = 0.050
    k_Rp_logistics_penalty = 0.050
    k_Rp_oversize_penalty = 0.160

    # Truss rationalisation effects
    k_Rt_setup_saving = 0.12
    k_Rt_site_saving = 0.020
    k_Rt_waste_saving = 0.007
    k_Rt_factory_penalty = 0.030
    k_Rt_logistics_penalty = 0.040
    k_Rt_oversize_penalty = 0.120

    # Opening/detailing complexity effects
    k_OC_low_design_cost = 0.030
    k_OC_low_factory_time = 0.030
    k_OC_low_material_penalty = 0.020

    k_OC_high_cost = 0.220
    k_OC_high_time = 0.280
    k_OC_high_carbon = 0.120
    k_OC_high_material_penalty = 0.060

    # Prefabrication response
    k_prefab_factory_time_increase = 0.28
    k_prefab_site_time_saving = 0.60
    k_prefab_factory_cost_increase = 0.20
    k_prefab_a5_carbon_saving = 0.35

    # Nonlinear penalty above high prefabrication
    P_f_factory_saturation = 0.78
    k_prefab_overload_cost = 0.42
    k_prefab_overload_time = 0.30
    k_prefab_overload_carbon = 0.08

    # ── NZ CALIBRATION CONSTRAINTS ────────────────────────────────────
    # These are optimization-level feasibility and constructability constraints.
    # They are not a substitute for detailed AS/NZS 4600 member design,
    # connection design, bracing design, or building consent documentation.

    # HARD FEASIBILITY CONSTRAINTS
    max_panel_width_m = 2.40          # conservative transport/factory limit
    max_panel_height_m = 3.00         # practical factory/lifting limit
    assumed_panel_height_m = 2.70     # assumed residential wall panel height

    max_stud_spacing_mm = 600.0
    assumed_stud_spacing_mm = 600.0

    # Actual project CFS section from the FRAMECAD/material schedule:
    # F325iT - Imported Section (89 S 41 / 0.75 / G500 / Z275)
    cfs_section_source = "F325iT - Imported Section"
    cfs_section_designation = "89 S 41"
    assumed_steel_thickness_mm = 0.75
    steel_grade = "G500"
    coating_class = "Z275"

    # Screening-only lower bound. This is not the project thickness.
    min_steel_thickness_mm = 0.55

    max_lift_mass_kg = 2000.0         # site handling / lifting limit

    # For panel mass estimation only.
    # If you later split wall/truss weights, replace this approximation.
    wall_steel_weight_fraction = 0.60

    # SOFT CONSTRUCTABILITY CONSTRAINTS
    max_unique_truss_spans_soft = 25
    min_factory_repetition_score = 0.65
    max_OCI_soft = 1.20

    # WEATHER / SCENARIO SETTINGS
    waikato_weather_downtime_fraction = 0.10
    weather_shielding_factor = 0.60

    # Scenario assumption only, not used as a hard code limit in the main run.
    carbon_cap_kgco2e_per_m2 = 50.0
    apply_carbon_cap_penalty = False

    # Penalty magnitudes in normalized objective space
    hard_constraint_penalty = 1.0e6
    soft_constraint_penalty = 1.0

    def __init__(self):
        """
        Recalculate all reference quantities at instance creation.

        The class stores many coefficients as defaults. This instance-level
        recomputation is essential because Monte Carlo analysis changes
        parameters on a CFSProjectBaseline() instance; without recomputation,
        C_S2_reference, CO2_S2_reference and C_ref_direct_subtotal remain frozen
        at the original class-level values.
        """
        self.recompute_references()

    def recompute_references(self):
        """
        Recompute model-calculated S2 reference values from the current instance
        coefficients. Call this after changing rates/factors in Monte Carlo or
        dashboard recalibration.
        """
        self.T_factory_production_base = (
            self.T_factory_truss_base
            + self.T_factory_panel_base
        )

        self.T_factory_base = self.T_factory_production_base

        self.T_installation_base = (
            self.T_installation_truss_base
            + self.T_installation_panel_base
        )

        self.delivery_logistics_rate_nzd_per_hr = self.site_labor_rate_nzd_per_hr
        self.site_mobilisation_rate_nzd_per_hr = self.site_labor_rate_nzd_per_hr

        self.Time_S2_reference = (
            self.T_factory_production_base
            + self.T_factory_setup_base
            + self.T_delivery_logistics_base
            + self.T_installation_base
            + self.T_site_mobilisation_base
        )

        self.P_norm_ref = (
            (self.x_ref[3] - self.P_f_range[0])
            / (self.P_f_range[1] - self.P_f_range[0])
        )

        self.factory_processing_multiplier_ref = 1.0 + 0.45 * self.P_norm_ref

        self.C_ref_steel_material = (
            self.W_base
            * self.raw_steel_material_nzd_per_kg
        )

        self.C_ref_factory_processing = (
            self.W_base
            * self.factory_processing_nzd_per_kg
            * self.factory_processing_multiplier_ref
        )

        self.C_ref_site_installation_labour = (
            self.T_installation_base
            * self.site_labor_rate_nzd_per_hr
        )

        self.C_ref_factory_setup = (
            self.baseline_panel_class_count * self.C_setup_per_panel_type
            + self.baseline_truss_type_count * self.C_setup_per_truss_type
            + self.C_jig_base
        )

        self.C_ref_transport = (
            (self.W_base / 1000.0)
            * self.a4_default_transport_distance_km
            * self.transport_cost_nzd_per_tonne_km
        )

        self.C_ref_delivery_logistics = (
            self.T_delivery_logistics_base
            * self.delivery_logistics_rate_nzd_per_hr
        )

        self.C_ref_site_mobilisation = (
            self.T_site_mobilisation_base
            * self.site_mobilisation_rate_nzd_per_hr
        )

        self.C_ref_waste_handling = (
            (self.W_base * self.steel_waste_rate_ref / 1000.0)
            * self.steel_waste_disposal_nzd_per_tonne
        )

        self.C_ref_lifting_equipment = self.lifting_equipment_cost_nzd

        self.C_ref_direct_subtotal = (
            self.C_ref_steel_material
            + self.C_ref_factory_processing
            + self.C_ref_site_installation_labour
            + self.C_ref_factory_setup
            + self.C_ref_transport
            + self.C_ref_delivery_logistics
            + self.C_ref_site_mobilisation
            + self.C_ref_waste_handling
            + self.C_ref_lifting_equipment
        )

        self.C_ref_overhead = (
            self.overhead_fraction_ref
            * self.C_ref_direct_subtotal
        )

        self.C_S2_reference = (
            self.C_ref_direct_subtotal
            + self.C_ref_overhead
        )

        self.a4_transport_carbon_kgco2e_per_kg = (
            self.freight_kgco2e_per_tonne_km
            * self.a4_default_transport_distance_km
            / 1000.0
        )

        self.CO2_ref_A1_A3_steel = (
            self.W_base
            * self.steel_carbon_factor_a1_a3_kgco2e_per_kg
        )

        self.CO2_ref_A4_transport = (
            (self.W_base / 1000.0)
            * self.a4_default_transport_distance_km
            * self.freight_kgco2e_per_tonne_km
        )

        self.CO2_ref_factory_energy = (
            self.T_factory_production_base
            * self.factory_average_power_kw
            * self.nz_grid_kgco2e_per_kwh
            if self.include_factory_energy_carbon else 0.0
        )

        self.CO2_ref_site_energy = (
            self.T_installation_base
            * self.site_average_power_kw
            * self.nz_grid_kgco2e_per_kwh
            if self.include_site_energy_carbon_separately else 0.0
        )

        self.CO2_ref_A5_installation = (
            self.envelope_area_m2
            * self.a5_installation_carbon_kgco2e_per_m2
        )

        self.CO2_ref_waste = (
            self.W_base
            * self.steel_waste_rate_ref
            * self.steel_carbon_factor_a1_a3_kgco2e_per_kg
            if self.include_waste_carbon else 0.0
        )

        self.CO2_S2_reference = (
            self.CO2_ref_A1_A3_steel
            + self.CO2_ref_A4_transport
            + self.CO2_ref_factory_energy
            + self.CO2_ref_site_energy
            + self.CO2_ref_A5_installation
            + self.CO2_ref_waste
        )

        self.C_baseline = self.C_S2_reference
        self.CO2_baseline = self.CO2_S2_reference
        self.Time_baseline = self.Time_S2_reference


def validate_scheme_counts(b):
    print("\n[SCHEME COUNT CHECK]")

    for k, spans in b.truss_span_schemes.items():
        declared = b.truss_scheme_to_types[k]
        actual_unique = len(set(spans))

        if declared != actual_unique:
            print(
                f"[WARNING] truss_scheme {k}: "
                f"declared={declared}, unique_spans={actual_unique}"
            )
        else:
            print(
                f"[OK] truss_scheme {k}: "
                f"declared={declared}, unique_spans={actual_unique}"
            )

    for k, widths in b.panel_width_schemes.items():
        declared = b.panel_scheme_to_classes[k]
        listed = len(set(widths))
        print(
            f"[CHECK] panel_scheme {k}: "
            f"listed_widths={listed}, active_classes={declared}"
        )
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: OBJECTIVE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
class CFSObjectiveFunctions:
    """
     ENGINEERING RESPONSE MODEL
    # Design vector:
    #     x = [R_p, R_t, opening_complexity_index, P_f]
    #
    # R_p = panel rationalisation intensity [0, 1]
    # R_t = truss rationalisation intensity [0, 1]
    """

    def __init__(self, b: CFSProjectBaseline):
        self.b = b
        self._C_ref = self._CO2_ref = self._T_ref = None

        # ─────────────────────────────────────────────────────────────────────
    # ENGINEERING RESPONSE MODEL
    # Design vector:
    #     x = [R_p, R_t, opening_complexity_index, P_f]
    #
    # R_p = panel rationalisation intensity [0, 1]
    # R_t = truss rationalisation intensity [0, 1]
    # ─────────────────────────────────────────────────────────────────────


    def _clip_Rp(self, val):
        return float(np.clip(float(val), self.b.R_p_range[0], self.b.R_p_range[1]))

    def _clip_Rt(self, val):
        return float(np.clip(float(val), self.b.R_t_range[0], self.b.R_t_range[1]))

    def _clip_opening_complexity(self, val):
        return float(np.clip(
            float(val),
            self.b.opening_complexity_range[0],
            self.b.opening_complexity_range[1]
        ))

    # ── REPORTING MAPPING ONLY ───────────────────────────────────────────
    # The optimiser uses continuous R_p and R_t.
    # These functions only map the continuous value back to a practical
    # scheme label for tables, plots and dashboard output.

    def panel_scheme_from_Rp(self, R_p):
        R_p = self._clip_Rp(R_p)

        if R_p < 0.20:
            return 0
        elif R_p < 0.45:
            return 1
        elif R_p < 0.75:
            return 2
        else:
            return 3

    def truss_scheme_from_Rt(self, R_t):
        R_t = self._clip_Rt(R_t)

        if R_t < 0.20:
            return 0
        elif R_t < 0.65:
            return 1
        else:
            return 2

    # ── EFFECTIVE CLASS / TYPE COUNTS ────────────────────────────────────

    def n_panel_classes(self, R_p):
        """
        Effective number of panel classes.

        R_p = 0.00 → 4 panel classes, S2-like
        R_p = 0.50 → 3 panel classes
        R_p = 1.00 → 2 panel classes
        """
        R_p = self._clip_Rp(R_p)
        return float(4.0 - 2.0 * R_p)

    def n_truss_types(self, R_t):
        """
        Effective number of truss types.

        R_t = 0.00 → 17 truss types, S2-like
        R_t = 1.00 → 14 truss types
        """
        R_t = self._clip_Rt(R_t)
        return float(17.0 - 3.0 * R_t)

    # ── REPETITION INDICES ───────────────────────────────────────────────

    def _RI_p(self, R_p):
        N_class = self.n_panel_classes(R_p)
        RI = self.b.num_wall_panels / N_class
        RI_baseline = self.b.num_wall_panels / self.b.baseline_panel_class_count
        return RI / RI_baseline

    def _RI_t(self, R_t):
        N_type = self.n_truss_types(R_t)
        RI = self.b.num_roof_trusses / N_type
        RI_baseline = self.b.num_roof_trusses / self.b.baseline_truss_type_count
        return RI / RI_baseline

    # ── PREFABRICATION / LOGISTICS RESPONSE ──────────────────────────────

    def _prefab_response(self, P_f, R_p, R_t):
        """
        Uses an exponential prefabrication response function.
        Objective effects are applied as deviation from the S2 reference condition.
        """
        P_f = float(np.clip(P_f, self.b.P_f_range[0], self.b.P_f_range[1]))

        P_norm = (
            (P_f - self.b.P_f_range[0])
            / (self.b.P_f_range[1] - self.b.P_f_range[0])
        )

        benefit = (
            (1.0 - np.exp(-2.3 * P_norm))
            / (1.0 - np.exp(-2.3))
        )

        # Engineering logic:
        # Higher prefabrication increases logistics sensitivity.
        # Higher panel/truss rationalisation may increase transport/lifting
        # sensitivity because elements become larger/more integrated.
        logistics = P_norm**2 * (
            0.70
            + 0.20 * self._clip_Rp(R_p)
            + 0.10 * self._clip_Rt(R_t)
        )

        return P_norm, benefit, logistics

    def _reference_response(self):
        """
        S2 response values.
        These are used only to calculate deltas.
        """
        R_p_ref = self.b.x_ref[0]
        R_t_ref = self.b.x_ref[1]
        P_ref = self.b.x_ref[3]

        P_norm_ref, benefit_ref, logistics_ref = self._prefab_response(
            P_ref,
            R_p_ref,
            R_t_ref
        )

        return P_norm_ref, benefit_ref, logistics_ref

    # ── OPENING / DETAILING RESPONSE ─────────────────────────────────────

    def _opening_response(self, oc):
        """
        Revised opening/detailing complexity response.

        Physical basis:
            OC < 1.00 : standardised lintels, repeated jamb modules, pre-cut
                        service penetrations → genuine labour and material saving.
                        Benefit is linear but capped: cannot exceed the physical
                        limit of fully standardised opening details (~15% saving
                        on opening-related labour, not total framing cost).
            OC > 1.00 : custom trimming, non-standard lintels, extra coordination.
                        Penalty is quadratic because complexity compounds:
                        each non-standard opening affects adjacent panels.
            OC = 1.00 : S2 reference condition → response = 0.

        Coefficients:
            k_benefit = 0.60  : fraction of (1 - OC) that translates to net
                                saving. At OC=0.80 → 0.60 × 0.20 = 0.12 benefit.
                                Upper-bounded at 0.12 (equivalent to OC floor).
            k_penalty = 1.80  : quadratic scaling above OC=1.0.
                                At OC=1.30 → 1.80 × 0.09 = 0.162 penalty.
                                This exceeds the maximum benefit, which is
                                physically correct: complexity hurts more than
                                standardisation helps.
        """
        oc = self._clip_opening_complexity(oc)

        oc_ref = self.b.opening_complexity_ref   # = 1.00

        if oc <= oc_ref:
            # Benefit: linear in (oc_ref - oc), capped at 12%
            raw_benefit = 0.60 * (oc_ref - oc)
            response = -min(raw_benefit, 0.12)   # negative = improvement
        else:
            # Penalty: quadratic in (oc - oc_ref)
            delta = oc - oc_ref
            response = 1.80 * (delta ** 2) + 0.50 * delta  # quadratic + linear

        oc_high = max(0.0, oc - oc_ref)
        oc_low  = max(0.0, oc_ref - oc)

        return float(response), float(oc_high), float(oc_low)

    # ── FACTORY OVERLOAD ─────────────────────────────────────────────────

    def _factory_overload(self, P_f):
        """
        Non-linear cost penalty for factory capacity saturation above P_f=0.80.
        
        Physical basis: additional jig reconfiguration, overtime, and crane
        scheduling costs when the factory operates above its designed throughput.
        Estimated at NZ$500-800 for a single residential project at maximum
        prefabrication intensity (based on fabricator interview data or 
        FRAMECAD production scheduling literature — cite here).
        
        Fixed scaling base avoids circular dependency on S2 reference cost.
        """
        if P_f <= self.b.P_f_factory_saturation:
            return 0.0

        overload = (P_f - self.b.P_f_factory_saturation) / 0.10
        # Fixed base: average NZ CFS factory setup cost allowance
        return 500.0 * overload ** 2

    # ── ENGINEERING STATE ────────────────────────────────────────────────

    def _engineering_state(self, x):
        """
        Shared engineering state for all objective functions.

        Design vector:
            x = [R_p, R_t, OC, P_f]

        R_p = panel rationalisation intensity
        R_t = truss rationalisation intensity
        OC  = opening/detailing complexity index
        P_f = prefabrication level
        """

        R_p, R_t, OC, P_f = decode_individual(x, self.b)

        R_p = float(np.clip(R_p, *self.b.R_p_range))
        R_t = float(np.clip(R_t, *self.b.R_t_range))
        OC = float(np.clip(OC, *self.b.opening_complexity_range))
        P_f = float(np.clip(P_f, *self.b.P_f_range))

        # Normalised prefabrication level
        P_norm = (
            (P_f - self.b.P_f_range[0])
            / (self.b.P_f_range[1] - self.b.P_f_range[0])
        )

        P_norm_ref = (
            (self.b.x_ref[3] - self.b.P_f_range[0])
            / (self.b.P_f_range[1] - self.b.P_f_range[0])
        )

        # Saturating site-benefit function
        prefab_benefit = (
            (1.0 - np.exp(-2.30 * P_norm))
            / (1.0 - np.exp(-2.30))
        )

        prefab_benefit_ref = (
            (1.0 - np.exp(-2.30 * P_norm_ref))
            / (1.0 - np.exp(-2.30))
        )

        # Factory overload activates only above saturation threshold
        prefab_overload = max(0.0, P_f - self.b.P_f_factory_saturation)
        prefab_overload = prefab_overload / (
            self.b.P_f_range[1] - self.b.P_f_factory_saturation
        )
        prefab_overload = prefab_overload ** 2

        # Rationalisation benefit and penalty
        panel_pen = max(0.0, R_p - self.b.R_p_knee) ** 2
        truss_pen = max(0.0, R_t - self.b.R_t_knee) ** 2

        # Effective class/type counts
        N_panel_classes = self.n_panel_classes(R_p)
        N_truss_types = self.n_truss_types(R_t)

        RI_panel = self._RI_p(R_p)
        RI_truss = self._RI_t(R_t)

        panel_rat = max(0.0, RI_panel - 1.0)
        truss_rat = max(0.0, RI_truss - 1.0)

        # Opening/detailing response
        OC_low = max(0.0, 1.0 - OC)
        OC_high = max(0.0, OC - 1.0)

        # Low OC is standardisation effort, not free removal of openings.
        OC_low_penalty = OC_low ** 2
        OC_high_penalty = OC_high + OC_high ** 2

        # Logistics interaction: high prefab + high rationalisation increases handling complexity
        logistics = P_norm ** 2 * (
            0.65
            + self.b.k_Rp_logistics_penalty * R_p
            + self.b.k_Rt_logistics_penalty * R_t
        )

        return {
            "R_p": R_p,
            "R_t": R_t,
            "OC": OC,
            "P_f": P_f,

            "P_norm": P_norm,
            "P_norm_ref": P_norm_ref,
            "dP_norm": P_norm - P_norm_ref,

            "prefab_benefit": prefab_benefit,
            "prefab_benefit_ref": prefab_benefit_ref,
            "d_prefab_benefit": prefab_benefit - prefab_benefit_ref,
            "prefab_overload": prefab_overload,

            "panel_pen": panel_pen,
            "truss_pen": truss_pen,

            "N_panel_classes": N_panel_classes,
            "N_truss_types": N_truss_types,
            "RI_panel": RI_panel,
            "RI_truss": RI_truss,
            "panel_rat": panel_rat,
            "truss_rat": truss_rat,

            "OC_low": OC_low,
            "OC_high": OC_high,
            "OC_low_penalty": OC_low_penalty,
            "OC_high_penalty": OC_high_penalty,

            "logistics": logistics,

            "panel_scheme_reporting": self.panel_scheme_from_Rp(R_p),
            "truss_scheme_reporting": self.truss_scheme_from_Rt(R_t),
        }

    # ─────────────────────────────────────────────────────────────────────
    # MATERIAL WEIGHT
    # ─────────────────────────────────────────────────────────────────────

    def calculate_material_weight(self, x):
        """
        CFS steel mass response.

        Steel mass is mostly fixed because the structural section and case-study
        geometry are fixed. The optimiser is not allowed to redesign member sizes.

        Changes are therefore small and represent:
        - offcut/waste reduction from repetition
        - small prefabrication precision saving
        - oversizing/grouping penalty from excessive rationalisation
        - opening/detailing steel effect
        """

        s = self._engineering_state(x)

        # Repetition reduces offcut and fabrication waste
        repetition_saving = (
            self.b.k_Rp_waste_saving * s["R_p"]
            + self.b.k_Rt_waste_saving * s["R_t"]
        )

        # Prefabrication improves cutting precision, but only slightly
        prefab_precision_saving = 0.012 * max(0.0, s["dP_norm"])

        # Excessive rationalisation causes grouping/rounding/oversizing
        rationalisation_oversizing = (
            self.b.k_Rp_oversize_penalty * s["panel_pen"]
            + self.b.k_Rt_oversize_penalty * s["truss_pen"]
        )

        # Low OC can require standardised lintel/jamb modules.
        # High OC causes extra local trimming/reinforcement.
        opening_mass_effect = (
            self.b.k_OC_low_material_penalty * s["OC_low"]
            + self.b.k_OC_high_material_penalty * s["OC_high_penalty"]
        )

        W = self.b.W_base * (
            1.0
            - repetition_saving
            - prefab_precision_saving
            + rationalisation_oversizing
            + opening_mass_effect
        )

        # Physical bound: CFS framing mass should not move unrealistically
        # because member section and building geometry are fixed.
        return float(np.clip(W, self.b.W_base * 0.94, self.b.W_base * 1.06))

    # ─────────────────────────────────────────────────────────────────────
    # TIME OBJECTIVE
    # ─────────────────────────────────────────────────────────────────────

    def calculate_time_components_raw(self, x):
        s = self._engineering_state(x)

        # Factory production time increases with prefabrication level.
        # Rationalisation helps repetition, but excessive rationalisation and
        # low-OC standardisation create coordination/fabrication burden.
        T_factory_production = self.b.T_factory_production_base * (
            1.0
            + self.b.k_prefab_factory_time_increase * s["P_norm"]
            - 0.020 * s["R_p"]
            - 0.010 * s["R_t"]
            + self.b.k_Rp_factory_penalty * s["panel_pen"]
            + self.b.k_Rt_factory_penalty * s["truss_pen"]
            + self.b.k_OC_low_factory_time * s["OC_low"]
            + 0.120 * s["OC_high_penalty"]
            + self.b.k_prefab_overload_time * s["prefab_overload"]
        )

        # Factory setup decreases with rationalisation but not indefinitely.
        T_factory_setup = self.b.T_factory_setup_base * (
            1.0
            - 0.22 * s["R_p"]
            - 0.12 * s["R_t"]
            + 0.45 * s["panel_pen"]
            + 0.35 * s["truss_pen"]
        )

        T_factory_setup = max(0.40 * self.b.T_factory_setup_base, T_factory_setup)

        # Site installation decreases with prefabrication, but logistics,
        # excessive rationalisation and opening complexity add time.
        T_site_installation = self.b.T_installation_base * (
            1.0
            - self.b.k_prefab_site_time_saving * s["prefab_benefit"]
            - self.b.k_Rp_site_saving * s["R_p"]
            - self.b.k_Rt_site_saving * s["R_t"]
            + 0.18 * s["panel_pen"]
            + 0.14 * s["truss_pen"]
            - 0.08 * s["OC_low"]
            + self.b.k_OC_high_time * s["OC_high_penalty"]
            + self.b.k_logistics_time * s["logistics"]
        )

        # Physical lower bound for site handling of 48 panels + 58 trusses
        T_site_installation = max(
            self.b.T_site_minimum_hrs,
            T_site_installation
        )

        T_delivery_logistics = self.b.T_delivery_logistics_base * (
            1.0
            + self.b.k_logistics_time * s["logistics"]
            + 0.03 * s["R_p"]
            + 0.02 * s["R_t"]
        )

        T_site_mobilisation = self.b.T_site_mobilisation_base

        T_weather = (
            self.b.waikato_weather_downtime_fraction
            * self.b.weather_shielding_factor
            * T_site_installation
        )

        T_total = (
            T_factory_production
            + T_factory_setup
            + T_delivery_logistics
            + T_site_mobilisation
            + T_site_installation
            + T_weather
        )

        return {
            "T_factory_production": T_factory_production,
            "T_factory_setup": T_factory_setup,
            "T_delivery_logistics": T_delivery_logistics,
            "T_site_mobilisation": T_site_mobilisation,
            "T_site_installation": T_site_installation,
            "T_weather": T_weather,
            "T_total": T_total,
        }

    def calculate_time_raw(self, x):
        return self.calculate_time_components_raw(x)["T_total"]

    # ─────────────────────────────────────────────────────────────────────
    # COST OBJECTIVE
    # ─────────────────────────────────────────────────────────────────────

    def calculate_cost_components_raw(self, x):
        s = self._engineering_state(x)
        W = self.calculate_material_weight(x)

        # Material cost
        C_steel = W * self.b.raw_steel_material_nzd_per_kg

        # Factory processing cost
        factory_processing_multiplier = (
            1.0
            + self.b.k_prefab_factory_cost_increase * s["P_norm"]
            + self.b.k_Rp_factory_penalty * s["R_p"]
            + self.b.k_Rt_factory_penalty * s["R_t"]
            + self.b.k_prefab_overload_cost * s["prefab_overload"]
            + self.b.k_OC_low_design_cost * s["OC_low_penalty"]
            + self.b.k_OC_high_cost * s["OC_high_penalty"]
        )

        C_factory_processing = (
            W
            * self.b.factory_processing_nzd_per_kg
            * factory_processing_multiplier
        )

        # Setup cost: rationalisation reduces number of families,
        # but excessive rationalisation creates grouping/jigging burden.
        C_setup_base = (
            self.b.baseline_panel_class_count * self.b.C_setup_per_panel_type
            + self.b.baseline_truss_type_count * self.b.C_setup_per_truss_type
            + self.b.C_jig_base
        )

        setup_saving = (
            self.b.k_Rp_setup_saving * s["R_p"]
            + self.b.k_Rt_setup_saving * s["R_t"]
        )

        setup_penalty = (
            0.40 * s["panel_pen"]
            + 0.30 * s["truss_pen"]
        )

        C_setup = C_setup_base * max(0.45, 1.0 - setup_saving + setup_penalty)

        # Site installation time used for labour cost
        T_components = self.calculate_time_components_raw(x)
        C_site_labour = (
            T_components["T_site_installation"]
            * self.b.site_labor_rate_nzd_per_hr
        )

        # Transport and logistics
        C_transport = (
            (W / 1000.0)
            * self.b.a4_default_transport_distance_km
            * self.b.transport_cost_nzd_per_tonne_km
            * (1.0 + 0.06 * s["P_norm"] + 0.025 * s["R_p"] + 0.020 * s["R_t"])
        )

        C_delivery_logistics = (
            self.b.T_delivery_logistics_base
            * self.b.delivery_logistics_rate_nzd_per_hr
            * (1.0 + self.b.k_logistics_cost * s["logistics"])
        )

        C_site_mobilisation = (
            self.b.T_site_mobilisation_base
            * self.b.site_mobilisation_rate_nzd_per_hr
        )

        # Opening/detailing cost
        C_opening_standardisation = (
            self.b.C_ref_direct_subtotal
            * self.b.k_OC_low_design_cost
            * s["OC_low"]
        )

        C_opening_complexity = (
            self.b.C_ref_direct_subtotal
            * self.b.k_OC_high_cost
            * s["OC_high_penalty"]
        )

        C_waste_handling = (
            (W * self.b.steel_waste_rate_ref / 1000.0)
            * self.b.steel_waste_disposal_nzd_per_tonne
        )

        direct = (
            C_steel
            + C_factory_processing
            + C_setup
            + C_site_labour
            + C_transport
            + C_delivery_logistics
            + C_site_mobilisation
            + C_opening_standardisation
            + C_opening_complexity
            + C_waste_handling
            + self.b.lifting_equipment_cost_nzd
        )

        overhead = self.b.overhead_fraction_ref * direct

        return {
            "C_steel": C_steel,
            "C_factory_processing": C_factory_processing,
            "C_setup": C_setup,
            "C_site_labour": C_site_labour,
            "C_transport": C_transport,
            "C_delivery_logistics": C_delivery_logistics,
            "C_site_mobilisation": C_site_mobilisation,
            "C_opening_standardisation": C_opening_standardisation,
            "C_opening_complexity": C_opening_complexity,
            "C_waste_handling": C_waste_handling,
            "C_overhead": overhead,
            "C_total": direct + overhead,
        }

    def calculate_cost_raw(self, x):
        return self.calculate_cost_components_raw(x)["C_total"]

    # ─────────────────────────────────────────────────────────────────────
    # CARBON OBJECTIVE
    # ─────────────────────────────────────────────────────────────────────

    def calculate_carbon_components_raw(self, x):
        s = self._engineering_state(x)
        W = self.calculate_material_weight(x)

        # A1-A3 steel
        CO2_A1_A3_steel = (
            W
            * self.b.steel_carbon_factor_a1_a3_kgco2e_per_kg
        )

        # Waste/offcut carbon
        waste_factor = (
            self.b.steel_waste_rate_ref
            * (
                1.0
                - self.b.k_Rp_waste_saving * s["R_p"]
                - self.b.k_Rt_waste_saving * s["R_t"]
                - self.b.k_prefab_waste_reduction * s["prefab_benefit"]
                + 0.04 * s["OC_high_penalty"]
            )
        )

        waste_factor = float(np.clip(waste_factor, 0.015, 0.080))

        CO2_waste = (
            W
            * waste_factor
            * self.b.steel_carbon_factor_a1_a3_kgco2e_per_kg
            if self.b.include_waste_carbon else 0.0
        )

        # A4 transport
        CO2_A4_transport = (
            (W / 1000.0)
            * self.b.a4_default_transport_distance_km
            * self.b.freight_kgco2e_per_tonne_km
            * (
                1.0
                + self.b.k_logistics_carbon * s["logistics"]
                + 0.025 * s["R_p"]
                + 0.020 * s["R_t"]
            )
        )

        # Factory energy
        T_comp = self.calculate_time_components_raw(x)

        CO2_factory_energy = (
            T_comp["T_factory_production"]
            * self.b.factory_average_power_kw
            * self.b.nz_grid_kgco2e_per_kwh
            if self.b.include_factory_energy_carbon else 0.0
        )

        # A5 site installation
        CO2_A5_installation = (
            self.b.envelope_area_m2
            * self.b.a5_installation_carbon_kgco2e_per_m2
            * (
                1.0
                - self.b.k_prefab_a5_carbon_saving * s["prefab_benefit"]
                + self.b.k_OC_high_carbon * s["OC_high_penalty"]
                + 0.015 * s["OC_low"]
            )
        )

        CO2_A5_installation = max(
            0.35
            * self.b.envelope_area_m2
            * self.b.a5_installation_carbon_kgco2e_per_m2,
            CO2_A5_installation
        )

        CO2_total = (
            CO2_A1_A3_steel
            + CO2_waste
            + CO2_A4_transport
            + CO2_factory_energy
            + CO2_A5_installation
        )

        return {
            "CO2_A1_A3_steel": CO2_A1_A3_steel,
            "CO2_waste": CO2_waste,
            "CO2_A4_transport": CO2_A4_transport,
            "CO2_factory_energy": CO2_factory_energy,
            "CO2_A5_installation": CO2_A5_installation,
            "CO2_total": CO2_total,
        }

    def calculate_carbon_raw(self, x):
        return self.calculate_carbon_components_raw(x)["CO2_total"]

    # ─────────────────────────────────────────────────────────────────────
    # ABSOLUTE OUTPUTS — NO S2 CALIBRATION SCALING
    # ─────────────────────────────────────────────────────────────────────

    def _ensure_refs(self):
        if self._C_ref is None:
            xr = self.b.x_ref
            self._C_ref = self.calculate_cost_raw(xr)
            self._CO2_ref = self.calculate_carbon_raw(xr)
            self._T_ref = self.calculate_time_raw(xr)

            assert np.isfinite(self._C_ref) and self._C_ref > 0.0
            assert np.isfinite(self._CO2_ref) and self._CO2_ref > 0.0
            assert np.isfinite(self._T_ref) and self._T_ref > 0.0

    def calculate_cost_abs(self, x):
        return self.calculate_cost_raw(x)

    def calculate_carbon_abs(self, x):
        return self.calculate_carbon_raw(x)

    def calculate_time_abs(self, x):
        return self.calculate_time_raw(x)

    def calculate_cost_components_abs(self, x):
        return self.calculate_cost_components_raw(x)

    def calculate_carbon_components_abs(self, x):
        return self.calculate_carbon_components_raw(x)

    def calculate_time_components_abs(self, x):
        return self.calculate_time_components_raw(x)

    # ─────────────────────────────────────────────────────────────────────
    # NORMALISED OBJECTIVES FOR NSGA-II AND POST-PROCESSING
    # ─────────────────────────────────────────────────────────────────────

    def calculate_cost(self, x):
        self._ensure_refs()
        return self.calculate_cost_abs(x) / self._C_ref

    def calculate_carbon(self, x):
        self._ensure_refs()
        return self.calculate_carbon_abs(x) / self._CO2_ref

    def calculate_time(self, x):
        self._ensure_refs()
        return self.calculate_time_abs(x) / self._T_ref
    
    # ─────────────────────────────────────────────────────────────────────
    # NZ FEASIBILITY / CONSTRUCTABILITY CONSTRAINTS
    # ─────────────────────────────────────────────────────────────────────

    def nz_constraint_metrics(self, x, CO2_abs=None):
        """
        Optimization-level NZ feasibility and constructability metrics.
        These are screening checks, not full structural/code compliance checks.
        """
        R_p, R_t, opening_complexity, P_f = x

        R_p = self._clip_Rp(R_p)
        R_t = self._clip_Rt(R_t)
        oc = self._clip_opening_complexity(opening_complexity)
        Pf = float(np.clip(P_f, self.b.P_f_range[0], self.b.P_f_range[1]))

        # Reporting categories only
        ps = self.panel_scheme_from_Rp(R_p)
        ts = self.truss_scheme_from_Rt(R_t)

        widths_mm = self.b.panel_width_schemes[ps]

        N_panel_classes = self.n_panel_classes(R_p)
        N_truss_types = self.n_truss_types(R_t)
        max_panel_width_m = max(widths_mm) / 1000.0
        panel_height_m = self.b.assumed_panel_height_m

        W_total = self.calculate_material_weight([R_p, R_t, oc, Pf])

        # Approximate maximum wall panel lift mass.
        # This is intentionally conservative but not a detailed lifting design.
        wall_steel_weight = W_total * self.b.wall_steel_weight_fraction
        avg_panel_mass = wall_steel_weight / self.b.num_wall_panels

        avg_width_m = np.mean(widths_mm) / 1000.0
        width_factor = max_panel_width_m / max(avg_width_m, 1e-9)

        panel_lift_mass_kg = avg_panel_mass * width_factor

        # Repetition score:
        # Higher = more repeated families, lower = more unique families.
        panel_repetition_score = 1.0 - (N_panel_classes - 1.0) / max(self.b.num_wall_panels - 1.0, 1.0)
        truss_repetition_score = 1.0 - (N_truss_types - 1.0) / max(self.b.num_roof_trusses - 1.0, 1.0)

        factory_repetition_score = min(panel_repetition_score, truss_repetition_score)

        if CO2_abs is None:
            CO2_abs = self.calculate_carbon_abs([R_p, R_t, oc, Pf])

        carbon_intensity_kgco2e_m2 = CO2_abs / self.b.footprint_area_m2

        P_norm = (Pf - self.b.P_f_range[0]) / (self.b.P_f_range[1] - self.b.P_f_range[0])

        return {
            "panel_scheme": ps,
            "truss_scheme": ts,
            "R_p_panel_rationalisation": R_p,
            "R_t_truss_rationalisation": R_t,
            "max_panel_width_m": max_panel_width_m,
            "panel_height_m": panel_height_m,
            "stud_spacing_mm": self.b.assumed_stud_spacing_mm,
            "steel_thickness_mm": self.b.assumed_steel_thickness_mm,
            "panel_lift_mass_kg": panel_lift_mass_kg,
            "unique_truss_spans": N_truss_types,
            "factory_repetition_score": factory_repetition_score,
            "opening_complexity_index": oc,
            "P_f_prefab_level": Pf,
            "P_norm": P_norm,
            "carbon_intensity_kgco2e_m2": carbon_intensity_kgco2e_m2,
        }

    def nz_constraint_report(self, x, C_abs=None, CO2_abs=None, T_abs=None):
        """
        Returns hard feasibility status and soft objective penalties.

        Hard constraints:
          - panel width
          - panel height
          - stud spacing
          - steel thickness
          - lift mass

        Soft constraints:
          - unique truss spans
          - factory repetition score
          - opening/detailing complexity
          - Waikato weather downtime
          - optional carbon cap scenario
        """
        if C_abs is None:
            C_abs = self.calculate_cost_abs(x)
        if CO2_abs is None:
            CO2_abs = self.calculate_carbon_abs(x)
        if T_abs is None:
            T_abs = self.calculate_time_abs(x)

        m = self.nz_constraint_metrics(x, CO2_abs=CO2_abs)

        hard_violations = []
        soft_violations = []

        hard_infeasible = False

        # ── HARD CONSTRAINTS ──────────────────────────────────────────────
        if m["max_panel_width_m"] > self.b.max_panel_width_m:
            hard_infeasible = True
            hard_violations.append("panel_width_gt_2.40m")

        if m["panel_height_m"] > self.b.max_panel_height_m:
            hard_infeasible = True
            hard_violations.append("panel_height_gt_3.00m")

        if m["stud_spacing_mm"] > self.b.max_stud_spacing_mm:
            hard_infeasible = True
            hard_violations.append("stud_spacing_gt_600mm")

        if m["steel_thickness_mm"] < self.b.min_steel_thickness_mm:
            hard_infeasible = True
            hard_violations.append(f"steel_thickness_lt_{self.b.min_steel_thickness_mm:.2f}mm")

        if m["panel_lift_mass_kg"] > self.b.max_lift_mass_kg:
            hard_infeasible = True
            hard_violations.append("panel_lift_mass_gt_2000kg")

        # ── SOFT CONSTRAINT PENALTIES ─────────────────────────────────────

        penalty_cost = 0.0
        penalty_carbon = 0.0
        penalty_time = 0.0

        # Unique truss spans: soft factory complexity penalty
        if m["unique_truss_spans"] > self.b.max_unique_truss_spans_soft:
            excess = (m["unique_truss_spans"] / self.b.max_unique_truss_spans_soft) - 1.0
            penalty_cost += 0.10 * excess
            penalty_time += 0.10 * excess
            soft_violations.append("unique_truss_spans_gt_25")

        # Factory repetition: lower repetition hurts factory efficiency
        if m["factory_repetition_score"] < self.b.min_factory_repetition_score:
            deficit = (self.b.min_factory_repetition_score - m["factory_repetition_score"]) / self.b.min_factory_repetition_score
            penalty_cost += 0.15 * deficit
            penalty_time += 0.10 * deficit
            soft_violations.append("factory_repetition_lt_0.65")

        # Opening/detailing complexity
        if m["opening_complexity_index"] > self.b.max_OCI_soft:
            excess = (m["opening_complexity_index"] / self.b.max_OCI_soft) - 1.0
            penalty_cost += 0.05 * excess
            penalty_carbon += 0.03 * excess
            penalty_time += 0.15 * excess
            soft_violations.append("OCI_gt_1.20")

        if self.b.apply_carbon_cap_penalty:
            if m["carbon_intensity_kgco2e_m2"] > self.b.carbon_cap_kgco2e_per_m2:
                excess = (m["carbon_intensity_kgco2e_m2"] / self.b.carbon_cap_kgco2e_per_m2) - 1.0
                penalty_carbon += 0.50 * excess
                soft_violations.append("carbon_cap_gt_50kgCO2e_m2")

        if hard_infeasible:
            penalty_cost += self.b.hard_constraint_penalty
            penalty_carbon += self.b.hard_constraint_penalty
            penalty_time += self.b.hard_constraint_penalty

        return {
            **m,
            "NZ_feasible": not hard_infeasible,
            "Hard_violations": "; ".join(hard_violations) if hard_violations else "None",
            "Soft_violations": "; ".join(soft_violations) if soft_violations else "None",
            "Penalty_cost": penalty_cost,
            "Penalty_carbon": penalty_carbon,
            "Penalty_time": penalty_time,
            "Total_constraint_penalty": penalty_cost + penalty_carbon + penalty_time,
        }

    def calculate_constrained_fitness(self, x):
        C_abs = self.calculate_cost_abs(x)
        CO2_abs = self.calculate_carbon_abs(x)
        T_abs = self.calculate_time_abs(x)

        r = self.nz_constraint_report(x, C_abs, CO2_abs, T_abs)

        self._ensure_refs()

        f_cost = C_abs / self._C_ref + r["Penalty_cost"]
        f_carbon = CO2_abs / self._CO2_ref + r["Penalty_carbon"]
        f_time = T_abs / self._T_ref + r["Penalty_time"]

        fitness = np.array([f_cost, f_carbon, f_time], dtype=float)

        if not np.isfinite(fitness).all():
            print("\n[NON-FINITE CONSTRAINED FITNESS]")
            print(f"  x = {x}")
            print(f"  C_abs = {C_abs}")
            print(f"  CO2_abs = {CO2_abs}")
            print(f"  T_abs = {T_abs}")
            print(f"  constraint_report = {r}")
            raise ValueError("Non-finite constrained fitness generated.")

        return float(f_cost), float(f_carbon), float(f_time)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: NSGA-II SETUP (refactored — 4 physical variables)
# ─────────────────────────────────────────────────────────────────────────────
def decode_individual(ind, b):
    """
    individual = [R_p, R_t, opening_complexity, P_f]
    """
    R_p = float(np.clip(float(ind[0]), b.R_p_range[0], b.R_p_range[1]))
    R_t = float(np.clip(float(ind[1]), b.R_t_range[0], b.R_t_range[1]))

    oc = float(np.clip(
        float(ind[2]),
        b.opening_complexity_range[0],
        b.opening_complexity_range[1]
    ))

    Pf = float(np.clip(
        float(ind[3]),
        b.P_f_range[0],
        b.P_f_range[1]
    ))

    return [R_p, R_t, oc, Pf]

def mixed_mutate(ind, b, indpb=0.30):
    if random.random() < indpb:
        ind[0] = float(np.clip(
            ind[0] + random.gauss(0, 0.12),
            b.R_p_range[0],
            b.R_p_range[1]
        ))

    if random.random() < indpb:
        ind[1] = float(np.clip(
            ind[1] + random.gauss(0, 0.12),
            b.R_t_range[0],
            b.R_t_range[1]
        ))

    if random.random() < indpb:
        ind[2] = float(np.clip(
            ind[2] + random.gauss(0, 0.08),
            b.opening_complexity_range[0],
            b.opening_complexity_range[1]
        ))

    if random.random() < indpb:
        ind[3] = float(np.clip(
            ind[3] + random.gauss(0, 0.08),
            b.P_f_range[0],
            b.P_f_range[1]
        ))

    return (ind,)

def mixed_crossover(ind1, ind2, alpha=0.5):
    """
    Continuous BLX-alpha crossover for [R_p, R_t, OC, P_f].
    """
    for i in range(4):
        if random.random() < 0.7:
            gamma = (1.0 + 2.0 * alpha) * random.random() - alpha

            u1 = (1.0 - gamma) * float(ind1[i]) + gamma * float(ind2[i])
            u2 = gamma * float(ind1[i]) + (1.0 - gamma) * float(ind2[i])

            ind1[i] = u1
            ind2[i] = u2

    return ind1, ind2

def lhs_unit_matrix(n_samples: int, n_vars: int, seed: int = None) -> np.ndarray:
    """
    Latin Hypercube Sampling matrix in [0, 1].
    Each variable column is stratified into n_samples intervals.
    """
    rng = np.random.default_rng(seed)
    H = np.zeros((n_samples, n_vars), dtype=float)

    for j in range(n_vars):
        perm = rng.permutation(n_samples)
        jitter = rng.random(n_samples)
        H[:, j] = (perm + jitter) / n_samples

    return H


def lhs_to_continuous(u: float, low: float, high: float) -> float:
    """
    Map LHS value in [0,1] to a continuous variable range.
    """
    return float(low + float(u) * (high - low))


def generate_lhs_designs(n_samples: int,
                         b: CFSProjectBaseline,
                         seed: int = 42) -> pd.DataFrame:
    """
    Generate an LHS design matrix for the current design vector:

        x = [R_p, R_t, opening_complexity_index, P_f]

    All four optimiser variables are continuous in this refactored model.
    panel_scheme and truss_scheme are added later only as reporting categories.
    """
    H = lhs_unit_matrix(n_samples=n_samples, n_vars=4, seed=seed)

    rows = []

    for i in range(n_samples):
        R_p = lhs_to_continuous(H[i, 0], b.R_p_range[0], b.R_p_range[1])
        R_t = lhs_to_continuous(H[i, 1], b.R_t_range[0], b.R_t_range[1])

        oc = lhs_to_continuous(
            H[i, 2],
            b.opening_complexity_range[0],
            b.opening_complexity_range[1]
        )

        pf = lhs_to_continuous(
            H[i, 3],
            b.P_f_range[0],
            b.P_f_range[1]
        )

        rows.append({
            "Sample_ID": i + 1,
            "R_p_panel_rationalisation": float(R_p),
            "R_t_truss_rationalisation": float(R_t),
            "opening_complexity_index": float(oc),
            "P_f_prefab_level": float(pf),
        })

    return pd.DataFrame(rows)


def generate_lhs_population(n_individuals: int,
                            b: CFSProjectBaseline,
                            seed: int = 42):
    """
    Generate DEAP initial population using LHS.
    """
    design_df = generate_lhs_designs(
        n_samples=n_individuals,
        b=b,
        seed=seed
    )

    population = []

    for _, row in design_df.iterrows():
        ind = creator.Individual([
            float(row["R_p_panel_rationalisation"]),
            float(row["R_t_truss_rationalisation"]),
            float(row["opening_complexity_index"]),
            float(row["P_f_prefab_level"])
        ])
        population.append(ind)

    return population


def audit_initial_population(population,
                             b: CFSProjectBaseline,
                             ts: str,
                             method_name: str = "lhs") -> pd.DataFrame:
    """
    Saves and prints generation-zero population coverage.
    """
    rows = []

    for i, ind in enumerate(population):
        x = decode_individual(ind, b)

        rows.append({
            "Individual": i + 1,
            "R_p_panel_rationalisation": round(float(x[0]), 5),
            "R_t_truss_rationalisation": round(float(x[1]), 5),
            "opening_complexity_index": round(float(x[2]), 5),
            "P_f_prefab_level": round(float(x[3]), 5),
        })

    df = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"initial_population_{method_name}_audit_{ts}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[INITIAL POPULATION AUDIT — {method_name.upper()}]")
    print(
        "\n  R_p_panel_rationalisation: "
        f"min={df['R_p_panel_rationalisation'].min():.3f}, "
        f"max={df['R_p_panel_rationalisation'].max():.3f}"
    )

    print(
        "  R_t_truss_rationalisation: "
        f"min={df['R_t_truss_rationalisation'].min():.3f}, "
        f"max={df['R_t_truss_rationalisation'].max():.3f}"
    )
    print(
        "\n  opening_complexity_index: "
        f"min={df['opening_complexity_index'].min():.3f}, "
        f"max={df['opening_complexity_index'].max():.3f}"
    )

    print(
        "  P_f_prefab_level: "
        f"min={df['P_f_prefab_level'].min():.3f}, "
        f"max={df['P_f_prefab_level'].max():.3f}"
    )

    print(f"[SAVED] {out_path}")

    return df

def setup_nsga2(b: CFSProjectBaseline):
    # Clear old DEAP creator classes if script is rerun in same Python session
    for attr in ["FitnessMin", "Individual"]:
        if hasattr(creator, attr):
            delattr(creator, attr)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    tb = base.Toolbox()

    # Random initialisation generators
    tb.register("R_p", random.uniform, b.R_p_range[0], b.R_p_range[1])
    tb.register("R_t", random.uniform, b.R_t_range[0], b.R_t_range[1])
    tb.register(
        "opening_complexity",
        random.uniform,
        b.opening_complexity_range[0],
        b.opening_complexity_range[1]
    )
    tb.register("P_f", random.uniform, b.P_f_range[0], b.P_f_range[1])

    tb.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (tb.R_p, tb.R_t, tb.opening_complexity, tb.P_f),
        n=1
    )

    tb.register("population", tools.initRepeat, list, tb.individual)

    # LHS initial population generator
    tb.register("lhs_population", generate_lhs_population, b=b)

    obj = CFSObjectiveFunctions(b)

    def evaluate(ind):
        x = decode_individual(ind, b)
        return obj.calculate_constrained_fitness(x)

    tb.register("evaluate", evaluate)

    LOW = [
        b.R_p_range[0],
        b.R_t_range[0],
        b.opening_complexity_range[0],
        b.P_f_range[0],
    ]

    UP = [
        b.R_p_range[1],
        b.R_t_range[1],
        b.opening_complexity_range[1],
        b.P_f_range[1],
    ]

    # Crossover and mutation for continuous bounded variables
    tb.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=LOW,
        up=UP,
        eta=15.0
    )

    tb.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=LOW,
        up=UP,
        eta=20.0,
        indpb=0.25
    )

    # NSGA-II environmental selection
    tb.register("select", tools.selNSGA2)

    # Safety check
    required = ["evaluate", "mate", "mutate", "select", "population", "lhs_population"]
    missing = [name for name in required if not hasattr(tb, name)]

    if missing:
        raise RuntimeError(f"Toolbox registration failed. Missing: {missing}")

    return tb, obj

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MONTE CARLO UNCERTAINTY PROPAGATION (NEW)
# ─────────────────────────────────────────────────────────────────────────────
def run_monte_carlo(pareto_df: pd.DataFrame,
                    baseline: CFSProjectBaseline,
                    n_runs: int = 1000,
                    seed: int = 42) -> dict:
    """
    Monte Carlo uncertainty propagation without S2 calibration scaling.
    S2 is used only as a paired benchmark under the same uncertain parameter sample.

    Important:
        No S2 calibration scaling is used. Each uncertain sample evaluates both the
        candidate solution and the S2 benchmark using the same perturbed parameter set.
        The paired comparison therefore measures robustness relative to S2 under the
        same uncertainty conditions.
    """

    rng = np.random.default_rng(seed)
    results = {}


    rep_solutions = pareto_df.head(6)

    for _, row in rep_solutions.iterrows():

        sid = row.get('Case', row.get('Solution_ID', 'Unknown'))

        x = [
            row['R_p_panel_rationalisation'],
            row['R_t_truss_rationalisation'],
            row['opening_complexity_index'],
            row['P_f_prefab_level']
        ]

        costs = []
        carbons = []
        times = []

        paired_cost_better = []
        paired_carbon_better = []
        paired_time_better = []
        paired_dominates_s2 = []
        for _ in range(n_runs):

            # Perturbed parameter set
            # Perturbed parameter set using validation-report ranges
            pb = CFSProjectBaseline()

            # Cost parameters
            pb.raw_steel_material_nzd_per_kg = rng.uniform(
                *pb.raw_steel_material_nzd_per_kg_range
            )

            pb.factory_processing_nzd_per_kg = rng.uniform(
                *pb.factory_processing_nzd_per_kg_range
            )

            pb.site_labor_rate_nzd_per_hr = rng.uniform(
                *pb.site_labor_rate_nzd_per_hr_range
            )

            pb.transport_cost_nzd_per_tonne_km = rng.uniform(
                *pb.transport_cost_nzd_per_tonne_km_range
            )

            pb.C_setup_per_panel_type = rng.uniform(
                *pb.C_setup_per_panel_type_range
            )

            pb.C_setup_per_truss_type = rng.uniform(
                *pb.C_setup_per_truss_type_range
            )

            # Carbon parameters
            pb.steel_carbon_factor_a1_a3_kgco2e_per_kg = rng.uniform(
                *pb.steel_carbon_factor_a1_a3_range
            )

            pb.a5_installation_carbon_kgco2e_per_m2 = rng.uniform(
                *pb.a5_installation_carbon_range
            )

            pb.freight_kgco2e_per_tonne_km = rng.uniform(
                *pb.freight_kgco2e_per_tonne_km_range
            )

            pb.a4_transport_carbon_kgco2e_per_kg = (
                pb.freight_kgco2e_per_tonne_km
                * pb.a4_default_transport_distance_km
                / 1000.0
            )

            # Prefabrication interaction coefficients

            pb.k_prefab_time_saving = rng.uniform(
                *pb.k_prefab_time_saving_range
            )

            pb.k_prefab_waste_reduction = rng.uniform(
                *pb.k_prefab_waste_reduction_range
            )

            pb.k_prefab_a5_carbon_saving = rng.uniform(
                *pb.k_prefab_a5_carbon_saving_range
            )

            # Logistics coefficients
            pb.k_logistics_cost = rng.uniform(
                *pb.k_logistics_cost_range
            )

            pb.k_logistics_time = rng.uniform(
                *pb.k_logistics_time_range
            )

            pb.k_logistics_carbon = rng.uniform(
                *pb.k_logistics_carbon_range
            )

            # Opening/detailing coefficients
            pb.k_opening_cost = rng.uniform(
                *pb.k_opening_cost_range
            )

            pb.k_opening_carbon = rng.uniform(
                *pb.k_opening_carbon_range
            )

            pb.k_opening_time = rng.uniform(
                *pb.k_opening_time_range
            )

            pb.lambda_opening_prefab_reduction = rng.uniform(
                *pb.lambda_opening_prefab_reduction_range
            )

            # Weather scenario uncertainty
            pb.waikato_weather_downtime_fraction = rng.uniform(
                *pb.waikato_weather_downtime_fraction_range
            )

            pb.weather_shielding_factor = rng.uniform(
                *pb.weather_shielding_factor_range
            )

            # Recompute S2 reference quantities after perturbing parameters.
            pb.recompute_references()

            of = CFSObjectiveFunctions(pb)

            C = of.calculate_cost_raw(x)
            E = of.calculate_carbon_raw(x)
            T = of.calculate_time_raw(x)

            # Paired S2 under same uncertain sample
            C_s2_sample = of.calculate_cost_raw(pb.x_ref)
            E_s2_sample = of.calculate_carbon_raw(pb.x_ref)
            T_s2_sample = of.calculate_time_raw(pb.x_ref)

            costs.append(C)
            carbons.append(E)
            times.append(T)

            paired_cost_better.append(C < C_s2_sample)
            paired_carbon_better.append(E < E_s2_sample)
            paired_time_better.append(T < T_s2_sample)
            paired_dominates_s2.append(
                (C <= C_s2_sample)
                and (E <= E_s2_sample)
                and (T <= T_s2_sample)
            )

        costs = np.asarray(costs)
        carbons = np.asarray(carbons)
        times = np.asarray(times)

        results[sid] = {
            'cost_p05': np.percentile(costs, 5),
            'cost_p50': np.percentile(costs, 50),
            'cost_p95': np.percentile(costs, 95),
            'cost_mean': np.mean(costs),
            'cost_std': np.std(costs),
            'cost_cv': np.std(costs) / np.mean(costs),

            'carbon_p05': np.percentile(carbons, 5),
            'carbon_p50': np.percentile(carbons, 50),
            'carbon_p95': np.percentile(carbons, 95),
            'carbon_mean': np.mean(carbons),
            'carbon_std': np.std(carbons),
            'carbon_cv': np.std(carbons) / np.mean(carbons),

            'time_p05': np.percentile(times, 5),
            'time_p50': np.percentile(times, 50),
            'time_p95': np.percentile(times, 95),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'time_cv': np.std(times) / np.mean(times),

            # Compared with fixed official S2
            'p_cost_below_official_S2': np.mean(costs < baseline.C_S2_reference),
            'p_carbon_below_official_S2': np.mean(carbons < baseline.CO2_S2_reference),
            'p_time_below_official_S2': np.mean(times < baseline.Time_S2_reference),

            # Compared with S2 under the same uncertain sample
            'p_cost_better_than_paired_S2': np.mean(paired_cost_better),
            'p_carbon_better_than_paired_S2': np.mean(paired_carbon_better),
            'p_time_better_than_paired_S2': np.mean(paired_time_better),
            'p_dominates_paired_S2': np.mean(paired_dominates_s2),
        }

        print(
            f"  MC {sid}: "
            f"Cost P50={results[sid]['cost_p50']:.0f} NZD "
            f"[{results[sid]['cost_p05']:.0f}–{results[sid]['cost_p95']:.0f}], "
            f"Time P50={results[sid]['time_p50']:.1f} hr "
            f"[{results[sid]['time_p05']:.1f}–{results[sid]['time_p95']:.1f}]"
        )

    return results


def build_mc_table(mc_results: dict, baseline: CFSProjectBaseline) -> pd.DataFrame:
    """
    Monte Carlo robustness table.

    Robust_score:
        lower = better

    It combines:
      1. expected normalized performance
      2. uncertainty penalty
      3. probability of outperforming paired S2
    """

    rows = []

    for sid, r in mc_results.items():

        mean_norm = (
            r['cost_mean'] / baseline.C_S2_reference
            + r['carbon_mean'] / baseline.CO2_S2_reference
            + r['time_mean'] / baseline.Time_S2_reference
        ) / 3.0

        uncertainty_norm = (
            r['cost_std'] / baseline.C_S2_reference
            + r['carbon_std'] / baseline.CO2_S2_reference
            + r['time_std'] / baseline.Time_S2_reference
        ) / 3.0

        paired_probability_benefit = (
            0.25 * r['p_cost_better_than_paired_S2']
            + 0.25 * r['p_carbon_better_than_paired_S2']
            + 0.25 * r['p_time_better_than_paired_S2']
            + 0.25 * r['p_dominates_paired_S2']
        )

        robust_score = (
            mean_norm
            + 0.50 * uncertainty_norm
            - 0.10 * paired_probability_benefit
        )

        rows.append({
            'Solution': sid,

            'Cost_p05_NZD': round(r['cost_p05'], 0),
            'Cost_p50_NZD': round(r['cost_p50'], 0),
            'Cost_p95_NZD': round(r['cost_p95'], 0),
            'Cost_CV_%': round(r['cost_cv'] * 100, 2),
            'P_cost_below_official_S2_%': round(r['p_cost_below_official_S2'] * 100, 1),
            'P_cost_better_than_paired_S2_%': round(r['p_cost_better_than_paired_S2'] * 100, 1),

            'Carbon_p05_kgCO2e': round(r['carbon_p05'], 0),
            'Carbon_p50_kgCO2e': round(r['carbon_p50'], 0),
            'Carbon_p95_kgCO2e': round(r['carbon_p95'], 0),
            'Carbon_CV_%': round(r['carbon_cv'] * 100, 2),
            'P_carbon_below_official_S2_%': round(r['p_carbon_below_official_S2'] * 100, 1),
            'P_carbon_better_than_paired_S2_%': round(r['p_carbon_better_than_paired_S2'] * 100, 1),

            'Time_p05_hrs': round(r['time_p05'], 1),
            'Time_p50_hrs': round(r['time_p50'], 1),
            'Time_p95_hrs': round(r['time_p95'], 1),
            'Time_CV_%': round(r['time_cv'] * 100, 2),
            'P_time_below_official_S2_%': round(r['p_time_below_official_S2'] * 100, 1),
            'P_time_better_than_paired_S2_%': round(r['p_time_better_than_paired_S2'] * 100, 1),

            'Mean_norm_performance': round(mean_norm, 4),
            'Uncertainty_penalty': round(uncertainty_norm, 4),
            'Paired_probability_benefit': round(paired_probability_benefit, 4),
            'Robust_score': round(robust_score, 4),
        })

    df = pd.DataFrame(rows)
    df['Robust_rank'] = df['Robust_score'].rank(method='min', ascending=True).astype(int)

    return df.sort_values('Robust_rank').reset_index(drop=True)

def build_component_breakdown_tables(rep_df: pd.DataFrame,
                                     obj_func: CFSObjectiveFunctions,
                                     baseline: CFSProjectBaseline,
                                     ts: str):

    cost_rows = []
    carbon_rows = []
    time_rows = []

    for _, row in rep_df.iterrows():
        label = row.get('Case', row.get('Solution_ID', 'Unknown'))

        x = [
            row['R_p_panel_rationalisation'],
            row['R_t_truss_rationalisation'],
            row['opening_complexity_index'],
            row['P_f_prefab_level']
        ]
        base_info = {
            'Solution': label,
            'R_p_panel_rationalisation': round(float(row['R_p_panel_rationalisation']), 4),
            'R_t_truss_rationalisation': round(float(row['R_t_truss_rationalisation']), 4),
            'panel_scheme': int(row['panel_scheme']),
            'truss_scheme': int(row['truss_scheme']),
            'opening_complexity_index': round(float(row['opening_complexity_index']), 3),
            'P_f_prefab_level': round(float(row['P_f_prefab_level']), 4),
        }

        cost_comps = obj_func.calculate_cost_components_abs(x)
        carbon_comps = obj_func.calculate_carbon_components_abs(x)
        time_comps = obj_func.calculate_time_components_abs(x)

        cost_row = base_info.copy()
        cost_row.update({k: round(v, 2) for k, v in cost_comps.items()})
        cost_row['Total_cost_NZD'] = round(cost_comps["C_total"], 2)
        cost_row['Cost_vs_S2_%'] = round(
            (cost_row['Total_cost_NZD'] / baseline.C_S2_reference - 1.0) * 100, 2
        )
        cost_rows.append(cost_row)

        carbon_row = base_info.copy()
        carbon_row.update({k: round(v, 2) for k, v in carbon_comps.items()})
        carbon_row['Total_carbon_kgCO2e'] = round(carbon_comps["CO2_total"], 2)
        carbon_row['Carbon_vs_S2_%'] = round(
            (carbon_row['Total_carbon_kgCO2e'] / baseline.CO2_S2_reference - 1.0) * 100, 2
        )
        carbon_rows.append(carbon_row)

        time_row = base_info.copy()
        time_row.update({k: round(v, 2) for k, v in time_comps.items()})
        time_row['Total_time_hours'] = round(time_comps["T_total"], 2)
        time_row['Time_vs_S2_%'] = round(
            (time_row['Total_time_hours'] / baseline.Time_S2_reference - 1.0) * 100, 2
        )
        time_rows.append(time_row)

    cost_df = pd.DataFrame(cost_rows)
    carbon_df = pd.DataFrame(carbon_rows)
    time_df = pd.DataFrame(time_rows)

    cost_path = TABLE_DIR / f"component_breakdown_cost_{ts}.csv"
    carbon_path = TABLE_DIR / f"component_breakdown_carbon_{ts}.csv"
    time_path = TABLE_DIR / f"component_breakdown_time_{ts}.csv"

    cost_df.to_csv(cost_path, index=False)
    carbon_df.to_csv(carbon_path, index=False)
    time_df.to_csv(time_path, index=False)

    print(f"[SAVED] Cost component breakdown:   {cost_path}")
    print(f"[SAVED] Carbon component breakdown: {carbon_path}")
    print(f"[SAVED] Time component breakdown:   {time_path}")

    return cost_df, carbon_df, time_df

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: ANALYSIS & ARCHETYPE CLUSTERING (k-means, FIXED)
# ─────────────────────────────────────────────────────────────────────────────
def kmeans_archetypes(pareto_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Robust K-means clustering using physical normalized objective values.
    Drops NaN/inf rows before clustering.
    """
    df = pareto_df.copy()

    cols = ['Cost_normalized', 'Carbon_normalized', 'Time_normalized']

    if len(df) == 0:
        raise ValueError("Pareto dataframe is empty before KMeans.")

    Zdf = df[cols].replace([np.inf, -np.inf], np.nan).astype(float)

    finite_mask = Zdf.notna().all(axis=1)
    n_bad = int((~finite_mask).sum())

    if n_bad > 0:
        print(f"[KMEANS WARNING] Dropping {n_bad} non-finite Pareto rows before clustering.")
        debug_path = TABLE_DIR / "debug_kmeans_nonfinite_rows.csv"
        df.loc[~finite_mask].to_csv(debug_path, index=False)
        print(f"[SAVED] {debug_path}")

    df = df.loc[finite_mask].copy().reset_index(drop=True)
    Z = Zdf.loc[finite_mask].values

    if len(df) == 0:
        raise ValueError("No finite Pareto rows available for KMeans.")

    if len(df) == 1:
        df['Cluster_ID'] = 0
        df['Archetype'] = "Single Feasible Solution"
        return df

    z_min = Z.min(axis=0)
    z_max = Z.max(axis=0)
    span = z_max - z_min
    span[span < 1e-12] = 1.0

    Zn = (Z - z_min) / span

    if not np.isfinite(Zn).all():
        raise ValueError("KMeans input still contains NaN/inf after cleaning.")

    n_unique = np.unique(np.round(Zn, 10), axis=0).shape[0]
    k = min(n_clusters, len(df), n_unique)

    if k <= 1:
        df['Cluster_ID'] = 0
        df['Archetype'] = "Collapsed Pareto Regime"
        return df

    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(Zn)
    centers = km.cluster_centers_

    cost_cluster = int(np.argmin(centers[:, 0]))
    carbon_cluster = int(np.argmin(centers[:, 1]))
    time_cluster = int(np.argmin(centers[:, 2]))

    target = np.array([0.5, 0.5, 0.5])
    balanced_cluster = int(np.argmin(np.linalg.norm(centers - target, axis=1)))

    cluster_name = {}
    cluster_name[cost_cluster] = "Low Cost"
    cluster_name[carbon_cluster] = "Low Carbon"
    cluster_name[time_cluster] = "Low Time"

    if balanced_cluster not in cluster_name:
        cluster_name[balanced_cluster] = "Balanced"

    for cluster_id in range(k):
        if cluster_id not in cluster_name:
            cluster_name[cluster_id] = "Transition"

    df['Cluster_ID'] = labels
    df['Archetype'] = [cluster_name[l] for l in labels]

    print(f"[KMEANS] Clustered {len(df)} finite Pareto rows into {k} clusters.")

    return df.reset_index(drop=True)

def build_pareto_degeneracy_report(pdf: pd.DataFrame, ts: str) -> pd.DataFrame:
    """
    Reports which optimizer variables collapse to constants on the Pareto front.
    This turns apparent degeneracy into an explicit thesis finding.
    """
    rows = []

    variables = [
        'R_p_panel_rationalisation',
        'R_t_truss_rationalisation',
        'panel_scheme',
        'truss_scheme',
        'opening_complexity_index',
        'P_f_prefab_level',
        'N_panel_classes',
        'N_truss_types'
    ]

    for var in variables:
        if var not in pdf.columns:
            continue

        unique_count = pdf[var].nunique()
        min_val = pdf[var].min()
        max_val = pdf[var].max()
        mean_val = pdf[var].mean() if pd.api.types.is_numeric_dtype(pdf[var]) else np.nan

        if unique_count == 1:
            status = "Collapsed / constant on Pareto front"
        else:
            status = "Active variation on Pareto front"

        rows.append({
            'Variable': var,
            'Unique_values_on_Pareto': unique_count,
            'Min': round(float(min_val), 4) if pd.api.types.is_numeric_dtype(pdf[var]) else min_val,
            'Max': round(float(max_val), 4) if pd.api.types.is_numeric_dtype(pdf[var]) else max_val,
            'Mean': round(float(mean_val), 4) if pd.api.types.is_numeric_dtype(pdf[var]) else "",
            'Status': status
        })

    out = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"pareto_degeneracy_report_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[PARETO DEGENERACY / VARIABLE COLLAPSE REPORT]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def build_variable_dominance_finding(pdf: pd.DataFrame,
                                      baseline: CFSProjectBaseline,
                                      ts: str) -> pd.DataFrame:
    """
    Thesis design variable dominance analysis.

    Documents which variables are:
      - Pareto-active: vary across the Pareto front and drive trade-offs
      - Pareto-collapsed: converge to one value, indicating a universal optimum

    A collapsed variable is not a failure. It is an actionable finding:
    the optimizer found that one value dominates across cost, carbon, and time.
    """

    findings = []

    var_configs = {
        'R_p_panel_rationalisation': {
            'range': baseline.R_p_range,
            'type': 'continuous',
            'interpretation': {
                'active': (
                    'Panel rationalisation intensity is Pareto-active. This means panel standardisation '
                    'creates genuine cost-carbon-time trade-offs through repetition benefits, transport/lifting '
                    'effects, and material/handling penalties.'
                ),
                'collapsed': (
                    'Panel rationalisation intensity collapses to a narrow range. This indicates that one '
                    'panel rationalisation level is consistently preferred for this case-study model.'
                )
            },
            'action': {
                'active': 'Use branch summary, TOPSIS, or stakeholder weighting to choose the preferred panel rationalisation level.',
                'collapsed': 'Report the selected R_p range as the case-study preferred panel rationalisation level.'
            }
        },

        'R_t_truss_rationalisation': {
            'range': baseline.R_t_range,
            'type': 'continuous',
            'interpretation': {
                'active': (
                    'Truss rationalisation intensity is Pareto-active. This means truss span-family '
                    'standardisation creates trade-offs between repetition benefits and possible span mismatch, '
                    'self-weight, transport, and detailing penalties.'
                ),
                'collapsed': (
                    'Truss rationalisation intensity collapses to a narrow range. This indicates that more '
                    'aggressive truss rationalisation does not provide enough benefit for this case-study geometry.'
                )
            },
            'action': {
                'active': 'Compare truss rationalisation using branch summary and stakeholder ranking.',
                'collapsed': 'Report the selected R_t range as a model-specific finding, not a universal rule.'
            }
        },
        'panel_scheme': {
            'range': baseline.panel_scheme_values,
            'type': 'discrete',
            'interpretation': {
                'active': (
                    'Panel rationalisation strategy creates genuine cost-carbon-time branches. '
                    'The preferred panel scheme depends on project priorities.'
                ),
                'collapsed': (
                    'One panel rationalisation scheme dominates the Pareto front. '
                    'This indicates a universal panel-width recommendation for this case study.'
                )
            },
            'action': {
                'active': 'Select panel scheme using TOPSIS/stakeholder weighting or branch summary.',
                'collapsed': 'Fix panel_scheme at the Pareto-selected value.'
            }
        },

        'truss_scheme': {
            'range': baseline.truss_scheme_values,
            'type': 'discrete',
            'interpretation': {
                'active': (
                    'Truss span rationalisation produces objective trade-offs. '
                    'Different truss schemes remain competitive depending on the selected objective priority.'
                ),
                'collapsed': (
                    'One truss scheme dominates the Pareto front. '
                    'If this is scheme 0, aggressive truss span reduction does not provide enough benefit '
                    'to offset grouping/oversizing/setup penalties for this building.'
                )
            },
            'action': {
                'active': 'Compare truss schemes using branch summary and stakeholder ranking.',
                'collapsed': 'Fix truss_scheme at the Pareto-selected value.'
            }
        },

        'opening_complexity_index': {
            'range': baseline.opening_complexity_range,
            'type': 'continuous',
            'interpretation': {
                'active': (
                    'Opening/detailing complexity creates objective trade-offs. '
                    'The preferred level depends on cost, carbon, and time priorities.'
                ),
                'collapsed': (
                    'Opening/detailing complexity converges to one value on the Pareto front. '
                    'If it converges to the lower bound, this is a universal design-for-manufacture finding: '
                    'standardising lintel modules, jamb details, service penetrations, and trimming details '
                    'improves cost, carbon, and time simultaneously.'
                )
            },
            'action': {
                'active': 'Treat opening/detailing complexity as a stakeholder-dependent detailing decision.',
                'collapsed': 'Adopt the Pareto-selected opening/detailing standardisation level.'
            }
        },

        'P_f_prefab_level': {
            'range': baseline.P_f_range,
            'type': 'continuous',
            'interpretation': {
                'active': (
                    'Prefabrication level is Pareto-active. '
                    'It drives the cost-carbon-time trade-off: higher prefabrication can reduce time '
                    'and site-stage carbon, but may increase factory/logistics cost beyond the optimal range.'
                ),
                'collapsed': (
                    'Prefabrication level converges to one value. '
                    'This indicates a universal optimum prefabrication level for the tested model and case study.'
                )
            },
            'action': {
                'active': 'Use TOPSIS, stakeholder weighting, or robustness ranking to select P_f.',
                'collapsed': 'Fix P_f at the Pareto-selected value.'
            }
        }
    }

    for var, config in var_configs.items():
        if var not in pdf.columns:
            continue

        series = pd.to_numeric(pdf[var], errors='coerce').dropna()

        if len(series) == 0:
            continue

        unique_vals = series.nunique()
        val_min = float(series.min())
        val_max = float(series.max())
        val_range = val_max - val_min

        if config['type'] == 'discrete':
            is_collapsed = unique_vals == 1
        else:
            full_range = config['range'][1] - config['range'][0]
            relative_range = val_range / full_range if full_range > 0 else 0.0
            is_collapsed = relative_range < 0.02

        status_key = 'collapsed' if is_collapsed else 'active'

        findings.append({
            'Variable': var,
            'Type': config['type'],
            'Unique_values_on_front': int(unique_vals),
            'Min_on_front': round(val_min, 4),
            'Max_on_front': round(val_max, 4),
            'Range_on_front': round(val_range, 4),
            'Status': (
                'Pareto-collapsed: universal optimum'
                if is_collapsed
                else 'Pareto-active: trade-off variable'
            ),
            'Thesis_finding': config['interpretation'][status_key],
            'Practitioner_action': config['action'][status_key]
        })

    df = pd.DataFrame(findings)

    out_path = TABLE_DIR / f"variable_dominance_finding_{ts}.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("THESIS FINDING: DESIGN VARIABLE DOMINANCE ANALYSIS")
   

    for _, row in df.iterrows():
        print(f"\n  {row['Variable']} [{row['Status']}]")
        print(f"  Range: {row['Min_on_front']} to {row['Max_on_front']}")
        print(f"  Finding: {row['Thesis_finding'][:160]}...")

    print(f"\n[SAVED] {out_path}")

    return df

def compute_igd(pareto_front_pts: np.ndarray,
                reference_pts: np.ndarray) -> float:
    """
    Finite-safe Inverted Generational Distance.
    If either point set has no valid finite points, returns NaN with warning.
    """
    pareto_front_pts = np.asarray(pareto_front_pts, dtype=float)
    reference_pts = np.asarray(reference_pts, dtype=float)

    pareto_front_pts = pareto_front_pts[np.isfinite(pareto_front_pts).all(axis=1)]
    reference_pts = reference_pts[np.isfinite(reference_pts).all(axis=1)]

    if len(pareto_front_pts) == 0:
        print("[IGD WARNING] NSGA-II Pareto point set has no finite rows.")
        return np.nan

    if len(reference_pts) == 0:
        print("[IGD WARNING] Grid reference Pareto point set has no finite rows.")
        return np.nan

    D = cdist(reference_pts, pareto_front_pts)

    if not np.isfinite(D).all():
        print("[IGD WARNING] Distance matrix contains NaN/inf.")
        return np.nan

    return float(D.min(axis=1).mean())

def approximate_hypervolume_from_fixed_samples(pts, ideal, ref, samples):
    """
    Fixed-sample Monte Carlo hypervolume for minimisation.
    Because the same samples are used every generation, archive HV should be
    non-decreasing when the archive is non-decreasing.
    """
    pts = np.asarray(pts, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    ref = np.asarray(ref, dtype=float)

    if len(pts) == 0:
        return 0.0

    pts = pts[np.all(pts <= ref, axis=1)]

    if len(pts) == 0:
        return 0.0

    dominated = np.zeros(len(samples), dtype=bool)

    for p in pts:
        dominated |= np.all(p <= samples, axis=1)

    return float(np.prod(ref - ideal) * dominated.mean())

def build_hypervolume_progress_table(hv_df: pd.DataFrame, ts: str) -> pd.DataFrame:
    out = hv_df.copy()

    out["HV_change"] = out["hypervolume"].diff()
    out["HV_percent_change"] = out["hypervolume"].pct_change() * 100.0

    out["HV_change"] = out["HV_change"].fillna(0.0)
    out["HV_percent_change"] = out["HV_percent_change"].fillna(0.0)

    out_path = TABLE_DIR / f"hypervolume_progress_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[HYPERVOLUME PROGRESS]")
    print(out.tail(10).to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def get_physical_objectives(ind, obj_func, baseline):
    """
    Physical normalized objectives for reporting/convergence.
    These exclude constraint penalties.
    """
    x = decode_individual(ind, baseline)

    C = obj_func.calculate_cost_abs(x) / baseline.C_S2_reference
    E = obj_func.calculate_carbon_abs(x) / baseline.CO2_S2_reference
    T = obj_func.calculate_time_abs(x) / baseline.Time_S2_reference

    vals = np.array([C, E, T], dtype=float)

    if not np.isfinite(vals).all():
        return None

    return tuple(vals)


def get_archive_physical_objective_array(archive, obj_func, baseline,
                                         max_reasonable_norm=1.50):
    """
    Convert DEAP archive into physical normalized objective array.
    Removes NaN, inf, and extreme objective outliers.
    """
    pts = []

    for ind in archive:
        vals = get_physical_objectives(ind, obj_func, baseline)

        if vals is None:
            continue

        vals = np.asarray(vals, dtype=float)

        if not np.isfinite(vals).all():
            continue

        if np.all(vals < max_reasonable_norm):
            pts.append(vals)

    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)

    return np.vstack(pts)

def get_pareto_front_df(df, obj_cols=('Cost_normalized', 'Carbon_normalized', 'Time_normalized')):
    """
    Finite-safe Pareto filter.
    Removes NaN/inf rows before dominance checking.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    out = df.copy()

    missing = [c for c in obj_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing objective columns for Pareto filtering: {missing}")

    # Replace inf with NaN, then remove invalid objective rows
    X = out[list(obj_cols)].replace([np.inf, -np.inf], np.nan)
    finite_mask = X.notna().all(axis=1)

    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        print(f"[PARETO FILTER WARNING] Dropping {n_bad} non-finite rows before Pareto filtering.")
        debug_path = TABLE_DIR / "debug_nonfinite_rows_in_pareto_filter.csv"
        out.loc[~finite_mask].to_csv(debug_path, index=False)
        print(f"[SAVED] {debug_path}")

    out = out.loc[finite_mask].copy().reset_index(drop=True)

    if len(out) == 0:
        print("[PARETO FILTER WARNING] No finite rows remain after filtering.")
        return out

    vals = out[list(obj_cols)].values.astype(float)
    n = len(out)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if np.all(vals[j] <= vals[i]) and np.any(vals[j] < vals[i]):
                dominated[i] = True
                break

    return out.loc[~dominated].copy().reset_index(drop=True)

def analyze_pareto_front(front, obj_func, baseline):
    rows = []
    for i, sol in enumerate(front):
        x = decode_individual(sol, baseline)
        R_p, R_t, opening_complexity, P_f = x

        # Reporting categories only
        ps = obj_func.panel_scheme_from_Rp(R_p)
        ts = obj_func.truss_scheme_from_Rt(R_t)

        N_panel_classes = obj_func.n_panel_classes(R_p)
        N_truss_types   = obj_func.n_truss_types(R_t)
        RI_panel = obj_func._RI_p(R_p)
        RI_truss = obj_func._RI_t(R_t)
        C_raw = obj_func.calculate_cost_abs(x)
        CO2_raw = obj_func.calculate_carbon_abs(x)
        T_raw = obj_func.calculate_time_abs(x)
        vals_check = np.array([C_raw, CO2_raw, T_raw], dtype=float)

        if not np.isfinite(vals_check).all():
            print("\n[ANALYZE WARNING] Non-finite objective detected.")
            print(f"  Solution index: {i}")
            print(f"  x = {x}")
            print(f"  Cost={C_raw}, Carbon={CO2_raw}, Time={T_raw}")
            continue
        constraint_report = obj_func.nz_constraint_report(x, C_raw, CO2_raw, T_raw)
        f_cost, f_carbon, f_time = sol.fitness.values
        obj_func._ensure_refs()

        C_S2_raw = obj_func._C_ref
        CO2_S2_raw = obj_func._CO2_ref
        T_S2_raw = obj_func._T_ref
        rows.append({
            'Solution_ID':       f"P{i+1}",
            'panel_scheme':      int(ps),
            'truss_scheme':      int(ts),
            'R_p_panel_rationalisation': round(R_p, 4),
            'R_t_truss_rationalisation': round(R_t, 4),
            'panel_widths_mm':   str(baseline.panel_width_schemes[int(ps)]),
            'truss_spans_m':     str(baseline.truss_span_schemes[int(ts)]),
            'N_panel_classes':   round(float(N_panel_classes), 3),
            'N_truss_types':     round(float(N_truss_types), 3),
            'RI_panel_norm':     round(RI_panel, 3),
            'RI_truss_norm':     round(RI_truss, 3),
            'opening_complexity_index': round(opening_complexity, 3),
            'N_openings_reference': int(baseline.N_openings_reference),
            'P_f_prefab_level':  round(P_f, 4),
            'Weight_kg':         round(obj_func.calculate_material_weight(x), 2),
            

            'Cost_NZD':          round(C_raw, 2),
            'Carbon_kgCO2e':     round(CO2_raw, 2),
            'Time_hours':        round(T_raw, 2),

            # Physical normalized values — use these for thesis plots and clustering
            'Cost_normalized':   C_raw / C_S2_raw,
            'Carbon_normalized': CO2_raw / CO2_S2_raw,
            'Time_normalized':   T_raw / T_S2_raw,

            # Optimizer fitness values — may include constraint penalties
            'Cost_fitness':   sol.fitness.values[0],
            'Carbon_fitness': sol.fitness.values[1],
            'Time_fitness':   sol.fitness.values[2],

            'Cost_vs_S2_%':      round((C_raw / C_S2_raw - 1.0) * 100, 2),
            'Carbon_vs_S2_%':    round((CO2_raw / CO2_S2_raw - 1.0) * 100, 2),
            'Time_vs_S2_%':      round((T_raw / T_S2_raw - 1.0) * 100, 2),
            'NZ_feasible': constraint_report['NZ_feasible'],
            'Hard_violations': constraint_report['Hard_violations'],
            'Soft_violations': constraint_report['Soft_violations'],
            'Penalty_cost': round(constraint_report['Penalty_cost'], 5),
            'Penalty_carbon': round(constraint_report['Penalty_carbon'], 5),
            'Penalty_time': round(constraint_report['Penalty_time'], 5),
            'Total_constraint_penalty': round(constraint_report['Total_constraint_penalty'], 5),

            'Max_panel_width_m': round(constraint_report['max_panel_width_m'], 3),
            'Panel_height_m': round(constraint_report['panel_height_m'], 3),
            'Stud_spacing_mm': round(constraint_report['stud_spacing_mm'], 1),
            'Steel_thickness_mm': round(constraint_report['steel_thickness_mm'], 2),
            'Panel_lift_mass_kg': round(constraint_report['panel_lift_mass_kg'], 2),
            'Factory_repetition_score': round(constraint_report['factory_repetition_score'], 3),
            'Unique_truss_spans': int(constraint_report['unique_truss_spans']),
            'Carbon_intensity_kgCO2e_m2': round(constraint_report['carbon_intensity_kgco2e_m2'], 3),
            })
    return pd.DataFrame(rows)
        
def select_representatives(df: pd.DataFrame) -> pd.DataFrame:
    def utopia_knee(d):
        cols = ['Cost_normalized','Carbon_normalized','Time_normalized']
        pts  = d[cols].values.astype(float)
        ut   = pts.min(axis=0); na = pts.max(axis=0)
        sp   = na - ut; sp[sp < 1e-12] = 1.0
        return d.index[np.argmin(np.linalg.norm((pts-ut)/sp, axis=1))]

    sel = [
        ("Min Cost",          df['Cost_NZD'].idxmin()),
        ("Min Carbon",        df['Carbon_kgCO2e'].idxmin()),
        ("Min Time",          df['Time_hours'].idxmin()),
        ("Balanced",          utopia_knee(df)),
        ("Highest Prefab",    df['P_f_prefab_level'].idxmax()),
        ("Lowest Prefab",     df['P_f_prefab_level'].idxmin()),
    ]
    rows = []
    seen = set()
    for name, idx in sel:
        if idx not in seen:
            seen.add(idx)
            r       = df.loc[idx].copy()
            r['Case'] = name
            rows.append(r)
    out = pd.DataFrame(rows)
    cols = [
        'Case', 'Solution_ID',
        'R_p_panel_rationalisation',
        'R_t_truss_rationalisation',
        'panel_scheme', 'truss_scheme',
        'panel_widths_mm', 'truss_spans_m',
        'N_panel_classes', 'N_truss_types',
        'RI_panel_norm', 'RI_truss_norm',
        'opening_complexity_index',
        'N_openings_reference',
        'P_f_prefab_level',
        'Weight_kg',

        'Cost_NZD', 'Carbon_kgCO2e', 'Time_hours',
        'Cost_vs_S2_%', 'Carbon_vs_S2_%', 'Time_vs_S2_%',

        'NZ_feasible',
        'Hard_violations',
        'Soft_violations',
        'Total_constraint_penalty',
        'Max_panel_width_m',
        'Panel_height_m',
        'Stud_spacing_mm',
        'Steel_thickness_mm',
        'Panel_lift_mass_kg',
        'Factory_repetition_score',
        'Unique_truss_spans',
        'Carbon_intensity_kgCO2e_m2',
    ]
    return out[[c for c in cols if c in out.columns]]

def compare_at_fixed_oc(obj_func, baseline, pareto_df, fixed_oc=1.0):
    """
    Recompute Pareto solution objectives with opening complexity fixed.

    Purpose:
    This diagnostic checks whether Pareto dominance is mainly caused by
    unrealistically low opening complexity. It keeps panel rationalisation,
    truss rationalisation, and prefabrication level unchanged, while fixing
    OC at the reference value.
    """

    rows = []

    for _, row in pareto_df.iterrows():
        x_fixed = [
            row["R_p_panel_rationalisation"],
            row["R_t_truss_rationalisation"],
            fixed_oc,
            row["P_f_prefab_level"],
        ]

        C = obj_func.calculate_cost_abs(x_fixed)
        E = obj_func.calculate_carbon_abs(x_fixed)
        T = obj_func.calculate_time_abs(x_fixed)

        rows.append({
            "Solution_ID": row["Solution_ID"],
            "OC_original": row["opening_complexity_index"],
            "OC_fixed": fixed_oc,
            "Cost_abs_fixed": C,
            "Carbon_abs_fixed": E,
            "Time_abs_fixed": T,
            "Cost_norm_fixed": C / baseline.C_S2_reference,
            "Carbon_norm_fixed": E / baseline.CO2_S2_reference,
            "Time_norm_fixed": T / baseline.Time_S2_reference,
            "Dominates_S2_fixed": (
                C < baseline.C_S2_reference
                and E < baseline.CO2_S2_reference
                and T < baseline.Time_S2_reference
            ),
        })

    return pd.DataFrame(rows)

def compute_improvements_vs_s2(rep_df: pd.DataFrame, ts: str) -> pd.DataFrame:
    """
    Reports actual percentage improvements of representative solutions
    relative to the S2 reference case.

    A solution is 'Pareto-improving vs S2' only if cost, carbon, and time
    are all less than or equal to S2.
    """
    results = []

    for _, row in rep_df.iterrows():
        case = row.get('Case', row.get('Solution_ID', 'Unknown'))

        cost_change = float(row['Cost_vs_S2_%'])
        carbon_change = float(row['Carbon_vs_S2_%'])
        time_change = float(row['Time_vs_S2_%'])

        all_better = (
            cost_change <= 0.0
            and carbon_change <= 0.0
            and time_change <= 0.0
        )

        any_better = (
            cost_change < 0.0
            or carbon_change < 0.0
            or time_change < 0.0
        )

        if all_better:
            comment = "Pareto-improving vs S2"
        elif any_better:
            comment = "Trade-off vs S2"
        else:
            comment = "Worse than S2 in all objectives"

        results.append({
            'Case': case,
            'Solution_ID': row.get('Solution_ID', ''),
            'Cost_vs_S2_%': round(cost_change, 2),
            'Carbon_vs_S2_%': round(carbon_change, 2),
            'Time_vs_S2_%': round(time_change, 2),
            'All_3_better_than_S2': all_better,
            'Any_objective_better_than_S2': any_better,
            'Comment': comment
        })

    out = pd.DataFrame(results)

    out_path = TABLE_DIR / f"representative_improvements_vs_S2_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[REPRESENTATIVE IMPROVEMENTS VS S2]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def compute_honest_improvements(rep_df: pd.DataFrame,
                                 baseline: CFSProjectBaseline,
                                 ts: str) -> pd.DataFrame:

    rows = []

    for _, sol in rep_df.iterrows():
        cost_pct = float(sol['Cost_vs_S2_%'])
        carbon_pct = float(sol['Carbon_vs_S2_%'])
        time_pct = float(sol['Time_vs_S2_%'])

        n_better = sum([
            cost_pct < 0.0,
            carbon_pct < 0.0,
            time_pct < 0.0
        ])

        if n_better == 3:
            classification = 'Dominates S2: better on cost, carbon, and time'
        elif n_better == 2:
            classification = 'Improves 2 of 3 objectives relative to S2'
        elif n_better == 1:
            classification = 'Specialist solution: improves 1 objective only'
        else:
            classification = 'No improvement over S2; diagnostic solution'

        if cost_pct > 0 and carbon_pct < 0 and time_pct < 0:
            trade_off_narrative = (
                f"Accepts +{cost_pct:.1f}% cost premium for "
                f"{abs(carbon_pct):.1f}% carbon and {abs(time_pct):.1f}% time reduction"
            )
        elif cost_pct < 0 and carbon_pct > 0 and time_pct < 0:
            trade_off_narrative = (
                f"Achieves {abs(cost_pct):.1f}% cost and {abs(time_pct):.1f}% time reduction "
                f"with +{carbon_pct:.1f}% carbon penalty"
            )
        elif cost_pct < 0 and carbon_pct > 0:
            trade_off_narrative = (
                f"Achieves {abs(cost_pct):.1f}% cost saving with "
                f"+{carbon_pct:.1f}% carbon penalty"
            )
        elif cost_pct < 0 and carbon_pct < 0 and time_pct < 0:
            trade_off_narrative = (
                f"Dominant solution: {abs(cost_pct):.1f}% cost, "
                f"{abs(carbon_pct):.1f}% carbon, and {abs(time_pct):.1f}% time reduction"
            )
        else:
            trade_off_narrative = (
                f"Cost {cost_pct:+.1f}%, Carbon {carbon_pct:+.1f}%, Time {time_pct:+.1f}%"
            )

        rows.append({
            'Case': sol.get('Case', sol.get('Solution_ID', '?')),
            'Solution_ID': sol.get('Solution_ID', '?'),
            'P_f': round(float(sol['P_f_prefab_level']), 3),
            'R_p_panel_rationalisation': round(float(sol['R_p_panel_rationalisation']), 4),
            'R_t_truss_rationalisation': round(float(sol['R_t_truss_rationalisation']), 4),
            'Panel_scheme': int(sol['panel_scheme']),
            'Truss_scheme': int(sol['truss_scheme']),
            'OC_index': round(float(sol['opening_complexity_index']), 3),
            'Cost_NZD': round(float(sol['Cost_NZD']), 0),
            'Carbon_kgCO2e': round(float(sol['Carbon_kgCO2e']), 0),
            'Time_hours': round(float(sol['Time_hours']), 1),
            'Cost_vs_S2_%': round(cost_pct, 2),
            'Carbon_vs_S2_%': round(carbon_pct, 2),
            'Time_vs_S2_%': round(time_pct, 2),
            'N_objectives_better_than_S2': n_better,
            'Classification': classification,
            'Trade_off_narrative': trade_off_narrative,
        })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("HONEST THESIS IMPROVEMENT NUMBERS")
  

    # Actual reductions only: negative values.
    cost_reductions = df.loc[df['Cost_vs_S2_%'] < 0, 'Cost_vs_S2_%']
    carbon_reductions = df.loc[df['Carbon_vs_S2_%'] < 0, 'Carbon_vs_S2_%']
    time_reductions = df.loc[df['Time_vs_S2_%'] < 0, 'Time_vs_S2_%']

    cost_penalties = df.loc[df['Cost_vs_S2_%'] > 0, 'Cost_vs_S2_%']
    carbon_penalties = df.loc[df['Carbon_vs_S2_%'] > 0, 'Carbon_vs_S2_%']
    time_penalties = df.loc[df['Time_vs_S2_%'] > 0, 'Time_vs_S2_%']

    def print_reduction_range(label, series):
        if len(series) == 0:
            print(f"  {label}: no reduction among representative solutions")
        else:
            small = abs(series.max())  # closest to zero
            large = abs(series.min())  # most negative
            print(f"  {label}: {small:.1f}% to {large:.1f}% reduction")

    def print_penalty_range(label, series):
        if len(series) == 0:
            print(f"  {label}: no penalty among representative solutions")
        else:
            print(f"  {label}: +{series.min():.1f}% to +{series.max():.1f}% penalty")

    print("\n[Reduction ranges]")
    print_reduction_range("Cost", cost_reductions)
    print_reduction_range("Carbon", carbon_reductions)
    print_reduction_range("Time", time_reductions)

    print("\n[Penalty ranges]")
    print_penalty_range("Cost", cost_penalties)
    print_penalty_range("Carbon", carbon_penalties)
    print_penalty_range("Time", time_penalties)

    dominant = df[df['N_objectives_better_than_S2'] == 3]
    improving_2plus = df[df['N_objectives_better_than_S2'] >= 2]

    print("\n[Classification counts]")
    print(df['Classification'].value_counts().to_string())

    if len(dominant) > 0:
        print("\n[Strongest thesis claim]")
        print(
            f"  {len(dominant)} representative solution(s) improve cost, carbon, and time "
            f"simultaneously relative to S2."
        )
    elif len(improving_2plus) > 0:
        print("\n[Strongest thesis claim]")
        print(
            f"  {len(improving_2plus)} representative solution(s) improve at least two objectives "
            f"relative to S2, but all require a trade-off in the remaining objective."
        )
    else:
        print("\n[WARNING]")
        print(
            "  No representative solution improves two or more objectives relative to S2. "
            "Do not claim broad improvement; frame the result as trade-off exploration."
        )

    print("\n[PER-SOLUTION BREAKDOWN]")
    print(
        df[
            [
                'Case',
                'Cost_vs_S2_%',
                'Carbon_vs_S2_%',
                'Time_vs_S2_%',
                'N_objectives_better_than_S2',
                'Trade_off_narrative'
            ]
        ].to_string(index=False)
    )

    out_path = TABLE_DIR / f"honest_improvements_vs_s2_{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")

    return df

def build_topsis_ranking(pdf: pd.DataFrame, ts: str) -> pd.DataFrame:
    """
    TOPSIS ranking for stakeholder-specific decision profiles.
    All objectives are minimised.
    """

    obj_cols = ['Cost_normalized', 'Carbon_normalized', 'Time_normalized']

    profiles = {
        'Cost_priority':   np.array([0.60, 0.20, 0.20]),
        'Carbon_priority': np.array([0.20, 0.60, 0.20]),
        'Time_priority':   np.array([0.20, 0.20, 0.60]),
        'Balanced':        np.array([1/3, 1/3, 1/3]),
    }

    rows = []

    X = pdf[obj_cols].values.astype(float)

    # Vector normalisation
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    Xn = X / denom

    for profile_name, w in profiles.items():
        V = Xn * w

        # Since all objectives are minimised:
        ideal_best = V.min(axis=0)
        ideal_worst = V.max(axis=0)

        d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

        closeness = d_worst / (d_best + d_worst + 1e-12)

        tmp = pdf.copy()
        tmp['TOPSIS_profile'] = profile_name
        tmp['TOPSIS_score'] = closeness
        tmp['TOPSIS_rank'] = tmp['TOPSIS_score'].rank(method='min', ascending=False).astype(int)

        best = tmp.sort_values('TOPSIS_rank').iloc[0]

        rows.append({
            'Profile': profile_name,
            'Best_solution': best['Solution_ID'],
            'R_p_panel_rationalisation': round(float(best['R_p_panel_rationalisation']), 4),
            'R_t_truss_rationalisation': round(float(best['R_t_truss_rationalisation']), 4),
            'panel_scheme': int(best['panel_scheme']),
            'truss_scheme': int(best['truss_scheme']),
            'opening_complexity_index': round(float(best['opening_complexity_index']), 3),
            'P_f_prefab_level': round(float(best['P_f_prefab_level']), 4),
            'Cost_NZD': round(float(best['Cost_NZD']), 2),
            'Carbon_kgCO2e': round(float(best['Carbon_kgCO2e']), 2),
            'Time_hours': round(float(best['Time_hours']), 2),
            'Cost_vs_S2_%': round(float(best['Cost_vs_S2_%']), 2),
            'Carbon_vs_S2_%': round(float(best['Carbon_vs_S2_%']), 2),
            'Time_vs_S2_%': round(float(best['Time_vs_S2_%']), 2),
            'TOPSIS_score': round(float(best['TOPSIS_score']), 4),
        })

    summary = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"topsis_stakeholder_ranking_{ts}.csv"
    summary.to_csv(out_path, index=False)

    print("\n[TOPSIS STAKEHOLDER RANKING]")
    print(summary.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return summary
    


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ENHANCED VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_pareto_3d(df, path):
    """
    3D Pareto plot showing discrete panel_scheme branches.

    Important:
    This does not pretend the front is a continuous smooth surface.
    Each panel_scheme is shown as a separate branch/curve.
    Point colour still represents prefabrication level P_f.
    """

    if df is None or len(df) == 0:
        print("[WARNING] Empty dataframe passed to plot_pareto_3d.")
        return

    required_cols = [
        'Cost_normalized',
        'Carbon_normalized',
        'Time_normalized',
        'P_f_prefab_level',
        'panel_scheme'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARNING] plot_pareto_3d missing columns: {missing}")
        return

    plot_df = df.copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=required_cols)

    if len(plot_df) == 0:
        print("[WARNING] No finite rows available for 3D Pareto plot.")
        return

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')

    scheme_colors = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728'
    }

    scheme_labels = {
        0: 'Scheme 0: baseline widths',
        1: 'Scheme 1: moderate rationalisation',
        2: 'Scheme 2: strong rationalisation',
        3: 'Scheme 3: aggressive rationalisation'
    }

    handles = []
    scatter_handle = None

    for scheme_id, grp in plot_df.groupby('panel_scheme'):
        grp = grp.sort_values('P_f_prefab_level')
        col = scheme_colors.get(int(scheme_id), 'grey')
        label = scheme_labels.get(int(scheme_id), f'Scheme {scheme_id}')

        scatter_handle = ax.scatter(
            grp['Cost_normalized'],
            grp['Carbon_normalized'],
            grp['Time_normalized'],
            c=grp['P_f_prefab_level'],
            s=45,
            alpha=0.80,
            cmap='plasma',
            edgecolors='none'
        )

        # Draw branch line only if enough points exist
        if len(grp) >= 2:
            ax.plot(
                grp['Cost_normalized'],
                grp['Carbon_normalized'],
                grp['Time_normalized'],
                color=col,
                lw=1.8,
                alpha=0.65
            )

        handles.append(
            plt.Line2D(
                [0], [0],
                color=col,
                lw=2,
                label=label
            )
        )

    # Mark S2 reference point
    ax.scatter(
        [1.0], [1.0], [1.0],
        c='red',
        s=160,
        marker='*',
        edgecolors='black',
        linewidths=0.8,
        zorder=10
    )

    handles.append(
        plt.Line2D(
            [0], [0],
            marker='*',
            color='w',
            markerfacecolor='red',
            markeredgecolor='black',
            markersize=14,
            label='S2 reference benchmark (not Pareto-optimal)'
        )
    )

    ax.set_xlabel('Cost / S2 cost', fontsize=11, fontweight='bold')
    ax.set_ylabel('Carbon / S2 carbon', fontsize=11, fontweight='bold')
    ax.set_zlabel('Time / S2 time', fontsize=11, fontweight='bold')

    ax.set_title(
        'Nondominated Pareto Front and S2 Benchmark\n'
        'Branch = panel rationalisation scheme; point colour = prefabrication level',
        fontsize=12,
        fontweight='bold'
    )

    ax.legend(
        handles=handles,
        loc='upper left',
        fontsize=8,
        frameon=True
    )

    if scatter_handle is not None:
        cbar = plt.colorbar(scatter_handle, ax=ax, pad=0.10, shrink=0.70)
        cbar.set_label('Prefabrication level $P_f$')

    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {path}")

def deduplicate_pareto_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes near-duplicate Pareto points for cleaner figures.
    Keeps thesis interpretation unchanged.
    """
    out = df.copy()

    out['P_f_round_plot'] = out['P_f_prefab_level'].round(3)
    out['OC_round_plot'] = out['opening_complexity_index'].round(3)
    out['Cost_round_plot'] = out['Cost_normalized'].round(5)
    out['Carbon_round_plot'] = out['Carbon_normalized'].round(5)
    out['Time_round_plot'] = out['Time_normalized'].round(5)

    out = out.drop_duplicates(
        subset=[
            'panel_scheme',
            'truss_scheme',
            'P_f_round_plot',
            'OC_round_plot',
            'Cost_round_plot',
            'Carbon_round_plot',
            'Time_round_plot'
        ]
    ).copy()

    drop_cols = [
        'P_f_round_plot',
        'OC_round_plot',
        'Cost_round_plot',
        'Carbon_round_plot',
        'Time_round_plot'
    ]

    return out.drop(columns=[c for c in drop_cols if c in out.columns]).reset_index(drop=True)

def plot_pareto_2d(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [('Cost_normalized','Carbon_normalized','Cost vs Carbon'),
             ('Cost_normalized','Time_normalized', 'Cost vs Time'),
             ('Carbon_normalized','Time_normalized','Carbon vs Time')]
    for ax, (x, y, t) in zip(axes, pairs):
        sc = ax.scatter(df[x], df[y], c=df['P_f_prefab_level'],
                        s=60, alpha=0.7, cmap='viridis')
        ax.set_xlabel(x.replace('_',' '), fontweight='bold')
        ax.set_ylabel(y.replace('_',' '), fontweight='bold')
        ax.set_title(t); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[2], label='P_f')
    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close(); print(f"[SAVED] {path}")

def plot_sensitivity(sens_df, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    y = np.arange(len(sens_df))
    objs = [('cost','Cost sensitivity','% change'),
            ('carbon','Carbon sensitivity','% change'),
            ('time','Time sensitivity','% change')]
    for ax, (key, title, xl) in zip(axes, objs):
        lo = sens_df[f'{key}_low_pct_change']
        hi = sens_df[f'{key}_high_pct_change']
        ax.barh(y - 0.2, lo, 0.38, label='Low case', color='#4472C4')
        ax.barh(y + 0.2, hi, 0.38, label='High case', color='#ED7D31')
        ax.set_yticks(y); ax.set_yticklabels(sens_df['variable'])
        ax.set_title(title); ax.set_xlabel(xl)
        ax.axvline(0, color='k', linewidth=0.8); ax.grid(True, alpha=0.3)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close(); print(f"[SAVED] {path}")

def plot_monte_carlo_uncertainty(mc_results: dict,
                                 baseline: CFSProjectBaseline,
                                 path):
    """Whisker plot showing P05-P95 ranges and S2 benchmark lines."""
    solutions = list(mc_results.keys())
    n = len(solutions)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    specs = [
        ("cost_p05", "cost_p50", "cost_p95", baseline.C_S2_reference, "Cost (NZD)"),
        ("carbon_p05", "carbon_p50", "carbon_p95", baseline.CO2_S2_reference, "CFS framing A1–A5 carbon (kgCO₂e)"),
        ("time_p05", "time_p50", "time_p95", baseline.Time_S2_reference, "Time (process-hours)"),
    ]

    for ax, (k05, k50, k95, s2_value, ylabel) in zip(axes, specs):
        p05 = [mc_results[s][k05] for s in solutions]
        p50 = [mc_results[s][k50] for s in solutions]
        p95 = [mc_results[s][k95] for s in solutions]

        ax.bar(x, p50, alpha=0.70, label="P50")
        for i in range(n):
            ax.plot([i, i], [p05[i], p95[i]], "k-", linewidth=2)
            ax.plot([i - 0.1, i + 0.1], [p05[i], p05[i]], "k-", linewidth=1.4)
            ax.plot([i - 0.1, i + 0.1], [p95[i], p95[i]], "k-", linewidth=1.4)

        ax.axhline(
            s2_value,
            linestyle="--",
            linewidth=1.4,
            label="Model-calculated S2 benchmark"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(solutions, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].legend(fontsize=8)

    plt.suptitle(
        "Monte Carlo uncertainty for representative Pareto solutions\n"
        "S2 is a model-calculated benchmark, not measured commercial data",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")

def plot_nsga_vs_grid_overlay(nsga_df: pd.DataFrame,
                              grid_df: pd.DataFrame,
                              path):
    """
    NSGA-II vs grid-reference overlay in the normalised cost-carbon plane.

    x-axis = normalised CFS framing process cost
    y-axis = normalised CFS framing A1-A5 carbon
    colour = normalised CFS framing process-hours

    This is a 2D projection of the three-objective Pareto result.
    """

    required_cols = [
        "Cost_normalized",
        "Carbon_normalized",
        "Time_normalized"
    ]

    for col in required_cols:
        if col not in nsga_df.columns:
            raise KeyError(f"NSGA-II dataframe is missing required column: {col}")
        if col not in grid_df.columns:
            raise KeyError(f"Grid dataframe is missing required column: {col}")

    # Clean plotting data
    nsga_plot = nsga_df[required_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    grid_plot = grid_df[required_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if nsga_plot.empty:
        raise ValueError("NSGA-II plotting dataframe is empty after removing non-finite values.")

    if grid_plot.empty:
        raise ValueError("Grid plotting dataframe is empty after removing non-finite values.")

    # Shared colour scale for time
    time_all = pd.concat(
        [nsga_plot["Time_normalized"], grid_plot["Time_normalized"]],
        ignore_index=True
    )

    norm = plt.Normalize(
        vmin=float(time_all.min()),
        vmax=float(time_all.max())
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    # Grid-reference front
    grid_scatter = ax.scatter(
        grid_plot["Cost_normalized"],
        grid_plot["Carbon_normalized"],
        c=grid_plot["Time_normalized"],
        cmap="viridis",
        norm=norm,
        s=35,
        alpha=0.35,
        marker="s",
        label="Grid reference Pareto front"
    )

    # NSGA-II Pareto front
    nsga_scatter = ax.scatter(
        nsga_plot["Cost_normalized"],
        nsga_plot["Carbon_normalized"],
        c=nsga_plot["Time_normalized"],
        cmap="viridis",
        norm=norm,
        s=45,
        alpha=0.85,
        marker="o",
        label="NSGA-II Pareto front"
    )

    # S2 reference benchmark
    ax.scatter(
        [1.0],
        [1.0],
        s=180,
        marker="*",
        color="green",
        edgecolors="black",
        linewidths=0.8,
        label="Model-calculated S2 benchmark",
        zorder=10
    )

    # S2 reference lines
    ax.axvline(1.0, linestyle="--", linewidth=0.8, alpha=0.45)
    ax.axhline(1.0, linestyle="--", linewidth=0.8, alpha=0.45)

    cbar = plt.colorbar(nsga_scatter, ax=ax)
    cbar.set_label(r"Normalised CFS framing process-hours, $\bar{T}$")

    ax.set_xlabel(r"Normalised CFS framing process cost, $\bar{C}$")
    ax.set_ylabel(r"Normalised CFS framing A1--A5 carbon, $\bar{E}$")

    # Leave title blank for thesis use; use the Word caption instead.
    ax.set_title("")

    ax.grid(True, alpha=0.30)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")

def plot_opening_interaction(obj_func, baseline, path):
    """Plot effect of opening/detailing complexity × prefab level."""
    pf = np.linspace(0.50, 0.90, 100)
    oc_cases = [0.80, 1.00, 1.15, 1.30]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = [f'Opening complexity={oc:.2f}' for oc in oc_cases]

    for ax, key, ylabel in zip(
        axes,
        ['calculate_cost', 'calculate_carbon', 'calculate_time'],
        ['Normalised cost', 'Normalised carbon', 'Normalised time']
    ):
        for oc, col, lbl in zip(oc_cases, colors, labels):
            vals = [
                getattr(obj_func, key)([baseline.x_ref[0], baseline.x_ref[1], oc, p])
                for p in pf
            ]
            ax.plot(pf, vals, linewidth=2, color=col, label=lbl)

        ax.axvline(
            baseline.x_ref[3],
            linestyle='--',
            linewidth=1.5,
            color='grey',
            label=f'S2 P_f={baseline.x_ref[3]:.2f}'
        )
        ax.set_xlabel('Prefabrication Level P_f')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_title('Cost vs Opening Complexity × Prefab')
    axes[1].set_title('Carbon vs Opening Complexity × Prefab')
    axes[2].set_title('Time vs Opening Complexity × Prefab')

    fig.suptitle(
        'Opening/Detailing Complexity × Prefabrication Interaction '
        '(S2 R_p/R_t reference condition)',
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")

def plot_parallel_coordinates(df, path):
    from matplotlib.lines import Line2D
    cols = [
        'R_p_panel_rationalisation',
        'R_t_truss_rationalisation',
        'panel_scheme',
        'truss_scheme',
        'N_panel_classes',
        'N_truss_types',
        'RI_panel_norm',
        'RI_truss_norm',
        'opening_complexity_index',
        'P_f_prefab_level',
        'Cost_normalized',
        'Carbon_normalized',
        'Time_normalized'
    ]
    varying = [c for c in cols if df[c].nunique() > 1 and df[c].max()-df[c].min() > 1e-10]
    ndf = df.copy()
    for c in varying:
        mn,mx = df[c].min(), df[c].max()
        ndf[c] = (df[c]-mn)/(mx-mn)

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(varying))
    for _, row in ndf.iterrows():
        ax.plot(
            x,
            row[varying].values,
            color='lightgray',
            lw=0.5,
            alpha=0.08,
            zorder=1
        )

    # Representative solutions highlighted on the parallel-coordinates plot.
    # Balanced / knee solution = closest solution to the normalised utopia point.

    obj_cols = ['Cost_normalized', 'Carbon_normalized', 'Time_normalized']

    pts = df[obj_cols].values.astype(float)

    utopia = pts.min(axis=0)
    nadir = pts.max(axis=0)

    span = nadir - utopia
    span[span < 1e-12] = 1.0

    balanced_idx = df.index[
        np.argmin(
            np.linalg.norm((pts - utopia) / span, axis=1)
        )
    ]

    rep_idx = {
        'Min Cost': df['Cost_normalized'].idxmin(),
        'Min Carbon': df['Carbon_normalized'].idxmin(),
        'Min Time': df['Time_normalized'].idxmin(),
        'Balanced / knee': balanced_idx,
    }

    cmap = {
        'Min Cost': 'tab:blue',
        'Min Carbon': 'tab:green',
        'Min Time': 'tab:red',
        'Balanced / knee': 'tab:purple',
    }

    handles = []

    for lbl, idx in rep_idx.items():
        ax.plot(
            x,
            ndf.loc[idx, varying].values,
            color=cmap[lbl],
            lw=3.0,
            alpha=0.98,
            zorder=4
        )

        ax.scatter(
            x,
            ndf.loc[idx, varying].values,
            color=cmap[lbl],
            s=18,
            zorder=5
        )

        handles.append(
            Line2D(
                [0],
                [0],
                color=cmap[lbl],
                lw=3.0,
                marker='o',
                markersize=4,
                label=lbl
            )
        )

    ax.set_xticks(x); ax.set_xticklabels(varying, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Normalised value'); ax.grid(True, alpha=0.3)
    ax.set_title('Parallel Coordinates — Pareto Solutions\n(Grey = all; coloured = key representatives)',
                 fontsize=13, fontweight='bold')
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01,1.0))
    print("\n[OPTIMIZER VARIABLE DISTRIBUTION ON PARETO FRONT]")

    for col in ['panel_scheme', 'truss_scheme']:
        print(f"  {col} reporting category: {dict(df[col].value_counts().sort_index())}")

    for col in ['R_p_panel_rationalisation', 'R_t_truss_rationalisation']:
        if col in df.columns:
            print(
                f"  {col}: "
                f"min={df[col].min():.3f}, "
                f"max={df[col].max():.3f}, "
                f"mean={df[col].mean():.3f}"
            )

    constant_notes = []

    if 'opening_complexity_index' in df.columns:
        print(
            "  opening_complexity_index: "
            f"min={df['opening_complexity_index'].min():.3f}, "
            f"max={df['opening_complexity_index'].max():.3f}, "
            f"mean={df['opening_complexity_index'].mean():.3f}"
        )
    else:
        print("  [WARNING] opening_complexity_index column not found in Pareto dataframe")

    if 'opening_complexity_index' in df.columns and df['opening_complexity_index'].nunique() == 1:
        constant_notes.append(
            f"opening_complexity_index excluded: constant at {df['opening_complexity_index'].iloc[0]:.2f}"
        )

    if 'truss_scheme' in df.columns and df['truss_scheme'].nunique() == 1:
        constant_notes.append(
            f"truss_scheme excluded/constant: scheme {int(df['truss_scheme'].iloc[0])}"
        )

    if constant_notes:
        ax.text(
            0.01,
            -0.22,
            "Note: " + "; ".join(constant_notes),
            transform=ax.transAxes,
            fontsize=9,
            ha='left',
            va='top'
        )
    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close(); print(f"[SAVED] {path}")

def plot_cost_breakeven(obj_func, baseline, path):
    pf = np.linspace(0.50, 0.90, 200)
    xr = baseline.x_ref

    costs = [
        obj_func.calculate_cost_abs([xr[0], xr[1], xr[2], p])
        for p in pf
    ]
    times = [
        obj_func.calculate_time_abs([xr[0], xr[1], xr[2], p])
        for p in pf
    ]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(pf, costs, 'b-', lw=2.5, label='Cost (NZD)')
    ax2.plot(pf, times, 'r--', lw=2.5, label='Time (hours)')

    mc = pf[np.argmin(costs)]
    mt = pf[np.argmin(times)]

    c_min = min(costs)
    c_s2 = obj_func.calculate_cost_abs([xr[0], xr[1], xr[2], xr[3]])
    delta_s2_min = c_s2 - c_min
    delta_s2_min_pct = delta_s2_min / c_min * 100

    ax1.axvline(mc, color='b', ls=':', lw=1.5, label=f'Min cost P_f={mc:.2f}')
    ax1.axvline(mt, color='r', ls='-.', lw=1.5, label=f'Min time P_f={mt:.2f}')
    ax1.axvline(xr[3], color='grey', ls='--', lw=1.5, label=f'S2 reference P_f={xr[3]:.2f}')

    ax1.set_xlabel('Prefabrication Level (P_f)', fontsize=12)
    ax1.set_ylabel('Cost (NZD)', fontsize=12)
    ax2.set_ylabel('Time (hours)', fontsize=12)

    ax1.set_title(
        'Cost and Schedule vs Prefabrication Level\n'
        '(R_p=0, R_t=0, opening_complexity=1.00 — S2 benchmark condition)',
        fontsize=11
    )

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc='best', fontsize=9)

    ax1.grid(True, alpha=0.3)
    ax1.annotate(
        f"S2 is NZ${delta_s2_min:.0f} above min cost\n({delta_s2_min_pct:.2f}%)",
        xy=(xr[3], c_s2),
        xytext=(xr[3] + 0.035, c_s2 + 250),
        arrowprops=dict(arrowstyle='->', linewidth=1.0),
        fontsize=9)

    ax1.scatter([mc], [c_min], s=45, zorder=5)
    ax1.scatter([xr[3]], [c_s2], s=45, zorder=5)
    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")


def _find_column(df, candidates, label):
    """
    Finds the first available column from a list of candidate names.
    This avoids breaking the code if your dataframe uses slightly different names.
    """
    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"Could not find column for {label}. "
        f"Tried: {candidates}. "
        f"Available columns are: {list(df.columns)}"
    )


def _safe_columns_for_design_variables(df):
    """
     design variables:
        R_p = panel rationalisation index
        R_t = truss rationalisation index
        OC  = opening complexity
        P_f = prefabrication level
    """

    rp_col = _find_column(
        df,
        [
            "R_p",
            "Rp",
            "RP",
            "R_p_panel_rationalisation",
            "R_p_panel_rationalization",
            "panel_repetition_index",
            "panel_RI",
            "RI_panel",
            "panel_rationalisation_index",
            "panel_rationalization_index",
        ],
        "panel repetition index R_p"
    )

    rt_col = _find_column(
        df,
        [
            "R_t",
            "Rt",
            "RT",
            "R_t_truss_rationalisation",
            "R_t_truss_rationalization",
            "truss_repetition_index",
            "truss_RI",
            "RI_truss",
            "truss_rationalisation_index",
            "truss_rationalization_index",
        ],
        "truss repetition index R_t"
    )

    oc_col = _find_column(
        df,
        [
            "opening_complexity_index",
            "OC",
            "opening_complexity",
            "OC_index",
        ],
        "opening complexity OC"
    )

    pf_col = _find_column(
        df,
        [
            "P_f_prefab_level",
            "P_f",
            "Pf",
            "prefabrication_level",
            "prefab_level",
        ],
        "prefabrication level P_f"
    )

    return [rp_col, rt_col, oc_col, pf_col]


def _objective_columns(df):
    """
    Use normalised objectives for fair comparison.
    """
    objs = [
        "Cost_normalized",
        "Carbon_normalized",
        "Time_normalized",
    ]

    missing = [c for c in objs if c not in df.columns]
    if missing:
        raise KeyError(f"Missing objective columns: {missing}")

    return objs


def plot_design_variable_tornado_from_grid(gdf: pd.DataFrame, path):
    """
    Tornado-style screening plot for:
        R_p, R_t, OC, P_f

    This reports screening-level effect range, not true causal sensitivity.
    """

    df = gdf.copy()
    design_vars = _safe_columns_for_design_variables(df)
    objectives = _objective_columns(df)

    rows = []

    for var in design_vars:
        temp = df.copy()

        # Round continuous variables for stable grouping.
        if temp[var].dtype.kind in "fc":
            temp[var] = temp[var].round(3)

        grouped = temp.groupby(var)[objectives].mean()

        for obj in objectives:
            effect_range = grouped[obj].max() - grouped[obj].min()
            best_level = grouped[obj].idxmin()
            worst_level = grouped[obj].idxmax()

            rows.append({
                "variable": var,
                "objective": obj,
                "effect_range_normalized": effect_range,
                "effect_range_percent_S2": effect_range * 100.0,
                "best_level": best_level,
                "worst_level": worst_level,
            })

    eff = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    objective_titles = {
        "Cost_normalized": "Cost",
        "Carbon_normalized": "Carbon",
        "Time_normalized": "Time",
    }

    for ax, obj in zip(axes, objectives):
        sub = eff[eff["objective"] == obj].copy()
        sub = sub.sort_values("effect_range_percent_S2", ascending=True)

        ax.barh(
            sub["variable"],
            sub["effect_range_percent_S2"]
        )

        ax.set_title(f"{objective_titles[obj]} effect range")
        ax.set_xlabel("Objective range (% of S2)")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        r"Screening-level design-variable effects: $R_p$, $R_t$, $OC$, and $P_f$",
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")

    return eff


def plot_repetition_index_heatmaps(gdf: pd.DataFrame, path, aggregation="min"):
    """
    Heatmaps of R_p × R_t versus objective performance.

    R_p = panel repetition/rationalisation index
    R_t = truss repetition/rationalisation index

    aggregation='min' means:
        for each R_p/R_t combination, show the best achievable objective
        after varying OC and P_f.
    """

    df = gdf.copy()

    rp_col, rt_col, oc_col, pf_col = _safe_columns_for_design_variables(df)

    required = [
        rp_col,
        rt_col,
        "Cost_normalized",
        "Carbon_normalized",
        "Time_normalized",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for R_p × R_t heatmap: {missing}")

    # Round to keep heatmap readable if R_p/R_t are floating-point values.
    df[rp_col] = df[rp_col].round(3)
    df[rt_col] = df[rt_col].round(3)

    objectives = [
        ("Cost_normalized", "Best cost / S2"),
        ("Carbon_normalized", "Best carbon / S2"),
        ("Time_normalized", "Best time / S2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (obj, title) in zip(axes, objectives):
        pivot = df.pivot_table(
            values=obj,
            index=rp_col,
            columns=rt_col,
            aggfunc=aggregation
        )

        im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))

        ax.set_xticklabels([f"{v:.3f}" for v in pivot.columns], rotation=45)
        ax.set_yticklabels([f"{v:.3f}" for v in pivot.index])

        ax.set_xlabel(r"Truss repetition index, $R_t$")
        ax.set_ylabel(r"Panel repetition index, $R_p$")
        ax.set_title(title)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7
                    )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(obj)

    fig.suptitle(
        r"$R_p \times R_t$ objective heatmaps"
        "\nCell value = best objective after varying opening complexity and prefabrication level",
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")


def plot_main_effects_from_grid(gdf: pd.DataFrame, path):
    """
    Main-effects plot for:
        R_p, R_t, OC, P_f

    Use grid_all, not only the Pareto front.
    """

    df = gdf.copy()
    design_vars = _safe_columns_for_design_variables(df)
    objectives = _objective_columns(df)

    objective_labels = {
        "Cost_normalized": "Cost / S2",
        "Carbon_normalized": "Carbon / S2",
        "Time_normalized": "Time / S2",
    }

    display_names = {
        "R_p": r"$R_p$ panel repetition",
        "Rp": r"$R_p$ panel repetition",
        "RP": r"$R_p$ panel repetition",
        "panel_repetition_index": r"$R_p$ panel repetition",
        "panel_RI": r"$R_p$ panel repetition",
        "RI_panel": r"$R_p$ panel repetition",

        "R_t": r"$R_t$ truss repetition",
        "Rt": r"$R_t$ truss repetition",
        "RT": r"$R_t$ truss repetition",
        "truss_repetition_index": r"$R_t$ truss repetition",
        "truss_RI": r"$R_t$ truss repetition",
        "RI_truss": r"$R_t$ truss repetition",

        "opening_complexity_index": r"$OC$ opening complexity",
        "OC": r"$OC$ opening complexity",

        "P_f_prefab_level": r"$P_f$ prefabrication level",
        "P_f": r"$P_f$ prefabrication level",
        "Pf": r"$P_f$ prefabrication level",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, var in zip(axes, design_vars):
        temp = df.copy()

        # Round continuous values to avoid noisy grouping.
        if temp[var].dtype.kind in "fc":
            temp[var] = temp[var].round(3)

        grouped = temp.groupby(var)[objectives].mean().reset_index()

        for obj in objectives:
            ax.plot(
                grouped[var],
                grouped[obj],
                marker="o",
                linewidth=1.8,
                label=objective_labels[obj]
            )

        ax.set_title(f"Main effect: {display_names.get(var, var)}")
        ax.set_xlabel(display_names.get(var, var))
        ax.set_ylabel("Mean normalised objective")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        r"Main-effects plot for $R_p$, $R_t$, $OC$, and $P_f$",
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")


def _monte_carlo_to_long_dataframe(mc) -> pd.DataFrame:
    """
    Convert Monte Carlo output into long format for plotting.

    Expected output columns:
        Solution
        Objective
        Value

    This version is robust to different MC dictionary structures:
    1. mc[solution]["Cost_NZD"] = array/list
    2. mc[solution]["cost_samples"] = array/list
    3. mc[solution]["cost"]["samples"] = array/list
    4. mc[solution] contains scalar summary values only -> skipped for boxplot
    """

    records = []

    if mc is None:
        return pd.DataFrame(columns=["Solution", "Objective", "Value"])

    if not isinstance(mc, dict):
        print("[WARNING] Monte Carlo object is not a dictionary. Boxplot skipped.")
        return pd.DataFrame(columns=["Solution", "Objective", "Value"])

    objective_aliases = {
        "Cost_NZD": [
            "Cost_NZD",
            "Cost",
            "cost",
            "cost_nzd",
            "cost_samples",
            "Cost_samples",
            "Cost_NZD_samples",
        ],
        "Carbon_kgCO2e": [
            "Carbon_kgCO2e",
            "Carbon",
            "carbon",
            "carbon_kgCO2e",
            "carbon_samples",
            "Carbon_samples",
            "Carbon_kgCO2e_samples",
            "CO2",
            "CO2_samples",
        ],
        "Time_hours": [
            "Time_hours",
            "Time",
            "time",
            "time_hours",
            "time_samples",
            "Time_samples",
            "Time_hours_samples",
        ],
    }

    nested_sample_keys = [
        "samples",
        "sample",
        "values",
        "runs",
        "draws",
        "simulations",
        "raw",
        "data",
    ]

    def to_numeric_array(value):
        """
        Convert value to a finite numeric numpy array.
        Scalars are returned as a one-value array, but scalar summary
        values are normally avoided before this function is called.
        """

        if value is None:
            return np.array([], dtype=float)

        if isinstance(value, pd.Series):
            arr = pd.to_numeric(value, errors="coerce").to_numpy(dtype=float)

        elif isinstance(value, pd.DataFrame):
            return np.array([], dtype=float)

        elif isinstance(value, np.ndarray):
            arr = value.astype(float).ravel()

        elif isinstance(value, (list, tuple)):
            arr = pd.to_numeric(pd.Series(list(value)), errors="coerce").to_numpy(dtype=float)

        else:
            try:
                arr = np.array([float(value)], dtype=float)
            except (TypeError, ValueError):
                return np.array([], dtype=float)

        arr = arr[np.isfinite(arr)]
        return arr

    for solution_name, payload in mc.items():

        if payload is None:
            continue

        # Ignore obvious global summary entries
        if str(solution_name).lower() in ["summary", "baseline", "s2", "s2 reference"]:
            continue

        # Case A: payload is a dataframe
        if isinstance(payload, pd.DataFrame):
            for objective in ["Cost_NZD", "Carbon_kgCO2e", "Time_hours"]:
                if objective in payload.columns:
                    arr = to_numeric_array(payload[objective])
                    for value in arr:
                        records.append(
                            {
                                "Solution": solution_name,
                                "Objective": objective,
                                "Value": value,
                            }
                        )
            continue

        # Case B: payload is a dictionary
        if isinstance(payload, dict):

            for objective, aliases in objective_aliases.items():
                extracted = False

                for key in aliases:
                    if key not in payload:
                        continue

                    value = payload[key]

                    # Nested structure, e.g. payload["cost"]["samples"]
                    if isinstance(value, dict):
                        sample_value = None

                        for sample_key in nested_sample_keys:
                            if sample_key in value:
                                sample_value = value[sample_key]
                                break

                        # If only scalar summary statistics are present, skip.
                        if sample_value is None:
                            continue

                        arr = to_numeric_array(sample_value)

                    else:
                        arr = to_numeric_array(value)

                    if len(arr) > 0:
                        for v in arr:
                            records.append(
                                {
                                    "Solution": solution_name,
                                    "Objective": objective,
                                    "Value": v,
                                }
                            )

                        extracted = True
                        break

                if extracted:
                    continue

            continue

        # Case C: payload is a raw array/list with three columns
        if isinstance(payload, (list, tuple, np.ndarray)):
            arr = np.asarray(payload)

            if arr.ndim == 2 and arr.shape[1] >= 3:
                objective_names = ["Cost_NZD", "Carbon_kgCO2e", "Time_hours"]

                for j, objective in enumerate(objective_names):
                    values = to_numeric_array(arr[:, j])

                    for v in values:
                        records.append(
                            {
                                "Solution": solution_name,
                                "Objective": objective,
                                "Value": v,
                            }
                        )

    long_df = pd.DataFrame(records, columns=["Solution", "Objective", "Value"])

    if long_df.empty:
        print(
            "[WARNING] No Monte Carlo sample arrays found for boxplot. "
            "This means mc contains only scalar summary values or uses unexpected keys."
        )

    return long_df

def plot_monte_carlo_boxplots(mc, path):
    """
    Plot Monte Carlo uncertainty distributions for representative Pareto solutions.

    """

    long_df = _monte_carlo_to_long_dataframe(mc)

    if long_df.empty:
        print("[WARNING] Monte Carlo boxplot skipped because no sample distributions were found.")
        return

    required = ["Solution", "Objective", "Value"]
    missing = [c for c in required if c not in long_df.columns]

    if missing:
        print(f"[WARNING] Monte Carlo boxplot skipped. Missing columns: {missing}")
        return

    # Keep only finite numeric values
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.dropna(subset=["Value"])
    long_df = long_df[np.isfinite(long_df["Value"])]

    if long_df.empty:
        print("[WARNING] Monte Carlo boxplot skipped because values are non-numeric.")
        return

    objective_order = ["Cost_NZD", "Carbon_kgCO2e", "Time_hours"]
    objective_labels = {
        "Cost_NZD": "Cost (NZD)",
        "Carbon_kgCO2e": "A1–A5 carbon (kgCO2e)",
        "Time_hours": "Process time (h)",
    }

    available_objectives = [
        obj for obj in objective_order
        if obj in long_df["Objective"].unique()
    ]

    if not available_objectives:
        print("[WARNING] Monte Carlo boxplot skipped. No recognised objective names found.")
        return

    fig, axes = plt.subplots(
        len(available_objectives),
        1,
        figsize=(9.5, 3.2 * len(available_objectives)),
        sharex=False
    )

    if len(available_objectives) == 1:
        axes = [axes]

    for ax, objective in zip(axes, available_objectives):
        sub = long_df[long_df["Objective"] == objective].copy()

        solution_order = list(sub["Solution"].dropna().astype(str).unique())

        data = [
            sub.loc[sub["Solution"].astype(str) == sol, "Value"].to_numpy()
            for sol in solution_order
        ]

        ax.boxplot(
            data,
            labels=solution_order,
            showmeans=True,
            patch_artist=False
        )

        ax.set_ylabel(objective_labels.get(objective, objective))
        ax.grid(axis="y", alpha=0.30)

        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_ha("right")

    fig.suptitle(
        "Monte Carlo uncertainty distributions for representative Pareto solutions",
        y=0.995
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")

def run_oat_sensitivity(obj_func, baseline, ts):
    xr = baseline.x_ref
    C0 = obj_func.calculate_cost_abs(xr)
    CO0 = obj_func.calculate_carbon_abs(xr)
    T0 = obj_func.calculate_time_abs(xr)
    cases = {
        'R_p_panel_rationalisation': (
            0,
            baseline.R_p_range[0],
            baseline.R_p_range[1]
        ),
        'R_t_truss_rationalisation': (
            1,
            baseline.R_t_range[0],
            baseline.R_t_range[1]
        ),
        'opening_complexity_index': (
            2,
            baseline.opening_complexity_range[0],
            baseline.opening_complexity_range[1]
        ),
        'P_f_prefab_level': (
            3,
            baseline.P_f_range[0],
            baseline.P_f_range[1]
        ),
    }
    rows = []
    for var, (idx, lo, hi) in cases.items():
        xl = list(xr); xh = list(xr); xl[idx]=lo; xh[idx]=hi
        rows.append({
            'variable': var,
            'cost_low_pct_change':   (obj_func.calculate_cost_abs(xl)-C0)/C0*100,
            'cost_high_pct_change':  (obj_func.calculate_cost_abs(xh)-C0)/C0*100,
            'carbon_low_pct_change': (obj_func.calculate_carbon_abs(xl)-CO0)/CO0*100,
            'carbon_high_pct_change':(obj_func.calculate_carbon_abs(xh)-CO0)/CO0*100,
            'time_low_pct_change':   (obj_func.calculate_time_abs(xl)-T0)/T0*100,
            'time_high_pct_change':  (obj_func.calculate_time_abs(xh)-T0)/T0*100,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / f"sensitivity_oat_{ts}.csv", index=False)
    plot_sensitivity(df, FIG_DIR / f"sensitivity_oat_{ts}.png")
    # Additional verification sweep for truss rationalisation trade-off
    Rt_sweep = np.linspace(0.0, 1.0, 50)

    costs_rt = [
        obj_func.calculate_cost_abs([xr[0], rt, xr[2], xr[3]])
        for rt in Rt_sweep
    ]

    min_rt = Rt_sweep[int(np.argmin(costs_rt))]

    print(f"[R_t OAT] Min cost at R_t={min_rt:.3f}")

    if min_rt < 0.05:
        print("[WARNING] R_t cost minimum still at lower bound — grouping penalty may still dominate")
    elif min_rt > 0.95:
        print("[WARNING] R_t cost minimum at upper bound — setup saving may dominate too strongly")
    else:
        print("[OK] R_t cost minimum is interior — truss trade-off exists")
    return df

def build_sensitivity_elasticity_table(sens_df: pd.DataFrame,
                                       baseline: CFSProjectBaseline,
                                       ts: str) -> pd.DataFrame:
    """
    Elasticity:
        E = (% change in objective) / (% change in variable)

    For discrete variables, this is reported as pseudo-elasticity because
    the variable is an index/category.
    """

    x_ref = {
        'R_p_panel_rationalisation': baseline.x_ref[0],
        'R_t_truss_rationalisation': baseline.x_ref[1],
        'opening_complexity_index': baseline.x_ref[2],
        'P_f_prefab_level': baseline.x_ref[3],
    }

    bounds = {
        'R_p_panel_rationalisation': baseline.R_p_range,
        'R_t_truss_rationalisation': baseline.R_t_range,
        'opening_complexity_index': baseline.opening_complexity_range,
        'P_f_prefab_level': baseline.P_f_range,
    }

    rows = []

    for _, r in sens_df.iterrows():
        var = r['variable']
        ref = x_ref[var]
        lo, hi = bounds[var]

        # Avoid divide by zero for baseline category = 0
        if ref == 0:
            low_var_pct = np.nan
            high_var_pct = np.nan
            variable_type = "continuous rationalisation variable; reference value is zero so elasticity not computed"
        else:
            low_var_pct = (lo - ref) / ref * 100
            high_var_pct = (hi - ref) / ref * 100
            variable_type = "continuous/index elasticity"

        for obj in ['cost', 'carbon', 'time']:
            low_y = r[f'{obj}_low_pct_change']
            high_y = r[f'{obj}_high_pct_change']

            low_E = low_y / low_var_pct if pd.notna(low_var_pct) and abs(low_var_pct) > 1e-12 else np.nan
            high_E = high_y / high_var_pct if pd.notna(high_var_pct) and abs(high_var_pct) > 1e-12 else np.nan

            rows.append({
                'Variable': var,
                'Objective': obj,
                'Variable_type': variable_type,
                'Low_case_%Y': round(low_y, 3),
                'High_case_%Y': round(high_y, 3),
                'Low_case_elasticity': round(low_E, 3) if pd.notna(low_E) else "",
                'High_case_elasticity': round(high_E, 3) if pd.notna(high_E) else "",
            })

    out = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"sensitivity_elasticity_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[SENSITIVITY ELASTICITY TABLE]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def evaluate_lhs_design_space(obj_func: CFSObjectiveFunctions,
                              baseline: CFSProjectBaseline,
                              ts: str,
                              n_samples: int = 2000,
                              seed: int = 42) -> pd.DataFrame:
    """
    Independent LHS design-space evaluation.

    This is separate from NSGA-II.
    It is used for global sensitivity analysis across the full design domain.
    """
    design_df = generate_lhs_designs(
        n_samples=n_samples,
        b=baseline,
        seed=seed
    )

    rows = []

    for _, row in design_df.iterrows():
        x = [
            float(row["R_p_panel_rationalisation"]),
            float(row["R_t_truss_rationalisation"]),
            float(row["opening_complexity_index"]),
            float(row["P_f_prefab_level"])
        ]
        C = obj_func.calculate_cost_abs(x)
        E = obj_func.calculate_carbon_abs(x)
        T = obj_func.calculate_time_abs(x)
        W = obj_func.calculate_material_weight(x)
        ps = obj_func.panel_scheme_from_Rp(x[0])
        ts = obj_func.truss_scheme_from_Rt(x[1])

        # Constraint report is optional; only use if your code already has nz_constraint_report()
        try:
            constraint_report = obj_func.nz_constraint_report(x, C, E, T)
            NZ_feasible = constraint_report["NZ_feasible"]
            Total_constraint_penalty = constraint_report["Total_constraint_penalty"]
            Factory_repetition_score = constraint_report["factory_repetition_score"]
            Carbon_intensity = constraint_report["carbon_intensity_kgco2e_m2"]
        except AttributeError:
            NZ_feasible = True
            Total_constraint_penalty = 0.0
            Factory_repetition_score = np.nan
            Carbon_intensity = E / baseline.footprint_area_m2

        rows.append({
            "Sample_ID": int(row["Sample_ID"]),
            "R_p_panel_rationalisation": float(x[0]),
            "R_t_truss_rationalisation": float(x[1]),
            "panel_scheme": int(ps),
            "truss_scheme": int(ts),
            "opening_complexity_index": float(x[2]),
            "P_f_prefab_level": float(x[3]),

            "Weight_kg": W,

            "Cost_NZD": C,
            "Carbon_kgCO2e": E,
            "Time_hours": T,

            "Cost_normalized": C / baseline.C_S2_reference,
            "Carbon_normalized": E / baseline.CO2_S2_reference,
            "Time_normalized": T / baseline.Time_S2_reference,

            "Cost_vs_S2_%": (C / baseline.C_S2_reference - 1.0) * 100,
            "Carbon_vs_S2_%": (E / baseline.CO2_S2_reference - 1.0) * 100,
            "Time_vs_S2_%": (T / baseline.Time_S2_reference - 1.0) * 100,

            "NZ_feasible": NZ_feasible,
            "Total_constraint_penalty": Total_constraint_penalty,
            "Factory_repetition_score": Factory_repetition_score,
            "Carbon_intensity_kgCO2e_m2": Carbon_intensity,
        })

    out = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"lhs_design_space_{n_samples}_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[LHS DESIGN SPACE EVALUATION]")
    print(f"  Samples evaluated: {len(out)}")
    print(f"  Saved: {out_path}")

    return out


def compute_prcc(df: pd.DataFrame,
                 x_cols: list,
                 y_col: str) -> pd.DataFrame:
    """
    Partial Rank Correlation Coefficient.

    PRCC estimates the rank relationship between each input and output
    while controlling for the other inputs.

    Note:
    panel_scheme and truss_scheme are categorical indices, so their PRCC
    should be interpreted as screening-level evidence only.
    """
    data = df[x_cols + [y_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()

    R = data.rank(method="average")

    X = R[x_cols].values.astype(float)
    y = R[y_col].values.astype(float)

    rows = []

    for i, var in enumerate(x_cols):
        Xi = X[:, i]
        X_others = np.delete(X, i, axis=1)

        A = np.column_stack([np.ones(len(X_others)), X_others])

        beta_x, *_ = np.linalg.lstsq(A, Xi, rcond=None)
        res_x = Xi - A @ beta_x

        beta_y, *_ = np.linalg.lstsq(A, y, rcond=None)
        res_y = y - A @ beta_y

        if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
            prcc = np.nan
        else:
            prcc = np.corrcoef(res_x, res_y)[0, 1]

        rows.append({
            "Output": y_col,
            "Variable": var,
            "PRCC": prcc
        })

    return pd.DataFrame(rows)


def run_lhs_global_sensitivity(lhs_df: pd.DataFrame,
                               ts: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    LHS-based global sensitivity analysis.

    Outputs:
      1. Spearman rank sensitivity
      2. PRCC sensitivity
      3.Grouped reporting effects for panel_scheme and truss_scheme reporting categories
    """
    x_cols = [
        "R_p_panel_rationalisation",
        "R_t_truss_rationalisation",
        "opening_complexity_index",
        "P_f_prefab_level"
    ]

    y_cols = [
        "Cost_NZD",
        "Carbon_kgCO2e",
        "Time_hours"
    ]

    # Spearman sensitivity
    spearman_rows = []

    for y in y_cols:
        corr = lhs_df[x_cols + [y]].corr(method="spearman")[y].drop(y)

        for var, val in corr.items():
            spearman_rows.append({
                "Output": y,
                "Variable": var,
                "Spearman_rho": val,
                "Abs_Spearman_rho": abs(val)
            })

    spearman_df = pd.DataFrame(spearman_rows)
    spearman_df = spearman_df.sort_values(
        ["Output", "Abs_Spearman_rho"],
        ascending=[True, False]
    )

    # PRCC sensitivity
    prcc_df = pd.concat(
        [compute_prcc(lhs_df, x_cols, y) for y in y_cols],
        ignore_index=True
    )

    prcc_df["Abs_PRCC"] = prcc_df["PRCC"].abs()
    prcc_df = prcc_df.sort_values(
        ["Output", "Abs_PRCC"],
        ascending=[True, False]
    )

    # Discrete grouped effects
    group_rows = []

    for discrete_var in ["panel_scheme", "truss_scheme"]:
        for y in y_cols:
            g = (
                lhs_df
                .groupby(discrete_var)[y]
                .agg(["count", "mean", "std", "min", "max"])
                .reset_index()
            )

            global_mean = lhs_df[y].mean()

            for _, r in g.iterrows():
                group_rows.append({
                    "Discrete_variable": discrete_var,
                    "Level": int(r[discrete_var]),
                    "Output": y,
                    "Count": int(r["count"]),
                    "Mean": r["mean"],
                    "Std": r["std"],
                    "Min": r["min"],
                    "Max": r["max"],
                    "Mean_vs_global_%": (r["mean"] / global_mean - 1.0) * 100
                })

    group_effect_df = pd.DataFrame(group_rows)

    spearman_path = TABLE_DIR / f"lhs_spearman_sensitivity_{ts}.csv"
    prcc_path = TABLE_DIR / f"lhs_prcc_sensitivity_{ts}.csv"
    group_path = TABLE_DIR / f"lhs_discrete_group_effects_{ts}.csv"

    spearman_df.to_csv(spearman_path, index=False)
    prcc_df.to_csv(prcc_path, index=False)
    group_effect_df.to_csv(group_path, index=False)

    print("\n[LHS GLOBAL SENSITIVITY — SPEARMAN]")
    print(spearman_df.to_string(index=False))
    print(f"[SAVED] {spearman_path}")

    print("\n[LHS GLOBAL SENSITIVITY — PRCC]")
    print(prcc_df.to_string(index=False))
    print(f"[SAVED] {prcc_path}")

    print(f"[SAVED] {group_path}")

    return spearman_df, prcc_df, group_effect_df

def run_grid_validation(obj_func, baseline, pf_step=0.05):
    recs = []

    R_p_grid = np.round(np.linspace(
        baseline.R_p_range[0],
        baseline.R_p_range[1],
        7 if RUN_MODE == "final" else 4
    ), 3)

    R_t_grid = np.round(np.linspace(
        baseline.R_t_range[0],
        baseline.R_t_range[1],
        7 if RUN_MODE == "final" else 3
    ), 3)

    opening_grid = np.round(
        np.linspace(
            baseline.opening_complexity_range[0],
            baseline.opening_complexity_range[1],
            6
        ),
        3
    )

    pf_grid = np.round(
        np.arange(
            baseline.P_f_range[0],
            baseline.P_f_range[1] + pf_step,
            pf_step
        ),
        3
    )

    for R_p, R_t, oc, P_f in iproduct(
        R_p_grid,
        R_t_grid,
        opening_grid,
        pf_grid
    ):
        x = [float(R_p), float(R_t), float(oc), float(P_f)]

        ps = obj_func.panel_scheme_from_Rp(R_p)
        ts = obj_func.truss_scheme_from_Rt(R_t)

        N_panel_classes = obj_func.n_panel_classes(R_p)
        N_truss_types = obj_func.n_truss_types(R_t)

        C_raw = obj_func.calculate_cost_abs(x)
        CO2_raw = obj_func.calculate_carbon_abs(x)
        T_raw = obj_func.calculate_time_abs(x)

        vals = np.array([C_raw, CO2_raw, T_raw], dtype=float)
        if not np.isfinite(vals).all():
            print(f"[GRID WARNING] Non-finite objective for x={x}. Skipped.")
            continue

        f_cost, f_carbon, f_time = obj_func.calculate_constrained_fitness(x)
        constraint_report = obj_func.nz_constraint_report(x, C_raw, CO2_raw, T_raw)

        recs.append({
            'R_p_panel_rationalisation': float(R_p),
            'R_t_truss_rationalisation': float(R_t),
            'panel_scheme': int(ps),
            'truss_scheme': int(ts),
            'N_panel_classes': round(float(N_panel_classes), 3),
            'N_truss_types': round(float(N_truss_types), 3),
            'opening_complexity_index': float(oc),
            'P_f_prefab_level': float(P_f),

            'Cost_NZD': C_raw,
            'Carbon_kgCO2e': CO2_raw,
            'Time_hours': T_raw,

            'Cost_normalized': C_raw / baseline.C_S2_reference,
            'Carbon_normalized': CO2_raw / baseline.CO2_S2_reference,
            'Time_normalized': T_raw / baseline.Time_S2_reference,

            'Cost_fitness': f_cost,
            'Carbon_fitness': f_carbon,
            'Time_fitness': f_time,

            'NZ_feasible': constraint_report['NZ_feasible'],
            'Hard_violations': constraint_report['Hard_violations'],
            'Soft_violations': constraint_report['Soft_violations'],
            'Penalty_cost': constraint_report['Penalty_cost'],
            'Penalty_carbon': constraint_report['Penalty_carbon'],
            'Penalty_time': constraint_report['Penalty_time'],
            'Total_constraint_penalty': constraint_report['Total_constraint_penalty'],

            'Max_panel_width_m': constraint_report['max_panel_width_m'],
            'Panel_height_m': constraint_report['panel_height_m'],
            'Panel_lift_mass_kg': constraint_report['panel_lift_mass_kg'],
            'Factory_repetition_score': constraint_report['factory_repetition_score'],
            'Carbon_intensity_kgCO2e_m2': constraint_report['carbon_intensity_kgco2e_m2'],
        })

    gdf = pd.DataFrame(recs)

    if len(gdf) == 0:
        raise ValueError("Grid validation produced zero valid rows.")

    gpdf = get_pareto_front_df(
        gdf,
        obj_cols=('Cost_normalized', 'Carbon_normalized', 'Time_normalized')
    )

    return gdf, gpdf

def run_ablation_grid_suite(obj_func, baseline, ts):
    """
    Ablation validation to test whether the Pareto front is driven by all
    variables or only by P_f / OC / rationalisation.

    This does not replace NSGA-II. It is an examiner-facing diagnostic.
    """

    scenarios = {
        "full_grid": {
            "R_p_values": np.linspace(baseline.R_p_range[0], baseline.R_p_range[1], 5),
            "R_t_values": np.linspace(baseline.R_t_range[0], baseline.R_t_range[1], 5),
            "OC_values": np.linspace(baseline.opening_complexity_range[0], baseline.opening_complexity_range[1], 6),
            "P_f_values": np.linspace(baseline.P_f_range[0], baseline.P_f_range[1], 9),
        },
        "prefab_only": {
            "R_p_values": [baseline.x_ref[0]],
            "R_t_values": [baseline.x_ref[1]],
            "OC_values": [baseline.x_ref[2]],
            "P_f_values": np.linspace(baseline.P_f_range[0], baseline.P_f_range[1], 21),
        },
        "rationalisation_only": {
            "R_p_values": np.linspace(baseline.R_p_range[0], baseline.R_p_range[1], 11),
            "R_t_values": np.linspace(baseline.R_t_range[0], baseline.R_t_range[1], 11),
            "OC_values": [baseline.x_ref[2]],
            "P_f_values": [baseline.x_ref[3]],
        },
        "OC_fixed_full": {
            "R_p_values": np.linspace(baseline.R_p_range[0], baseline.R_p_range[1], 5),
            "R_t_values": np.linspace(baseline.R_t_range[0], baseline.R_t_range[1], 5),
            "OC_values": [baseline.x_ref[2]],
            "P_f_values": np.linspace(baseline.P_f_range[0], baseline.P_f_range[1], 9),
        },
        "OC_only": {
            "R_p_values": [baseline.x_ref[0]],
            "R_t_values": [baseline.x_ref[1]],
            "OC_values": np.linspace(baseline.opening_complexity_range[0], baseline.opening_complexity_range[1], 21),
            "P_f_values": [baseline.x_ref[3]],
        },
    }

    all_summary = []
    all_rows = []

    obj_func._ensure_refs()

    for scenario_name, grid in scenarios.items():
        rows = []

        for R_p, R_t, OC, P_f in iproduct(
            grid["R_p_values"],
            grid["R_t_values"],
            grid["OC_values"],
            grid["P_f_values"],
        ):
            x = [float(R_p), float(R_t), float(OC), float(P_f)]

            C = obj_func.calculate_cost_abs(x)
            E = obj_func.calculate_carbon_abs(x)
            T = obj_func.calculate_time_abs(x)

            rows.append({
                "Scenario": scenario_name,
                "R_p_panel_rationalisation": x[0],
                "R_t_truss_rationalisation": x[1],
                "panel_scheme": obj_func.panel_scheme_from_Rp(x[0]),
                "truss_scheme": obj_func.truss_scheme_from_Rt(x[1]),
                "opening_complexity_index": x[2],
                "P_f_prefab_level": x[3],
                "Cost_NZD": C,
                "Carbon_kgCO2e": E,
                "Time_hours": T,
                "Cost_normalized": C / obj_func._C_ref,
                "Carbon_normalized": E / obj_func._CO2_ref,
                "Time_normalized": T / obj_func._T_ref,
            })

        df = pd.DataFrame(rows)

        pdf_scenario = get_pareto_front_df(
            df,
            obj_cols=("Cost_normalized", "Carbon_normalized", "Time_normalized")
        )

        all_rows.append(df)

        all_summary.append({
            "Scenario": scenario_name,
            "Evaluated_designs": len(df),
            "Pareto_designs": len(pdf_scenario),

            "Cost_min": pdf_scenario["Cost_normalized"].min(),
            "Cost_max": pdf_scenario["Cost_normalized"].max(),
            "Carbon_min": pdf_scenario["Carbon_normalized"].min(),
            "Carbon_max": pdf_scenario["Carbon_normalized"].max(),
            "Time_min": pdf_scenario["Time_normalized"].min(),
            "Time_max": pdf_scenario["Time_normalized"].max(),

            "R_p_min": pdf_scenario["R_p_panel_rationalisation"].min(),
            "R_p_max": pdf_scenario["R_p_panel_rationalisation"].max(),
            "R_t_min": pdf_scenario["R_t_truss_rationalisation"].min(),
            "R_t_max": pdf_scenario["R_t_truss_rationalisation"].max(),
            "OC_min": pdf_scenario["opening_complexity_index"].min(),
            "OC_max": pdf_scenario["opening_complexity_index"].max(),
            "P_f_min": pdf_scenario["P_f_prefab_level"].min(),
            "P_f_max": pdf_scenario["P_f_prefab_level"].max(),
        })

        pdf_path = TABLE_DIR / f"ablation_pareto_{scenario_name}_{ts}.csv"
        pdf_scenario.to_csv(pdf_path, index=False)
        print(f"[SAVED] {pdf_path}")

    summary_df = pd.DataFrame(all_summary)
    all_df = pd.concat(all_rows, ignore_index=True)

    summary_path = TABLE_DIR / f"ablation_summary_{ts}.csv"
    all_path = TABLE_DIR / f"ablation_all_designs_{ts}.csv"

    summary_df.to_csv(summary_path, index=False)
    all_df.to_csv(all_path, index=False)

    print("\n[ABLATION SUMMARY]")
    print(summary_df.to_string(index=False))
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {all_path}")

    return summary_df, all_df

def evaluate_prefab_scenarios(obj_func, baseline, ts):
    """
    Evaluate the predefined low / medium / high prefabrication scenarios.
    These are benchmark scenarios, not fixed alternatives selected by the optimizer.

    S1 = low prefab
    S2 = medium prefab reference
    S3 = high prefab
    """

    prefab_scenarios = {
        "S1_low_prefab": 0.50,
        "S2_medium_prefab_reference": 0.72,
        "S3_high_prefab": 0.90,
    }

    rows = []

    # S2 reference vector from your current code:
    # x_ref = [R_p, R_t, opening_complexity, P_f]
    x_s2 = list(baseline.x_ref)

    C_s2 = obj_func.calculate_cost_abs(x_s2)
    CO2_s2 = obj_func.calculate_carbon_abs(x_s2)
    T_s2 = obj_func.calculate_time_abs(x_s2)

    for name, Pf in prefab_scenarios.items():
        x = [
            baseline.x_ref[0],   # panel_scheme
            baseline.x_ref[1],   # truss_scheme
            baseline.x_ref[2],   
            Pf
        ]

        C = obj_func.calculate_cost_abs(x)
        CO2 = obj_func.calculate_carbon_abs(x)
        T = obj_func.calculate_time_abs(x)

        rows.append({
            "Scenario": name,
            "R_p_panel_rationalisation": x[0],
            "R_t_truss_rationalisation": x[1],
            "panel_scheme": obj_func.panel_scheme_from_Rp(x[0]),
            "truss_scheme": obj_func.truss_scheme_from_Rt(x[1]),
            "opening_complexity_index": x[2],
            "N_openings_reference": baseline.N_openings_reference,
            "P_f_prefab_level": Pf,

            "Cost_NZD_raw": round(C, 2),
            "Carbon_kgCO2e_raw": round(CO2, 2),
            "Time_hours_raw": round(T, 2),

            "Cost_vs_S2_%": round((C / C_s2 - 1.0) * 100, 2),
            "Carbon_vs_S2_%": round((CO2 / CO2_s2 - 1.0) * 100, 2),
            "Time_vs_S2_%": round((T / T_s2 - 1.0) * 100, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / f"prefab_scenario_comparison_{ts}.csv", index=False)

    print("\n[PREFAB SCENARIO COMPARISON]")
    print(df.to_string(index=False))
    print(f"[SAVED] Prefab scenario table: {TABLE_DIR / f'prefab_scenario_comparison_{ts}.csv'}")

    return df

def build_design_variable_table(baseline: CFSProjectBaseline) -> pd.DataFrame:
    rows = [
        {
            "Variable": "R_p_panel_rationalisation",
            "Symbol": "R_p",
            "Type": "Continuous optimiser variable",
            "Values": str(baseline.R_p_range),
            "Baseline_S2": baseline.x_ref[0],
            "Unit": "dimensionless",
            "Meaning": "Panel rationalisation intensity; higher value means fewer/larger/more standardised panel families",
            "Why_included": "Captures panelisation trade-off between factory repetition, site labour reduction, transport/lifting constraints, and material/handling penalties"
        },
        {
            "Variable": "R_t_truss_rationalisation",
            "Symbol": "R_t",
            "Type": "Continuous optimiser variable",
            "Values": str(baseline.R_t_range),
            "Baseline_S2": baseline.x_ref[1],
            "Unit": "dimensionless",
            "Meaning": "Truss rationalisation intensity; higher value means fewer/more standardised truss span families",
            "Why_included": "Captures truss repetition benefits against span mismatch, connection/detailing, self-weight, and lifting penalties"
        },
        {"Variable": "opening_complexity_index", "Symbol": "OC",
        "Type": "Continuous actionable detailing variable",
        "Values": str(baseline.opening_complexity_range),
        "Baseline_S2": baseline.x_ref[2],
        "Unit": "dimensionless",
        "Meaning": "Design-for-manufacture/detailing rationalisation index with fixed number of openings",
        "Why_included": "Captures standardisation of lintels, jamb studs, trimming, service penetrations, CNC cutting and installation difficulty without changing the architectural opening count"},
        {"Variable": "P_f_prefab_level", "Symbol": "Pf",
         "Type": "Continuous optimizer variable",
         "Values": str(baseline.P_f_range),
         "Baseline_S2": baseline.x_ref[3], "Unit": "fraction",
         "Meaning": "Prefabrication level (factory vs site share)",
         "Why_included": "Controls factory/site trade-off, labour saving, A5 carbon, time"},

        {"Variable": "max_panel_width_m", "Symbol": "W_panel,max",
         "Type": "Hard feasibility constraint",
         "Values": baseline.max_panel_width_m,
         "Baseline_S2": "All panel schemes currently satisfy",
         "Unit": "m",
         "Meaning": "Maximum allowed panel transport/factory width",
         "Why_included": "Prevents unrealistic oversized wall-panel solutions"},

        {"Variable": "max_panel_height_m", "Symbol": "H_panel,max",
         "Type": "Hard feasibility constraint",
         "Values": baseline.max_panel_height_m,
         "Baseline_S2": baseline.assumed_panel_height_m,
         "Unit": "m",
         "Meaning": "Maximum practical panel height",
         "Why_included": "Represents factory and lifting practicality"},

        {"Variable": "max_stud_spacing_mm", "Symbol": "s_stud,max",
         "Type": "Hard code-screening constraint",
         "Values": baseline.max_stud_spacing_mm,
         "Baseline_S2": baseline.assumed_stud_spacing_mm,
         "Unit": "mm",
         "Meaning": "Maximum assumed stud spacing",
         "Why_included": "Preliminary structural/code-screening condition"},

        {"Variable": "assumed_steel_thickness_mm", "Symbol": "t_steel",
        "Type": "Fixed project boundary condition",
        "Values": baseline.assumed_steel_thickness_mm,
        "Baseline_S2": baseline.assumed_steel_thickness_mm,
        "Unit": "mm",
        "Meaning": "Actual case-study CFS section thickness",
        "Why_included": "Taken from project section: F325iT - Imported Section (89 S 41 / 0.75 / G500 / Z275). Fixed because structural section sizing is not optimized."},

        {"Variable": "min_steel_thickness_mm", "Symbol": "t_min",
        "Type": "Hard screening threshold",
        "Values": baseline.min_steel_thickness_mm,
        "Baseline_S2": baseline.assumed_steel_thickness_mm,
        "Unit": "mm",
        "Meaning": "Minimum permitted CFS thickness for feasibility screening",
        "Why_included": "Prevents unrealistic thin-section assumptions if thickness is later varied; not an active design variable in the current model."},

        {"Variable": "max_lift_mass_kg", "Symbol": "m_lift,max",
         "Type": "Hard site handling constraint",
         "Values": baseline.max_lift_mass_kg,
         "Baseline_S2": "computed",
         "Unit": "kg",
         "Meaning": "Maximum estimated panel lift mass",
         "Why_included": "Screens panel options for site handling practicality"},

        {"Variable": "min_factory_repetition_score", "Symbol": "R_factory,min",
         "Type": "Soft constructability constraint",
         "Values": baseline.min_factory_repetition_score,
         "Baseline_S2": "computed",
         "Unit": "dimensionless",
         "Meaning": "Minimum factory repetition score",
         "Why_included": "Penalizes inefficient prefabrication with excessive unique families"},

        {"Variable": "max_OCI_soft", "Symbol": "OCI_max",
         "Type": "Soft constructability constraint",
         "Values": baseline.max_OCI_soft,
         "Baseline_S2": baseline.x_ref[2],
         "Unit": "dimensionless",
         "Meaning": "Maximum preferred opening/detailing complexity",
         "Why_included": "Penalizes detailing complexity that increases site coordination and labour"},

        {"Variable": "waikato_weather_downtime_fraction", "Symbol": "D_weather",
         "Type": "Scenario assumption",
         "Values": baseline.waikato_weather_downtime_fraction,
         "Baseline_S2": "applied to time objective",
         "Unit": "fraction",
         "Meaning": "Weather downtime allowance",
         "Why_included": "Represents Waikato site productivity risk"},
        
        
    ]
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6.5: CALIBRATION AUDIT + MULTI-SEED CONVERGENCE
# ─────────────────────────────────────────────────────────────────────────────
def build_calibration_audit_table(params, s2_reference, output_path):
    rows = [
        {
            "parameter": "C_S2_reference",
            "value": s2_reference["cost"],
            "unit": "NZD",
            "range": "Not applicable",
            "source_basis": "Model-calculated CFS framing process-cost reference",
            "validation_status": "Calculated from current objective-function engine",
            "model_use": "S2 benchmark for normalisation and comparison",
        },
        {
            "parameter": "CO2_S2_reference",
            "value": s2_reference["carbon"],
            "unit": "kgCO2e",
            "range": "Not applicable",
            "source_basis": "Model-calculated CFS framing A1-A5 carbon reference",
            "validation_status": "Calculated from current objective-function engine",
            "model_use": "S2 benchmark for normalisation and comparison",
        },
        {
            "parameter": "Time_S2_reference",
            "value": s2_reference["time"],
            "unit": "process-hours",
            "range": "Not applicable",
            "source_basis": "Model-calculated CFS framing process-hour reference",
            "validation_status": "Calculated from current objective-function engine",
            "model_use": "S2 benchmark for normalisation and comparison",
        },
    ]

    # Add current coefficient values from the active model parameters
    for name, item in params.items():
        rows.append({
            "parameter": name,
            "value": item.get("value"),
            "unit": item.get("unit"),
            "range": item.get("range", "Not specified"),
            "source_basis": item.get("source_basis"),
            "validation_status": item.get("validation_status"),
            "model_use": item.get("model_use"),
        })

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_path, index=False)
    return audit_df

def safe_cv_percent(series):
    """
    Coefficient of variation in percent.
    Returns NaN if the mean is zero or invalid.
    """
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) <= 1:
        return np.nan

    mean_val = series.mean()
    std_val = series.std()

    if abs(mean_val) < 1e-12:
        return np.nan

    return float(std_val / mean_val * 100.0)


def run_multi_seed_convergence(pop_size=200,
                               generations=300,
                               seeds=(11, 22, 33, 44, 55)):
    """
    Multi-seed NSGA-II stability validation.

    Purpose:
    Runs the same optimisation problem using independent random seeds.
    The outputs test whether the final Pareto front is stable or whether
    the result depends strongly on one lucky stochastic run.

    Outputs:
    1. seed_df:
       One-row summary for each seed.

    2. summary_df:
       Cross-seed stability statistics.

    3. reference_front_df:
       Approximated multi-seed nondominated reference front.

    4. combined_seed_fronts_df:
       All seed-level Pareto fronts combined for plotting/checking.
    """

    seed_pareto_dfs = []
    seed_records = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        print("\n" + "=" * 70)
        print(f"[MULTI-SEED RUN: SEED {seed}]")
        print("=" * 70)

        final_archive, logbook, tb, baseline, obj, hv_df, seed_ts, eval_audit = run_optimization(
            pop_size=pop_size,
            generations=generations,
            seed=seed
        )

        if len(final_archive) == 0:
            print(f"[WARNING] Seed {seed} produced an empty archive.")
            seed_records.append({
                "Seed": seed,
                "Evaluated_fitness_calls": eval_audit.get("total_evaluated_fitness_calls", 0),
                "Hard_feasible_evaluated": eval_audit.get("hard_feasible_evaluated", 0),
                "Archive_size": 0,
                "Final_nondominated_solutions": 0,
                "Final_hypervolume": np.nan,
                "HV_initial": np.nan,
                "HV_gain": np.nan,
                "Min_cost_norm": np.nan,
                "Min_carbon_norm": np.nan,
                "Min_time_norm": np.nan,
                "Cost_range_norm": np.nan,
                "Carbon_range_norm": np.nan,
                "Time_range_norm": np.nan,
                "Rp_min": np.nan,
                "Rp_max": np.nan,
                "Rt_min": np.nan,
                "Rt_max": np.nan,
                "OC_min": np.nan,
                "OC_max": np.nan,
                "Pf_min": np.nan,
                "Pf_max": np.nan,
            })
            continue

        front = tools.sortNondominated(
            final_archive,
            len(final_archive),
            first_front_only=True
        )[0]

        seed_pdf = analyze_pareto_front(front, obj, baseline)

        seed_pdf = clean_finite_objective_rows(
            seed_pdf,
            name=f"Seed_{seed}_Pareto",
            ts=seed_ts
        )

        seed_pdf = get_pareto_front_df(
            seed_pdf,
            obj_cols=("Cost_normalized", "Carbon_normalized", "Time_normalized")
        )

        seed_pdf["Seed"] = seed
        seed_pareto_dfs.append(seed_pdf)

        final_hv = float(hv_df["hypervolume"].iloc[-1]) if len(hv_df) > 0 else np.nan
        initial_hv = float(hv_df["hypervolume"].iloc[0]) if len(hv_df) > 0 else np.nan
        hv_gain = final_hv - initial_hv if np.isfinite(final_hv) and np.isfinite(initial_hv) else np.nan

        seed_records.append({
            "Seed": seed,
            "Evaluated_fitness_calls": eval_audit.get("total_evaluated_fitness_calls", 0),
            "Hard_feasible_evaluated": eval_audit.get("hard_feasible_evaluated", 0),
            "Archive_size": len(final_archive),
            "Final_nondominated_solutions": len(seed_pdf),
            "Final_hypervolume": final_hv,
            "HV_initial": initial_hv,
            "HV_gain": hv_gain,

            "Min_cost_norm": seed_pdf["Cost_normalized"].min(),
            "Min_carbon_norm": seed_pdf["Carbon_normalized"].min(),
            "Min_time_norm": seed_pdf["Time_normalized"].min(),

            "Cost_range_norm": seed_pdf["Cost_normalized"].max() - seed_pdf["Cost_normalized"].min(),
            "Carbon_range_norm": seed_pdf["Carbon_normalized"].max() - seed_pdf["Carbon_normalized"].min(),
            "Time_range_norm": seed_pdf["Time_normalized"].max() - seed_pdf["Time_normalized"].min(),

            "Rp_min": seed_pdf["R_p_panel_rationalisation"].min(),
            "Rp_max": seed_pdf["R_p_panel_rationalisation"].max(),
            "Rt_min": seed_pdf["R_t_truss_rationalisation"].min(),
            "Rt_max": seed_pdf["R_t_truss_rationalisation"].max(),
            "OC_min": seed_pdf["opening_complexity_index"].min(),
            "OC_max": seed_pdf["opening_complexity_index"].max(),
            "Pf_min": seed_pdf["P_f_prefab_level"].min(),
            "Pf_max": seed_pdf["P_f_prefab_level"].max(),
        })

    seed_df = pd.DataFrame(seed_records)

    if len(seed_pareto_dfs) == 0:
        raise ValueError("All multi-seed runs failed or produced empty Pareto fronts.")

    combined_seed_fronts_df = pd.concat(seed_pareto_dfs, ignore_index=True)

    reference_front_df = get_pareto_front_df(
        combined_seed_fronts_df,
        obj_cols=("Cost_normalized", "Carbon_normalized", "Time_normalized")
    )

    reference_pts = reference_front_df[
        ["Cost_normalized", "Carbon_normalized", "Time_normalized"]
    ].values

    # IGD of each seed against the approximated multi-seed reference front
    for i, seed in enumerate(seeds):
        if i >= len(seed_pareto_dfs):
            seed_df.loc[seed_df["Seed"] == seed, "IGD_vs_multiseed_reference"] = np.nan
            continue

        seed_pdf = seed_pareto_dfs[i]

        seed_pts = seed_pdf[
            ["Cost_normalized", "Carbon_normalized", "Time_normalized"]
        ].values

        igd_seed = compute_igd(seed_pts, reference_pts)
        seed_df.loc[seed_df["Seed"] == seed, "IGD_vs_multiseed_reference"] = igd_seed

    summary_rows = []

    for col in [
        "Final_hypervolume",
        "IGD_vs_multiseed_reference",
        "Final_nondominated_solutions",
        "Min_cost_norm",
        "Min_carbon_norm",
        "Min_time_norm",
        "Cost_range_norm",
        "Carbon_range_norm",
        "Time_range_norm",
    ]:
        summary_rows.append({
            "Metric": col,
            "Mean": seed_df[col].mean(),
            "Std": seed_df[col].std(),
            "Min": seed_df[col].min(),
            "Max": seed_df[col].max(),
            "CV_%": safe_cv_percent(seed_df[col])
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n[MULTI-SEED RUN TABLE]")
    print(seed_df.to_string(index=False))

    print("\n[MULTI-SEED STABILITY SUMMARY]")
    print(summary_df.to_string(index=False))

    return seed_df, summary_df, reference_front_df, combined_seed_fronts_df

def build_fixed_boundary_condition_table(baseline: CFSProjectBaseline,
                                         ts: str) -> pd.DataFrame:
    """
    Chapter 4 fixed boundary and S2 reference table.

    This table records the fixed case-study boundary, model-calculated
    S2 benchmark values, and excluded scope for the reported model run.
    """

    rows = [
        {
            "Boundary_item": "Case-study type",
            "Value": "Single-storey CFS residential framing case",
            "Unit": "-",
            "Status_in_model": "Fixed",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Footprint area",
            "Value": baseline.footprint_area_m2,
            "Unit": "m2",
            "Status_in_model": "Fixed",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Envelope area",
            "Value": baseline.envelope_area_m2,
            "Unit": "m2",
            "Status_in_model": "Fixed",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Total wall panels",
            "Value": baseline.num_wall_panels,
            "Unit": "count",
            "Status_in_model": "Fixed",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Total roof trusses",
            "Value": baseline.num_roof_trusses,
            "Unit": "count",
            "Status_in_model": "Fixed",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Baseline steel mass",
            "Value": baseline.W_base,
            "Unit": "kg",
            "Status_in_model": "Fixed reference quantity",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "CFS section source",
            "Value": baseline.cfs_section_source,
            "Unit": "-",
            "Status_in_model": "Fixed",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "CFS section designation",
            "Value": baseline.cfs_section_designation,
            "Unit": "-",
            "Status_in_model": "Fixed",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Steel thickness",
            "Value": baseline.assumed_steel_thickness_mm,
            "Unit": "mm",
            "Status_in_model": "Fixed; not optimised",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Steel grade",
            "Value": baseline.steel_grade,
            "Unit": "-",
            "Status_in_model": "Fixed",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Coating class",
            "Value": baseline.coating_class,
            "Unit": "-",
            "Status_in_model": "Fixed",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Panel height",
            "Value": baseline.assumed_panel_height_m,
            "Unit": "m",
            "Status_in_model": "Fixed project boundary condition",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Stud spacing",
            "Value": baseline.assumed_stud_spacing_mm,
            "Unit": "mm",
            "Status_in_model": "Fixed project boundary condition",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Maximum panel width",
            "Value": baseline.max_panel_width_m,
            "Unit": "m",
            "Status_in_model": "Hard feasibility constraint",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "Maximum lift mass",
            "Value": baseline.max_lift_mass_kg,
            "Unit": "kg",
            "Status_in_model": "Hard feasibility constraint",
            "Output_evidence": f"fixed_boundary_conditions_{ts}.csv",
        },
        {
            "Boundary_item": "S2 reference vector",
            "Value": str(baseline.x_ref),
            "Unit": "dimensionless / fraction",
            "Status_in_model": "Reference design vector",
            "Output_evidence": f"s2_reference_values_{ts}.csv",
        },
        {
            "Boundary_item": "S2 CFS framing process cost",
            "Value": round(float(baseline.C_S2_reference), 2),
            "Unit": "NZD",
            "Status_in_model": "Model-calculated reference objective",
            "Output_evidence": f"s2_reference_values_{ts}.csv",
        },
        {
            "Boundary_item": "S2 CFS framing A1-A5 carbon",
            "Value": round(float(baseline.CO2_S2_reference), 2),
            "Unit": "kgCO2e",
            "Status_in_model": "Model-calculated reference objective",
            "Output_evidence": f"s2_reference_values_{ts}.csv",
        },
        {
            "Boundary_item": "S2 CFS framing process-hours",
            "Value": round(float(baseline.Time_S2_reference), 2),
            "Unit": "h",
            "Status_in_model": "Model-calculated reference objective",
            "Output_evidence": f"s2_reference_values_{ts}.csv",
        },
        {
            "Boundary_item": "Optimised building geometry",
            "Value": "Not varied",
            "Unit": "-",
            "Status_in_model": "Excluded from optimisation",
            "Output_evidence": f"validation_status_{ts}.csv",
        },
        {
            "Boundary_item": "Structural member sizing",
            "Value": "Not varied",
            "Unit": "-",
            "Status_in_model": "Excluded from optimisation",
            "Output_evidence": f"validation_status_{ts}.csv",
        },
        {
            "Boundary_item": "Whole-building LCA scope",
            "Value": "Not included",
            "Unit": "-",
            "Status_in_model": "Outside model boundary",
            "Output_evidence": f"validation_status_{ts}.csv",
        },
        {
            "Boundary_item": "Contractor tender benchmarking",
            "Value": "Not included",
            "Unit": "-",
            "Status_in_model": "Outside model boundary",
            "Output_evidence": f"validation_status_{ts}.csv",
        },
    ]

    out = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"fixed_boundary_conditions_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[FIXED BOUNDARY CONDITIONS AND S2 REFERENCE]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def build_active_input_inventory_table(baseline: CFSProjectBaseline,
                                       ts: str) -> pd.DataFrame:
    """
    Chapter 4 input/configuration inventory for the reported model run.
    """

    rows = [
        {
            "Input_or_configuration_source": "CFSProjectBaseline",
            "Active_values_taken_from": "Python model configuration",
            "Content": "Project geometry, panel/truss quantities, baseline steel mass, and S2 reference vector",
            "Model_use": "Fixed CFS framing case-study boundary",
            "Legacy_CSV_dependency": "None",
            "Output_evidence": f"active_input_inventory_{ts}.csv",
        },
        {
            "Input_or_configuration_source": "CFSProjectBaseline",
            "Active_values_taken_from": "Python model configuration",
            "Content": "Cost coefficients and uncertainty ranges",
            "Model_use": "CFS framing process cost and Monte Carlo uncertainty",
            "Legacy_CSV_dependency": "None",
            "Output_evidence": f"calibration_audit_{ts}.csv",
        },
        {
            "Input_or_configuration_source": "CFSProjectBaseline",
            "Active_values_taken_from": "Python model configuration",
            "Content": "Carbon coefficients and uncertainty ranges",
            "Model_use": "CFS framing A1-A5 carbon and Monte Carlo uncertainty",
            "Legacy_CSV_dependency": "None",
            "Output_evidence": f"calibration_audit_{ts}.csv",
        },
        {
            "Input_or_configuration_source": "CFSProjectBaseline",
            "Active_values_taken_from": "Python model configuration",
            "Content": "Factory, logistics, site installation, mobilisation, and weather time parameters",
            "Model_use": "CFS framing process-hours and robustness analysis",
            "Legacy_CSV_dependency": "None",
            "Output_evidence": f"calibration_audit_{ts}.csv",
        },
        {
            "Input_or_configuration_source": "CFSProjectBaseline",
            "Active_values_taken_from": "Python model configuration",
            "Content": "Decision-variable ranges for R_p, R_t, OC, and P_f",
            "Model_use": "NSGA-II search space and LHS sampling bounds",
            "Legacy_CSV_dependency": "None",
            "Output_evidence": f"design_variable_table_{ts}.csv",
        },
        {
            "Input_or_configuration_source": "CFSObjectiveFunctions",
            "Active_values_taken_from": "Objective-function engine",
            "Content": "Cost, carbon, time, material-weight, feasibility, and component-breakdown calculations",
            "Model_use": "Objective evaluation, Pareto-front extraction, and dashboard outputs",
            "Legacy_CSV_dependency": "None",
            "Output_evidence": f"pareto_front_{ts}.csv",
        },
        {
            "Input_or_configuration_source": "CFSObjectiveFunctions reporting mappings",
            "Active_values_taken_from": "Current mapping functions",
            "Content": "Derived panel_scheme and truss_scheme labels from continuous R_p and R_t",
            "Model_use": "Tables, plots, and dashboard filtering",
            "Legacy_CSV_dependency": "None",
            "Output_evidence": f"pareto_front_{ts}.csv",
        },
    ]

    out = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"active_input_inventory_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[ACTIVE INPUT INVENTORY]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def build_s2_reference_table(baseline: CFSProjectBaseline,
                             obj_func: CFSObjectiveFunctions,
                             ts: str) -> pd.DataFrame:
    """
    Records the model-calculated S2 benchmark values for the reported model run.
    """

    x_s2 = list(baseline.x_ref)

    C_s2 = obj_func.calculate_cost_abs(x_s2)
    E_s2 = obj_func.calculate_carbon_abs(x_s2)
    T_s2 = obj_func.calculate_time_abs(x_s2)

    rows = [
        {
            "Reference_item": "S2 design vector",
            "Value": str(x_s2),
            "Unit": "dimensionless / fraction",
            "Source_in_current_code": "baseline.x_ref",
            "Model_use": "Reference configuration for normalisation and comparison",
        },
        {
            "Reference_item": "S2 CFS framing process cost",
            "Value": round(float(C_s2), 2),
            "Unit": "NZD",
            "Source_in_current_code": "obj_func.calculate_cost_abs(baseline.x_ref)",
            "Model_use": "Cost denominator for normalisation and S2 percentage comparison",
        },
        {
            "Reference_item": "S2 CFS framing A1-A5 carbon",
            "Value": round(float(E_s2), 2),
            "Unit": "kgCO2e",
            "Source_in_current_code": "obj_func.calculate_carbon_abs(baseline.x_ref)",
            "Model_use": "Carbon denominator for normalisation and S2 percentage comparison",
        },
        {
            "Reference_item": "S2 CFS framing process-hours",
            "Value": round(float(T_s2), 2),
            "Unit": "h",
            "Source_in_current_code": "obj_func.calculate_time_abs(baseline.x_ref)",
            "Model_use": "Time denominator for normalisation and S2 percentage comparison",
        },
        {
            "Reference_item": "S2 data status",
            "Value": baseline.S2_reference_data_status,
            "Unit": "-",
            "Source_in_current_code": "baseline.S2_reference_data_status",
            "Model_use": "Scope and limitation statement",
        },
        {
            "Reference_item": "S2 cost scope",
            "Value": baseline.S2_cost_scope,
            "Unit": "-",
            "Source_in_current_code": "baseline.S2_cost_scope",
            "Model_use": "Cost boundary statement",
        },
        {
            "Reference_item": "S2 carbon scope",
            "Value": baseline.S2_carbon_scope,
            "Unit": "-",
            "Source_in_current_code": "baseline.S2_carbon_scope",
            "Model_use": "Carbon boundary statement",
        },
        {
            "Reference_item": "S2 time scope",
            "Value": baseline.S2_time_scope,
            "Unit": "-",
            "Source_in_current_code": "baseline.S2_time_scope",
            "Model_use": "Time boundary statement",
        },
    ]

    out = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"s2_reference_values_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[S2 REFERENCE VALUES]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out


def build_current_parameter_dictionary(baseline: CFSProjectBaseline) -> dict:
    """
    Builds the calibration-audit parameter dictionary from current code values.
    No legacy CSV files are used.
    """

    return {
        "footprint_area_m2": {
            "value": baseline.footprint_area_m2,
            "unit": "m2",
            "range": "Not applicable",
            "source_basis": "Current code-defined case-study geometry",
            "validation_status": "Model configuration value",
            "model_use": "Fixed CFS framing boundary",
        },
        "envelope_area_m2": {
            "value": baseline.envelope_area_m2,
            "unit": "m2",
            "range": "Not applicable",
            "source_basis": "Current code-defined envelope/framing area",
            "validation_status": "Model configuration value",
            "model_use": "A5 carbon and reporting normalisation",
        },
        "num_wall_panels": {
            "value": baseline.num_wall_panels,
            "unit": "count",
            "range": "Not applicable",
            "source_basis": "Current code-defined case-study panel quantity",
            "validation_status": "Model configuration value",
            "model_use": "Panel rationalisation and constructability metrics",
        },
        "num_roof_trusses": {
            "value": baseline.num_roof_trusses,
            "unit": "count",
            "range": "Not applicable",
            "source_basis": "Current code-defined case-study truss quantity",
            "validation_status": "Model configuration value",
            "model_use": "Truss rationalisation and constructability metrics",
        },
        "W_base": {
            "value": baseline.W_base,
            "unit": "kg",
            "range": "Not applicable",
            "source_basis": "Current code-defined baseline CFS steel mass",
            "validation_status": "Model configuration value",
            "model_use": "Material cost and A1-A3 steel carbon",
        },
        "raw_steel_material_nzd_per_kg": {
            "value": baseline.raw_steel_material_nzd_per_kg,
            "unit": "NZD/kg",
            "range": str(baseline.raw_steel_material_nzd_per_kg_range),
            "source_basis": "Current code-defined market-rate assumption",
            "validation_status": "Uncertainty-tested parameter",
            "model_use": "CFS framing process cost",
        },
        "factory_processing_nzd_per_kg": {
            "value": baseline.factory_processing_nzd_per_kg,
            "unit": "NZD/kg",
            "range": str(baseline.factory_processing_nzd_per_kg_range),
            "source_basis": "Current code-defined factory processing assumption",
            "validation_status": "Uncertainty-tested parameter",
            "model_use": "CFS framing process cost",
        },
        "site_labor_rate_nzd_per_hr": {
            "value": baseline.site_labor_rate_nzd_per_hr,
            "unit": "NZD/h",
            "range": str(baseline.site_labor_rate_nzd_per_hr_range),
            "source_basis": "Current code-defined labour-rate assumption",
            "validation_status": "Uncertainty-tested parameter",
            "model_use": "CFS framing process cost and process-hours coupling",
        },
        "transport_cost_nzd_per_tonne_km": {
            "value": baseline.transport_cost_nzd_per_tonne_km,
            "unit": "NZD/t-km",
            "range": str(baseline.transport_cost_nzd_per_tonne_km_range),
            "source_basis": "Current code-defined freight cost assumption",
            "validation_status": "Uncertainty-tested parameter",
            "model_use": "Transport cost",
        },
        "steel_carbon_factor_a1_a3_kgco2e_per_kg": {
            "value": baseline.steel_carbon_factor_a1_a3_kgco2e_per_kg,
            "unit": "kgCO2e/kg",
            "range": str(baseline.steel_carbon_factor_a1_a3_range),
            "source_basis": "Current code-defined A1-A3 steel carbon factor",
            "validation_status": "Uncertainty-tested parameter",
            "model_use": "CFS framing A1-A5 carbon",
        },
        "freight_kgco2e_per_tonne_km": {
            "value": baseline.freight_kgco2e_per_tonne_km,
            "unit": "kgCO2e/t-km",
            "range": str(baseline.freight_kgco2e_per_tonne_km_range),
            "source_basis": "Current code-defined A4 freight carbon factor",
            "validation_status": "Uncertainty-tested parameter",
            "model_use": "A4 transport carbon",
        },
        "a5_installation_carbon_kgco2e_per_m2": {
            "value": baseline.a5_installation_carbon_kgco2e_per_m2,
            "unit": "kgCO2e/m2",
            "range": str(baseline.a5_installation_carbon_range),
            "source_basis": "Current code-defined A5 installation carbon factor",
            "validation_status": "Uncertainty-tested parameter",
            "model_use": "A5 installation carbon",
        },
        "T_factory_truss_base": {
            "value": baseline.T_factory_truss_base,
            "unit": "h",
            "range": "Not specified",
            "source_basis": "Current code-defined process-hour assumption",
            "validation_status": "Model configuration value",
            "model_use": "CFS framing process-hours",
        },
        "T_factory_panel_base": {
            "value": baseline.T_factory_panel_base,
            "unit": "h",
            "range": "Not specified",
            "source_basis": "Current code-defined process-hour assumption",
            "validation_status": "Model configuration value",
            "model_use": "CFS framing process-hours",
        },
        "T_installation_truss_base": {
            "value": baseline.T_installation_truss_base,
            "unit": "h",
            "range": "Not specified",
            "source_basis": "Current code-defined process-hour assumption",
            "validation_status": "Model configuration value",
            "model_use": "CFS framing process-hours",
        },
        "T_installation_panel_base": {
            "value": baseline.T_installation_panel_base,
            "unit": "h",
            "range": "Not specified",
            "source_basis": "Current code-defined process-hour assumption",
            "validation_status": "Model configuration value",
            "model_use": "CFS framing process-hours",
        },
        "k_prefab_site_time_saving": {
            "value": baseline.k_prefab_site_time_saving,
            "unit": "dimensionless",
            "range": str(baseline.k_prefab_time_saving_range),
            "source_basis": "Current code-defined prefabrication response coefficient",
            "validation_status": "Uncertainty-tested response coefficient",
            "model_use": "CFS framing process-hours",
        },
        "k_prefab_a5_carbon_saving": {
            "value": baseline.k_prefab_a5_carbon_saving,
            "unit": "dimensionless",
            "range": str(baseline.k_prefab_a5_carbon_saving_range),
            "source_basis": "Current code-defined prefabrication carbon response coefficient",
            "validation_status": "Uncertainty-tested response coefficient",
            "model_use": "CFS framing A1-A5 carbon",
        },
        "k_logistics_cost": {
            "value": baseline.k_logistics_cost,
            "unit": "dimensionless",
            "range": str(baseline.k_logistics_cost_range),
            "source_basis": "Current code-defined logistics response coefficient",
            "validation_status": "Uncertainty-tested response coefficient",
            "model_use": "CFS framing process cost",
        },
        "k_logistics_time": {
            "value": baseline.k_logistics_time,
            "unit": "dimensionless",
            "range": str(baseline.k_logistics_time_range),
            "source_basis": "Current code-defined logistics response coefficient",
            "validation_status": "Uncertainty-tested response coefficient",
            "model_use": "CFS framing process-hours",
        },
        "k_logistics_carbon": {
            "value": baseline.k_logistics_carbon,
            "unit": "dimensionless",
            "range": str(baseline.k_logistics_carbon_range),
            "source_basis": "Current code-defined logistics carbon response coefficient",
            "validation_status": "Uncertainty-tested response coefficient",
            "model_use": "CFS framing A1-A5 carbon",
        },
        "opening_complexity_range": {
            "value": str(baseline.opening_complexity_range),
            "unit": "dimensionless",
            "range": str(baseline.opening_complexity_range),
            "source_basis": "Current code-defined decision-variable range",
            "validation_status": "Optimisation search bound",
            "model_use": "Opening/detailing complexity variable",
        },
        "P_f_range": {
            "value": str(baseline.P_f_range),
            "unit": "fraction",
            "range": str(baseline.P_f_range),
            "source_basis": "Current code-defined decision-variable range",
            "validation_status": "Optimisation search bound",
            "model_use": "Prefabrication-level variable",
        },
        "R_p_range": {
            "value": str(baseline.R_p_range),
            "unit": "dimensionless",
            "range": str(baseline.R_p_range),
            "source_basis": "Current code-defined decision-variable range",
            "validation_status": "Optimisation search bound",
            "model_use": "Panel rationalisation variable",
        },
        "R_t_range": {
            "value": str(baseline.R_t_range),
            "unit": "dimensionless",
            "range": str(baseline.R_t_range),
            "source_basis": "Current code-defined decision-variable range",
            "validation_status": "Optimisation search bound",
            "model_use": "Truss rationalisation variable",
        },
    }

def build_solution_count_audit(eval_audit: dict,
                               final_archive,
                               pdf: pd.DataFrame,
                               rep_df: pd.DataFrame,
                               ts: str) -> pd.DataFrame:
    """
    Separates evaluated candidates, feasible candidates, nondominated archive,
    final analysed Pareto set, and representative selected solutions.

    This prevents the thesis from confusing feasible candidates with Pareto solutions.
    """

    rows = [
        {
            "Metric": "Total evaluated fitness calls",
            "Value": int(eval_audit.get("total_evaluated_fitness_calls", 0)),
            "Meaning": "All candidate designs evaluated by NSGA-II, including repeated offspring evaluations."
        },
        {
            "Metric": "Hard-feasible evaluated candidates",
            "Value": int(eval_audit.get("hard_feasible_evaluated", 0)),
            "Meaning": "Evaluated candidates satisfying hard NZ constructability constraints."
        },
        {
            "Metric": "Hard-infeasible evaluated candidates",
            "Value": int(eval_audit.get("hard_infeasible_evaluated", 0)),
            "Meaning": "Evaluated candidates violating at least one hard constraint."
        },
        {
            "Metric": "External nondominated archive size",
            "Value": len(final_archive),
            "Meaning": "DEAP ParetoFront archive returned by the optimizer."
        },
        {
            "Metric": "Final analysed nondominated Pareto solutions",
            "Value": len(pdf),
            "Meaning": "Finite nondominated solutions after physical-objective filtering."
        },
        {
            "Metric": "Representative selected solutions",
            "Value": len(rep_df),
            "Meaning": "Selected solutions such as Min Cost, Min Carbon, Min Time, Balanced, etc."
        },
    ]

    out = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"solution_count_audit_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[SOLUTION COUNT AUDIT]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def build_prefab_boundary_diagnostic(pdf: pd.DataFrame,
                                     baseline: CFSProjectBaseline,
                                     ts: str) -> pd.DataFrame:
    """
    Checks whether P_f collapses to lower/upper bound.
    This diagnoses whether prefabrication is overpowering the optimizer.
    """

    var = "P_f_prefab_level"
    lower, upper = baseline.P_f_range
    tol = 1e-6

    at_lower = np.isclose(pdf[var], lower, atol=tol).mean() * 100.0
    at_upper = np.isclose(pdf[var], upper, atol=tol).mean() * 100.0

    out = pd.DataFrame([
        {
            "Variable": var,
            "Lower_bound": lower,
            "Upper_bound": upper,
            "Min_on_Pareto": pdf[var].min(),
            "Max_on_Pareto": pdf[var].max(),
            "Mean_on_Pareto": pdf[var].mean(),
            "Percent_at_lower_bound": at_lower,
            "Percent_at_upper_bound": at_upper,
            "Interpretation": (
                "If most solutions are at the upper bound, prefabrication may be too dominant. "
                "If solutions spread across the range, P_f is acting as a trade-off variable."
            )
        }
    ])

    out_path = TABLE_DIR / f"prefab_boundary_diagnostic_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[PREFABRICATION BOUNDARY DIAGNOSTIC]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def build_nz_constraint_summary(pdf: pd.DataFrame, ts: str) -> pd.DataFrame:
    """
    Summarises hard and soft NZ thesis constraints across final Pareto solutions.
    """
    rows = []

    rows.append({
        "Metric": "Pareto solutions",
        "Value": len(pdf)
    })

    if "NZ_feasible" in pdf.columns:
        rows.append({
            "Metric": "Hard-feasible solutions",
            "Value": int(pdf["NZ_feasible"].sum())
        })

        rows.append({
            "Metric": "Hard-infeasible solutions",
            "Value": int((~pdf["NZ_feasible"]).sum())
        })

    for col in [
        "Max_panel_width_m",
        "Panel_height_m",
        "Stud_spacing_mm",
        "Steel_thickness_mm",
        "Panel_lift_mass_kg",
        "Factory_repetition_score",
        "Carbon_intensity_kgCO2e_m2",
        "Total_constraint_penalty",
    ]:
        if col in pdf.columns:
            rows.append({
                "Metric": f"{col}_min",
                "Value": round(float(pdf[col].min()), 4)
            })
            rows.append({
                "Metric": f"{col}_max",
                "Value": round(float(pdf[col].max()), 4)
            })

    if "Hard_violations" in pdf.columns:
        for k, v in pdf["Hard_violations"].value_counts().items():
            rows.append({
                "Metric": f"Hard violation pattern: {k}",
                "Value": int(v)
            })

    if "Soft_violations" in pdf.columns:
        for k, v in pdf["Soft_violations"].value_counts().items():
            rows.append({
                "Metric": f"Soft violation pattern: {k}",
                "Value": int(v)
            })

    out = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"nz_constraint_summary_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[NZ THESIS CONSTRAINT SUMMARY]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def thin_hv_points(pts: np.ndarray, decimals: int = 4, max_points: int = 3000) -> np.ndarray:
    """
    Deduplicate and cap physical objective points for faster HV calculation.
    Used only for convergence plotting, not for final Pareto CSV.
    """
    pts = np.asarray(pts, dtype=float)

    if len(pts) == 0:
        return pts

    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) == 0:
        return pts

    pts_round = np.round(pts, decimals=decimals)
    _, unique_idx = np.unique(pts_round, axis=0, return_index=True)
    pts = pts[np.sort(unique_idx)]

    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]

    return pts

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_optimization(pop_size=200, generations=300, seed=42):
    
    print("CFS MULTI-OBJECTIVE OPTIMISATION")
   
    random.seed(seed)
    np.random.seed(seed)
    baseline = CFSProjectBaseline()
    validate_scheme_counts(baseline)
    tb, obj  = setup_nsga2(baseline)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_audit = {
        "total_evaluated_fitness_calls": 0,
        "hard_feasible_evaluated": 0,
        "hard_infeasible_evaluated": 0,
    }

    def record_evaluation(ind):
        """
        Count evaluated candidate solutions and feasibility status.
        This avoids confusing evaluated/feasible candidates with Pareto solutions.
        """
        eval_audit["total_evaluated_fitness_calls"] += 1

        x = decode_individual(ind, baseline)
        C_abs = obj.calculate_cost_abs(x)
        E_abs = obj.calculate_carbon_abs(x)
        T_abs = obj.calculate_time_abs(x)

        report = obj.nz_constraint_report(x, C_abs, E_abs, T_abs)

        if report["NZ_feasible"]:
            eval_audit["hard_feasible_evaluated"] += 1
        else:
            eval_audit["hard_infeasible_evaluated"] += 1

    # Validate S2 benchmark consistency.
    obj._ensure_refs()
    # Synchronise all downstream reporting with the actual objective-function S2 values.
    # This avoids old baseline references being used in plots/tables after the
    # component objective functions have been changed.
    baseline.C_S2_reference = obj._C_ref
    baseline.CO2_S2_reference = obj._CO2_ref
    baseline.Time_S2_reference = obj._T_ref

    baseline.C_baseline = obj._C_ref
    baseline.CO2_baseline = obj._CO2_ref
    baseline.Time_baseline = obj._T_ref
    err = max(
        abs(obj.calculate_cost_abs(baseline.x_ref) / obj._C_ref - 1.0),
        abs(obj.calculate_carbon_abs(baseline.x_ref) / obj._CO2_ref - 1.0),
        abs(obj.calculate_time_abs(baseline.x_ref) / obj._T_ref - 1.0)
    )
    if err > 1e-6:
        raise ValueError(f"S2 benchmark consistency error {err:.2e} > tolerance")

    print(f"\n[OK] S2 benchmark consistency error = {err:.2e}")
    print("[NOTE] No S2 calibration scaling is used. S2 is used only for post-optimisation comparison.")

    print("\n[S2 MODEL-CALCULATED BENCHMARK CHECK]")
    print(f"  Objective cost at x_ref:   {obj._C_ref:.2f} NZD")
    print(f"  S2 benchmark cost:         {baseline.C_S2_reference:.2f} NZD")

    print(f"  Objective carbon at x_ref: {obj._CO2_ref:.2f} kgCO2e")
    print(f"  S2 benchmark carbon:       {baseline.CO2_S2_reference:.2f} kgCO2e")

    print(f"  Objective time at x_ref:   {obj._T_ref:.2f} hrs")
    print(f"  S2 benchmark time:         {baseline.Time_S2_reference:.2f} hrs")
    print(f"  Component-sum time diagnostic: {baseline.Time_S2_reference:.2f} hrs")
    print("\n[S2 MODEL-CALCULATED REFERENCE BREAKDOWN]")

    print("\n  Cost components:")
    print(f"    Steel material:        {baseline.C_ref_steel_material:.2f} NZD")
    print(f"    Factory processing:    {baseline.C_ref_factory_processing:.2f} NZD")
    print(f"    Site labour:           {baseline.C_ref_site_installation_labour:.2f} NZD")
    print(f"    Factory setup:         {baseline.C_ref_factory_setup:.2f} NZD")
    print(f"    Transport:             {baseline.C_ref_transport:.2f} NZD")
    print(f"    Delivery/logistics:    {baseline.C_ref_delivery_logistics:.2f} NZD")
    print(f"    Site mobilisation:     {baseline.C_ref_site_mobilisation:.2f} NZD")
    print(f"    Waste handling:        {baseline.C_ref_waste_handling:.2f} NZD")
    print(f"    Lifting equipment:     {baseline.C_ref_lifting_equipment:.2f} NZD")
    print(f"    Overhead:              {baseline.C_ref_overhead:.2f} NZD")
    print(f"    TOTAL C_S2_reference:  {baseline.C_S2_reference:.2f} NZD")

    print("\n  Carbon components:")
    print(f"    A1-A3 steel:           {baseline.CO2_ref_A1_A3_steel:.2f} kgCO2e")
    print(f"    A4 transport:          {baseline.CO2_ref_A4_transport:.2f} kgCO2e")
    print(f"    Factory energy:        {baseline.CO2_ref_factory_energy:.2f} kgCO2e")
    print(f"    Site energy:           {baseline.CO2_ref_site_energy:.2f} kgCO2e")
    print(f"    A5 installation:       {baseline.CO2_ref_A5_installation:.2f} kgCO2e")
    print(f"    Waste allowance:       {baseline.CO2_ref_waste:.2f} kgCO2e")
    print(f"    TOTAL CO2_S2_reference:{baseline.CO2_S2_reference:.2f} kgCO2e")

    print("\n[S2 REFERENCE SCOPE]")
    print(f"  Data status:  {baseline.S2_reference_data_status}")
    print(f"  Cost scope:   {baseline.S2_cost_scope}")
    print(f"  Carbon scope: {baseline.S2_carbon_scope}")
    print(f"  Time scope:   {baseline.S2_time_scope}")
   
    # Initial population
    if INIT_METHOD.lower() == "lhs":
        pop = tb.lhs_population(
            n_individuals=pop_size,
            seed=seed
        )
        init_audit_df = audit_initial_population(
            pop,
            baseline,
            ts,
            method_name="lhs"
        )

    elif INIT_METHOD.lower() == "random":
        pop = tb.population(n=pop_size)
        init_audit_df = audit_initial_population(
            pop,
            baseline,
            ts,
            method_name="random"
        )

    else:
        raise ValueError(f"Unknown INIT_METHOD: {INIT_METHOD}. Use 'lhs' or 'random'.")

    for ind, fit in zip(pop, map(tb.evaluate, pop)):
        ind.fitness.values = fit
        record_evaluation(ind)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

  
    print("\n[INFO] Building fixed hypervolume reference box from physical grid objectives ...")
    
    gdf_for_hv, gpdf_for_hv = run_grid_validation(obj, baseline, pf_step=GRID_PF_STEP)

    hv_cols = ['Cost_normalized', 'Carbon_normalized', 'Time_normalized']

    hv_base = (
        gdf_for_hv[hv_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # remove extreme physical outliers
    hv_base = hv_base[(hv_base[hv_cols] < 1.50).all(axis=1)]

    if len(hv_base) == 0:
        raise ValueError("No finite physical grid points available for hypervolume reference box.")

    hv_ideal = hv_base[hv_cols].min().values * 0.995
    hv_ref = hv_base[hv_cols].max().values * 1.050

    # reference must be worse than ideal
    hv_ref = np.maximum(hv_ref, hv_ideal + 1e-6)
    
    hv_rng = np.random.default_rng(42)
    hv_samples = hv_rng.uniform(hv_ideal, hv_ref, size=(HV_SAMPLES, 3))
    print(f"[HV BOX] ideal = {hv_ideal}")
    print(f"[HV BOX] ref   = {hv_ref}")

    # External archive keeps non-dominated solutions discovered across generations.
    archive = tools.ParetoFront()
    archive.update(pop)

    hv_hist = []
    print(f"\n[INFO] Running {generations} generations × {pop_size} population …")
    for gen in range(generations):
        off = [tb.clone(ind) for ind in tb.select(pop, len(pop))]
        for i in range(1, len(off), 2):
            if random.random() < 0.7:
                off[i-1], off[i] = tb.mate(off[i-1], off[i])
                del off[i-1].fitness.values, off[i].fitness.values
        for ind in off:
            if random.random() < 0.3:
                ind, = tb.mutate(ind); del ind.fitness.values
        inv = [ind for ind in off if not ind.fitness.valid]
        for ind, fit in zip(inv, map(tb.evaluate, inv)):
            ind.fitness.values = fit
            record_evaluation(ind)
        pop = tb.select(pop + off, pop_size)
        rec = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(inv), **rec)
        archive.update(pop)

        if (gen + 1) % HV_EVERY == 0:
            archive_pts = get_archive_physical_objective_array(
                archive,
                obj_func=obj,
                baseline=baseline,
                max_reasonable_norm=1.50
            )

            archive_pts = thin_hv_points(
                archive_pts,
                decimals=4,
                max_points=1000 if RUN_MODE == "debug" else 3000
            )

            if len(archive_pts) == 0:
                print(f"[HV WARNING] Generation {gen+1}: no finite physical archive points.")
                hv = hv_hist[-1]['hypervolume'] if len(hv_hist) > 0 else 0.0
                physical_archive_size = 0
            else:
                hv = approximate_hypervolume_from_fixed_samples(
                    archive_pts,
                    ideal=hv_ideal,
                    ref=hv_ref,
                    samples=hv_samples
                )
                physical_archive_size = len(archive_pts)

            # Numerical safety: archive HV should not decrease.
            if len(hv_hist) > 0:
                hv = max(hv, hv_hist[-1]['hypervolume'])

            hv_hist.append({
                'generation': gen + 1,
                'hypervolume': hv,
                'pareto_size': len(archive),
                'physical_archive_size': physical_archive_size
            })

        if (gen + 1) % 50 == 0:
            current_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            print(
                f"  Gen {gen+1:3d}/{generations}: "
                f"Current_front={len(current_front)} | "
                f"Archive={len(archive)} | "
                f"Min cost={rec['min'][0]:.4f}"
            )
    print("\n[INFO] Optimisation complete.")

    # Return archive for final Pareto analysis, not only the last population.
    final_archive = list(archive)

    return final_archive, logbook, tb, baseline, obj, pd.DataFrame(hv_hist), ts, eval_audit

def build_pareto_branch_summary(pdf: pd.DataFrame, ts: str) -> pd.DataFrame:
    """
    Summarises discrete Pareto branches created by panel/truss scheme choices.
    """
    branch_df = (
        pdf.groupby(['panel_scheme', 'truss_scheme'])
        .agg(
            n_solutions=('Solution_ID', 'count'),
            P_f_min=('P_f_prefab_level', 'min'),
            P_f_max=('P_f_prefab_level', 'max'),
            OC_min=('opening_complexity_index', 'min'),
            OC_max=('opening_complexity_index', 'max'),
            Cost_min_NZD=('Cost_NZD', 'min'),
            Cost_max_NZD=('Cost_NZD', 'max'),
            Carbon_min_kgCO2e=('Carbon_kgCO2e', 'min'),
            Carbon_max_kgCO2e=('Carbon_kgCO2e', 'max'),
            Time_min_hr=('Time_hours', 'min'),
            Time_max_hr=('Time_hours', 'max'),
        )
        .reset_index()
    )

    for col in branch_df.columns:
        if col not in ['panel_scheme', 'truss_scheme', 'n_solutions']:
            branch_df[col] = branch_df[col].round(3)

    out_path = TABLE_DIR / f"pareto_branch_summary_{ts}.csv"
    branch_df.to_csv(out_path, index=False)

    print("\n[PARETO BRANCH SUMMARY]")
    print(branch_df.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return branch_df

def plot_pareto_2d_by_panel_scheme(df, path):
    """
    Pareto front coloured by panel_scheme to explain disconnected branches.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    pairs = [
        ('Cost_normalized', 'Carbon_normalized', 'Cost vs Carbon'),
        ('Cost_normalized', 'Time_normalized', 'Cost vs Time'),
        ('Carbon_normalized', 'Time_normalized', 'Carbon vs Time')
    ]

    for ax, (x, y, title) in zip(axes, pairs):
        sc = ax.scatter(
            df[x],
            df[y],
            c=df['panel_scheme'],
            s=55,
            alpha=0.75,
            cmap='tab10',
            edgecolors='none'
        )

        ax.set_xlabel(x.replace('_', ' '), fontweight='bold')
        ax.set_ylabel(y.replace('_', ' '), fontweight='bold')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=axes[2])
    cbar.set_label('panel_scheme')

    fig.suptitle(
        'Pareto Front Branch Structure\n(Colour = panel_scheme)',
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {path}")

def diagnose_objective_dataframe(df: pd.DataFrame, name: str,
                                 cols=('Cost_normalized', 'Carbon_normalized', 'Time_normalized'),
                                 ts: str = "debug"):
    """
    Prints and saves rows containing NaN/inf objective values.
    This should be called before IGD, KMeans, TOPSIS, and plotting.
    """
    print(f"\n[FINITE CHECK] {name}")

    if df is None or len(df) == 0:
        print(f"  [ERROR] {name} is empty.")
        return False

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        print(f"  [ERROR] Missing columns: {missing_cols}")
        return False

    X = df[list(cols)].replace([np.inf, -np.inf], np.nan)

    print(X.describe().to_string())

    bad_mask = X.isna().any(axis=1)
    n_bad = int(bad_mask.sum())

    if n_bad > 0:
        print(f"\n  [ERROR] {n_bad} rows contain NaN/inf in {cols}")

        debug_cols = list(cols)
        for extra in [
            'Solution_ID', 'panel_scheme', 'truss_scheme',
            'opening_complexity_index', 'P_f_prefab_level',
            'Cost_NZD', 'Carbon_kgCO2e', 'Time_hours',
            'Cost_fitness', 'Carbon_fitness', 'Time_fitness',
            'Penalty_cost', 'Penalty_carbon', 'Penalty_time',
            'Total_constraint_penalty',
            'Hard_violations', 'Soft_violations'
        ]:
            if extra in df.columns and extra not in debug_cols:
                debug_cols.append(extra)

        bad_df = df.loc[bad_mask, debug_cols].copy()
        out_path = TABLE_DIR / f"debug_nonfinite_{name.replace(' ', '_')}_{ts}.csv"
        bad_df.to_csv(out_path, index=False)

        print(f"  [SAVED DEBUG ROWS] {out_path}")
        print(bad_df.head(20).to_string(index=False))
        return False

    print("  [OK] All objective values are finite.")
    return True    

def clean_finite_objective_rows(df: pd.DataFrame,
                                name: str,
                                ts: str,
                                cols=('Cost_normalized', 'Carbon_normalized', 'Time_normalized')) -> pd.DataFrame:
    """
    Removes rows with NaN/inf objective values and saves debug rows.
    Use before IGD, KMeans, TOPSIS, and plotting.
    """
    print(f"\n[FINITE CHECK] {name}")

    if df is None or len(df) == 0:
        print(f"  [ERROR] {name} is empty.")
        return df

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing objective columns: {missing}")

    X = df[list(cols)].replace([np.inf, -np.inf], np.nan)

    print(X.describe().to_string())

    finite_mask = X.notna().all(axis=1)
    n_bad = int((~finite_mask).sum())

    if n_bad > 0:
        print(f"  [WARNING] {n_bad} rows contain NaN/inf and will be dropped.")

        debug_cols = list(cols)
        for extra in [
            'Solution_ID',
            'panel_scheme',
            'truss_scheme',
            'opening_complexity_index',
            'P_f_prefab_level',
            'Cost_NZD',
            'Carbon_kgCO2e',
            'Time_hours',
            'Cost_fitness',
            'Carbon_fitness',
            'Time_fitness',
            'Penalty_cost',
            'Penalty_carbon',
            'Penalty_time',
            'Total_constraint_penalty',
            'Hard_violations',
            'Soft_violations'
        ]:
            if extra in df.columns and extra not in debug_cols:
                debug_cols.append(extra)

        bad_path = TABLE_DIR / f"debug_nonfinite_{name}_{ts}.csv"
        df.loc[~finite_mask, debug_cols].to_csv(bad_path, index=False)
        print(f"  [SAVED] {bad_path}")

    out = df.loc[finite_mask].copy().reset_index(drop=True)

    if len(out) == 0:
        raise ValueError(f"All rows in {name} have non-finite objective values.")

    print(f"  [OK] finite rows kept: {len(out)} / {len(df)}")

    return out

def add_carbon_scope_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds aliases for carbon scope columns to ensure consistency.
    Internal model columns remain:
        Carbon_kgCO2e
        Carbon_normalized
        Carbon_vs_S2_%
        CFS_framing_A1_A5_carbon_kgCO2e
        CFS_framing_A1_A5_carbon_normalized
        CFS_framing_A1_A5_carbon_vs_S2_%
    """

    out = df.copy()

    if "Carbon_kgCO2e" in out.columns:
        out["CFS_framing_A1_A5_carbon_kgCO2e"] = out["Carbon_kgCO2e"]

    if "Carbon_normalized" in out.columns:
        out["CFS_framing_A1_A5_carbon_normalized"] = out["Carbon_normalized"]

    if "Carbon_vs_S2_%" in out.columns:
        out["CFS_framing_A1_A5_carbon_vs_S2_%"] = out["Carbon_vs_S2_%"]

    return out

def _json_safe_value(value):
    """
    Converts NumPy/Pandas values into JSON-safe Python values.
    """
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.ndarray,)):
        return value.tolist()

    return value


def _json_safe_records(df: pd.DataFrame, max_rows=None):
    """
    Converts dataframe rows to JSON-safe list of dictionaries.
    Optionally limits the number of rows to keep dashboard files lightweight.
    """
    if df is None or len(df) == 0:
        return []

    out = df.copy()

    if max_rows is not None and len(out) > max_rows:
        out = out.head(max_rows).copy()

    records = []

    for row in out.to_dict(orient="records"):
        records.append({
            str(k): _json_safe_value(v)
            for k, v in row.items()
        })

    return records


def _get_attr(obj, name, default=None):
    """
    Safe attribute getter for baseline parameters.
    """
    return getattr(obj, name, default)

def _improvement_summary(df: pd.DataFrame, value_col: str, baseline_value: float) -> dict:
    """
    Positive value = percentage reduction relative to S2.
    Negative value = percentage increase relative to S2.
    """
    if df is None or len(df) == 0 or value_col not in df.columns:
        return {}

    vals = pd.to_numeric(df[value_col], errors="coerce").dropna()

    if len(vals) == 0 or baseline_value == 0:
        return {}

    improvement = (baseline_value - vals) / baseline_value * 100.0

    return {
        "min_reduction_pct": round(float(improvement.min()), 4),
        "max_reduction_pct": round(float(improvement.max()), 4),
        "mean_reduction_pct": round(float(improvement.mean()), 4),
    }


def build_dashboard_key_findings(pareto_df: pd.DataFrame,
                                 baseline: CFSProjectBaseline) -> dict:
    """
    Computes headline findings directly from the Pareto dataframe.
    This avoids manually typed thesis/dashboard claims.
    """

    if pareto_df is None or len(pareto_df) == 0:
        return {}

    df = pareto_df.copy()

    min_cost_row = df.loc[df["Cost_NZD"].idxmin()]
    min_carbon_row = df.loc[df["Carbon_kgCO2e"].idxmin()]
    min_time_row = df.loc[df["Time_hours"].idxmin()]

    s2_cost = float(baseline.C_S2_reference)
    s2_carbon = float(baseline.CO2_S2_reference)
    s2_time = float(baseline.Time_S2_reference)

    s2_above_min_cost_nzd = s2_cost - float(min_cost_row["Cost_NZD"])
    s2_above_min_cost_pct = s2_above_min_cost_nzd / s2_cost * 100.0

    panel_schemes = sorted([int(v) for v in df["panel_scheme"].dropna().unique()])
    truss_schemes = sorted([int(v) for v in df["truss_scheme"].dropna().unique()])

    branch_count = (
        df[["panel_scheme", "truss_scheme"]]
        .drop_duplicates()
        .shape[0]
    )

    oc_min = float(df["opening_complexity_index"].min())
    oc_max = float(df["opening_complexity_index"].max())

    pf_min = float(df["P_f_prefab_level"].min())
    pf_max = float(df["P_f_prefab_level"].max())

    return {
        "pareto_solution_count": int(len(df)),

        "s2_cost_above_min_cost_nzd": round(s2_above_min_cost_nzd, 2),
        "s2_cost_above_min_cost_pct": round(s2_above_min_cost_pct, 4),

        "min_cost_solution_id": str(min_cost_row.get("Solution_ID", "")),
        "min_cost_nzd": round(float(min_cost_row["Cost_NZD"]), 2),
        "min_cost_R_p_panel_rationalisation": round(float(min_cost_row["R_p_panel_rationalisation"]), 4),
        "min_cost_R_t_truss_rationalisation": round(float(min_cost_row["R_t_truss_rationalisation"]), 4),
        "min_cost_panel_scheme": int(min_cost_row["panel_scheme"]),
        "min_cost_truss_scheme": int(min_cost_row["truss_scheme"]),
        "min_cost_opening_complexity_index": round(float(min_cost_row["opening_complexity_index"]), 4),
        "min_cost_prefab_level": round(float(min_cost_row["P_f_prefab_level"]), 4),

        "min_carbon_solution_id": str(min_carbon_row.get("Solution_ID", "")),
        "min_carbon_kgco2e": round(float(min_carbon_row["Carbon_kgCO2e"]), 2),
        "min_carbon_prefab_level": round(float(min_carbon_row["P_f_prefab_level"]), 4),

        "min_time_solution_id": str(min_time_row.get("Solution_ID", "")),
        "min_time_hours": round(float(min_time_row["Time_hours"]), 2),
        "min_time_prefab_level": round(float(min_time_row["P_f_prefab_level"]), 4),

        "pareto_panel_schemes_present": panel_schemes,
        "pareto_truss_schemes_present": truss_schemes,
        "pareto_discrete_branch_count": int(branch_count),

        "opening_complexity_min": round(oc_min, 4),
        "opening_complexity_max": round(oc_max, 4),
        "opening_complexity_collapsed": bool(np.isclose(oc_min, oc_max, atol=1e-4)),

        "prefabrication_level_min": round(pf_min, 4),
        "prefabrication_level_max": round(pf_max, 4),

        "cost_improvement_summary_pct": _improvement_summary(df, "Cost_NZD", s2_cost),
        "carbon_improvement_summary_pct": _improvement_summary(df, "Carbon_kgCO2e", s2_carbon),
        "time_improvement_summary_pct": _improvement_summary(df, "Time_hours", s2_time),
    }

def export_dashboard_config(
    baseline,
    pareto_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    ts: str,
    sensitivity_df: pd.DataFrame = None,
    monte_carlo_df: pd.DataFrame = None,
    topsis_df: pd.DataFrame = None,
    output_dir: Path = DASHBOARD_DATA_DIR
):
    """
    Exports calibrated optimization outputs to dashboard_config.json.

    This is the key link between:
        Python NSGA-II optimizer
        → dashboard/data/dashboard_config.json
        → recalibratable decision dashboard

    The dashboard should use this JSON as its official data source.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    key_findings = build_dashboard_key_findings(
        pareto_df=pareto_df,
        baseline=baseline
    )

    calibration_export = {
        "transport_distance_km": float(_get_attr(baseline, "a4_default_transport_distance_km", 121.0)),
        "a4_default_transport_distance_km": float(_get_attr(baseline, "a4_default_transport_distance_km", 121.0)),
        "transport_cost_nzd_per_tonne_km": float(_get_attr(baseline, "transport_cost_nzd_per_tonne_km", 3.0)),
        "freight_kgco2e_per_tonne_km": float(_get_attr(baseline, "freight_kgco2e_per_tonne_km", 0.135)),

        "raw_steel_material_nzd_per_kg": float(_get_attr(baseline, "raw_steel_material_nzd_per_kg", 2.40)),
        "factory_processing_nzd_per_kg": float(_get_attr(baseline, "factory_processing_nzd_per_kg", 2.80)),
        "site_labor_rate_nzd_per_hr": float(_get_attr(baseline, "site_labor_rate_nzd_per_hr", 48.0)),

        "steel_carbon_factor_a1_a3_kgco2e_per_kg": float(_get_attr(baseline, "steel_carbon_factor_a1_a3_kgco2e_per_kg", 2.10)),
        "a5_installation_carbon_kgco2e_per_m2": float(_get_attr(baseline, "a5_installation_carbon_kgco2e_per_m2", 3.20)),

        "overhead_fraction_ref": float(_get_attr(baseline, "overhead_fraction_ref", 0.10)),
        "steel_waste_rate_ref": float(_get_attr(baseline, "steel_waste_rate_ref", 0.05)),
        # Central interaction coefficients used in deterministic optimisation
        "k_prefab_time_saving": float(_get_attr(baseline, "k_prefab_time_saving", 0.620)),
        "k_prefab_waste_reduction": float(_get_attr(baseline, "k_prefab_waste_reduction", 0.150)),
        "k_prefab_a5_carbon_saving": float(_get_attr(baseline, "k_prefab_a5_carbon_saving", 0.450)),

        "k_logistics_cost": float(_get_attr(baseline, "k_logistics_cost", 0.110)),
        "k_logistics_time": float(_get_attr(baseline, "k_logistics_time", 0.150)),
        "k_logistics_carbon": float(_get_attr(baseline, "k_logistics_carbon", 0.040)),

        "k_opening_cost": float(_get_attr(baseline, "k_opening_cost", 0.160)),
        "k_opening_carbon": float(_get_attr(baseline, "k_opening_carbon", 0.090)),
        "k_opening_time": float(_get_attr(baseline, "k_opening_time", 0.160)),
        "lambda_opening_prefab_reduction": float(_get_attr(baseline, "lambda_opening_prefab_reduction", 0.450)),

        "waikato_weather_downtime_fraction": float(_get_attr(baseline, "waikato_weather_downtime_fraction", 0.10)),
        "weather_shielding_factor": float(_get_attr(baseline, "weather_shielding_factor", 0.60)),
    }
    pareto_export = pareto_df.copy()

    # Prefer useful columns if available
    preferred_pareto_cols = [
        "Solution_ID",
        "R_p_panel_rationalisation",
        "R_t_truss_rationalisation",
        "panel_scheme",
        "truss_scheme",
        "opening_complexity_index",
        "P_f_prefab_level",
        "Cost_NZD",
        "Carbon_kgCO2e",
        "Time_hours",
        "Cost_normalized",
        "Carbon_normalized",
        "Time_normalized",
        "Cost_vs_S2_%",
        "Carbon_vs_S2_%",
        "Time_vs_S2_%",
        "NZ_feasible",
    ]

    existing_pareto_cols = [
        c for c in preferred_pareto_cols
        if c in pareto_export.columns
    ]

    if existing_pareto_cols:
        pareto_export = pareto_export[existing_pareto_cols].copy()

    representative_export = representative_df.copy()

    preferred_rep_cols = [
        "Case",
        "Label",
        "Solution_ID",
        "R_p_panel_rationalisation",
        "R_t_truss_rationalisation",
        "panel_scheme",
        "truss_scheme",
        "opening_complexity_index",
        "P_f_prefab_level",
        "Cost_NZD",
        "Carbon_kgCO2e",
        "Time_hours",
        "Cost_normalized",
        "Carbon_normalized",
        "Time_normalized",
        "Cost_vs_S2_%",
        "Carbon_vs_S2_%",
        "Time_vs_S2_%",
    ]

    existing_rep_cols = [
        c for c in preferred_rep_cols
        if c in representative_export.columns
    ]

    if existing_rep_cols:
        representative_export = representative_export[existing_rep_cols].copy()

    dashboard_data = {
        "schema_version": "1.1-validated",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_timestamp": ts,

        "project": {
            "name": "CFS residential framing optimization",
            "case_study": "Waikato S2 medium-prefabrication reference",
            "tool_role": "Recalibratable decision-support dashboard",
            "optimizer_role": "Python NSGA-II optimization engine",
        },

        "scope": {
            "cost": "CFS framing process cost only",
            "carbon": "CFS framing A1-A5 embodied carbon only",
            "time": "Construction process-hours only",
            "exclusions": [
                "whole-house construction cost",
                "whole-building embodied carbon",
                "operational carbon",
                "calendar project duration",
                "AS/NZS 4600 member-capacity checks",
                "connection design",
                "bracing verification",
                "quantity-surveyor pricing",
                "building consent documentation",
            ],
        },

        "baseline": {
            "cost_nzd": float(_get_attr(baseline, "C_S2_reference", np.nan)),
            "carbon_kgco2e": float(_get_attr(baseline, "CO2_S2_reference", np.nan)),
            "time_hours": float(_get_attr(baseline, "Time_S2_reference", np.nan)),
            "data_status": str(_get_attr(
                baseline,
                "S2_reference_data_status",
                "model-calculated CFS framing reference, not measured commercial data"
            )),
            "cost_scope": str(_get_attr(baseline, "S2_cost_scope", "")),
            "carbon_scope": str(_get_attr(baseline, "S2_carbon_scope", "")),
            "time_scope": str(_get_attr(baseline, "S2_time_scope", "")),
            "prefab_level": 0.72,
            "opening_complexity_index": float(_get_attr(baseline, "opening_complexity_ref", 1.00)),
            "footprint_m2": float(_get_attr(baseline, "footprint_area_m2", np.nan)),
            "envelope_m2": float(_get_attr(baseline, "envelope_area_m2", np.nan)),
            "steel_weight_kg": float(_get_attr(baseline, "W_base", np.nan)),
            "num_wall_panels": int(_get_attr(baseline, "num_wall_panels", 0)),
            "num_roof_trusses": int(_get_attr(baseline, "num_roof_trusses", 0)),
            "section": {
                "source": str(_get_attr(baseline, "cfs_section_source", "F325iT - Imported Section")),
                "designation": str(_get_attr(baseline, "cfs_section_designation", "89 S 41")),
                "thickness_mm": float(_get_attr(baseline, "assumed_steel_thickness_mm", 0.75)),
                "grade": str(_get_attr(baseline, "steel_grade", "G500")),
                "coating": str(_get_attr(baseline, "coating_class", "Z275")),
            }
        },

        "design_variables": {
            "R_p_panel_rationalisation": {
                "symbol": "R_p",
                "label": "Panel rationalisation intensity",
                "type": "continuous",
                "range": list(_get_attr(baseline, "R_p_range", (0.0, 1.0))),
                "s2_value": float(baseline.x_ref[0]),
                "description": "Optimiser variable. Higher value means stronger panel standardisation/rationalisation."
            },
            "R_t_truss_rationalisation": {
                "symbol": "R_t",
                "label": "Truss rationalisation intensity",
                "type": "continuous",
                "range": list(_get_attr(baseline, "R_t_range", (0.0, 1.0))),
                "s2_value": float(baseline.x_ref[1]),
                "description": "Optimiser variable. Higher value means stronger truss span-family rationalisation."
            },
            "opening_complexity_index": {
                "symbol": "OC",
                "label": "Opening/detailing complexity",
                "type": "continuous",
                "range": list(_get_attr(baseline, "opening_complexity_range", (0.80, 1.30))),
                "s2_value": float(baseline.x_ref[2]),
                "description": "Design-for-manufacture/detailing complexity variable."
            },
            "P_f_prefab_level": {
                "symbol": "P_f",
                "label": "Prefabrication level",
                "type": "continuous",
                "range": list(_get_attr(baseline, "P_f_range", (0.50, 0.90))),
                "s2_value": float(baseline.x_ref[3]),
                "description": "Fractional prefabrication level."
            }
        },
        "reporting_categories": {
            "panel_scheme": {
                "label": "Panel scheme reporting category",
                "values": list(_get_attr(baseline, "panel_scheme_values", [0, 1, 2, 3])),
                "description": "Mapped from R_p for reporting and plotting only."
            },
            "truss_scheme": {
                "label": "Truss scheme reporting category",
                "values": list(_get_attr(baseline, "truss_scheme_values", [0, 1, 2])),
                "description": "Mapped from R_t for reporting and plotting only."
            }
        },

        "recalibration_defaults": calibration_export,

        "calibration": calibration_export,

        "validation_ranges": {
            "raw_steel_material_nzd_per_kg": list(baseline.raw_steel_material_nzd_per_kg_range),
            "factory_processing_nzd_per_kg": list(baseline.factory_processing_nzd_per_kg_range),
            "site_labor_rate_nzd_per_hr": list(baseline.site_labor_rate_nzd_per_hr_range),
            "transport_cost_nzd_per_tonne_km": list(baseline.transport_cost_nzd_per_tonne_km_range),

            "steel_carbon_factor_a1_a3_kgco2e_per_kg": list(baseline.steel_carbon_factor_a1_a3_range),
            "a5_installation_carbon_kgco2e_per_m2": list(baseline.a5_installation_carbon_range),
            "freight_kgco2e_per_tonne_km": list(baseline.freight_kgco2e_per_tonne_km_range),

            "k_prefab_time_saving": list(baseline.k_prefab_time_saving_range),
            "k_prefab_waste_reduction": list(baseline.k_prefab_waste_reduction_range),
            "k_prefab_a5_carbon_saving": list(baseline.k_prefab_a5_carbon_saving_range),

            "k_logistics_cost": list(baseline.k_logistics_cost_range),
            "k_logistics_time": list(baseline.k_logistics_time_range),
            "k_logistics_carbon": list(baseline.k_logistics_carbon_range),

            "k_opening_cost": list(baseline.k_opening_cost_range),
            "k_opening_carbon": list(baseline.k_opening_carbon_range),
            "k_opening_time": list(baseline.k_opening_time_range),
            "lambda_opening_prefab_reduction": list(baseline.lambda_opening_prefab_reduction_range),

            "waikato_weather_downtime_fraction": list(baseline.waikato_weather_downtime_fraction_range),
            "weather_shielding_factor": list(baseline.weather_shielding_factor_range)
        },

        "key_findings": key_findings,

        "pareto": _json_safe_records(pareto_export, max_rows=2500),
        "representatives": _json_safe_records(representative_export),
        "sensitivity": _json_safe_records(sensitivity_df),
        "monte_carlo": _json_safe_records(monte_carlo_df),
        "topsis": _json_safe_records(topsis_df),
    }

    json_path = output_dir / "dashboard_config.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    # Also save a timestamped copy for audit trail
    json_path_ts = output_dir / f"dashboard_config_{ts}.json"

    with open(json_path_ts, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    print("\n[DASHBOARD EXPORT]")
    print(f"[SAVED] Live dashboard config: {json_path}")
    print(f"[SAVED] Timestamped dashboard config: {json_path_ts}")
    print("\n[DASHBOARD KEY FINDINGS]")
    for k, v in key_findings.items():
        print(f"  {k}: {v}")
    return dashboard_data

def build_validation_status_table(baseline: CFSProjectBaseline,
                                  ts: str) -> pd.DataFrame:
    """
    Records scope and validation-status notes for the reported model run.
    """

    rows = [
        {
            "Item": "S2 reference values",
            "Status": "Model-calculated reference",
            "Required_wording": "Model-calculated reference; not a measured commercial benchmark.",
            "Where_to_use": "All thesis tables, figure captions, dashboard baseline cards.",
        },
        {
            "Item": "k-values",
            "Status": "Calibrated scenario parameters",
            "Required_wording": "Interaction coefficients are calibrated scenario parameters tested through uncertainty analysis, not directly measured constants.",
            "Where_to_use": "Chapter 3 methodology, calibration audit table, Chapter 6 limitations.",
        },
        {
            "Item": "Opening complexity index",
            "Status": "Design-of-experiments variable",
            "Required_wording": "OC represents standardisation of lintels, jambs, service penetrations and trimming details; it does not change the number of architectural openings.",
            "Where_to_use": "Design-variable table, methodology, dashboard tooltip.",
        },
        {
            "Item": "Time objective",
            "Status": "Process-hours only",
            "Required_wording": "Construction time is reported as aggregated CFS framing process-hours, not calendar duration.",
            "Where_to_use": "Objective definition, all plots, dashboard labels.",
        },
        {
            "Item": "Carbon objective",
            "Status": "CFS framing A1-A5 only",
            "Required_wording": "Carbon includes CFS framing A1-A5 only and excludes operational and whole-building carbon.",
            "Where_to_use": "Objective definition, figures, dashboard labels.",
        },
        {
            "Item": "Building geometry",
            "Status": "Fixed",
            "Required_wording": "The optimisation does not vary footprint area, envelope area, wall-panel count, roof-truss count, or building layout.",
            "Where_to_use": "Section 4.2 and fixed boundary table.",
        },
        {
            "Item": "Structural member sizing",
            "Status": "Not optimised",
            "Required_wording": "The framework does not perform live AS/NZS 4600 member sizing, connection design, bracing design, or consent-level structural verification.",
            "Where_to_use": "Methodology boundary and limitations.",
        },
        {
            "Item": "Contractor tender benchmark",
            "Status": "Outside scope",
            "Required_wording": "The model reports CFS framing process cost, not a contractor tender price or whole-house construction cost.",
            "Where_to_use": "Cost objective and limitations.",
        },
    ]

    out = pd.DataFrame(rows)

    out_path = TABLE_DIR / f"validation_status_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[VALIDATION STATUS]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out

def plot_component_breakdown_comparison(cost_df, carbon_df, time_df, 
                                         baseline, path):
    """
    Stacked bar chart comparing component contributions across 
    representative solutions vs S2 reference.
    This is the most examiner-critical figure: it shows WHERE 
    cost/carbon/time differences come from.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    spec = [
        (
            cost_df,
            "Total_cost_NZD",
            baseline.C_S2_reference,
            "Cost components (NZD)",
            [
                "C_steel",
                "C_factory_processing",
                "C_setup",
                "C_site_labour",
                "C_transport",
                "C_delivery_logistics",
                "C_site_mobilisation",
                "C_opening_standardisation",
                "C_opening_complexity",
                "C_waste_handling",
                "C_overhead",
            ],
        ),
        (
            carbon_df,
            "Total_carbon_kgCO2e",
            baseline.CO2_S2_reference,
            "Carbon components (kgCO₂e)",
            [
                "CO2_A1_A3_steel",
                "CO2_waste",
                "CO2_A4_transport",
                "CO2_factory_energy",
                "CO2_A5_installation",
            ],
        ),
        (
            time_df,
            "Total_time_hours",
            baseline.Time_S2_reference,
            "Time components (hours)",
            [
                "T_factory_production",
                "T_factory_setup",
                "T_delivery_logistics",
                "T_site_mobilisation",
                "T_site_installation",
                "T_weather",
            ],
        ),
    ]

    for ax, (df, total_col, s2_ref, ylabel, comp_cols) in zip(axes, spec):
        available = [c for c in comp_cols if c in df.columns]
        solutions = df["Solution"].tolist()

        bottom = np.zeros(len(solutions))
        cmap   = plt.cm.get_cmap("tab10", len(available))

        for k, comp in enumerate(available):
            vals = df[comp].values.astype(float)
            ax.bar(solutions, vals, bottom=bottom,
                   label=comp.replace("C_", "").replace("CO2_", "").replace("T_", "").replace("_", " "),
                   color=cmap(k), alpha=0.85)
            bottom += vals

        ax.axhline(s2_ref, color="red", linestyle="--",
                   linewidth=1.8, label="S2 reference")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Component breakdown: representative Pareto solutions vs S2\n"
        "(red dashed = model-calculated S2 reference)",
        fontweight="bold", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")

def add_standard_design_aliases(df):
    """
    Adds short canonical design-variable aliases used by plotting/sensitivity
    functions without deleting the original thesis-readable column names.
    """

    df = df.copy()

    alias_map = {
        "R_p": [
            "R_p",
            "R_p_panel_rationalisation",
            "R_p_panel_rationalization",
            "panel_repetition_index",
            "panel_RI",
            "RI_panel",
            "panel_rationalisation_index",
            "panel_rationalization_index",
        ],
        "R_t": [
            "R_t",
            "R_t_truss_rationalisation",
            "R_t_truss_rationalization",
            "truss_repetition_index",
            "truss_RI",
            "RI_truss",
            "truss_rationalisation_index",
            "truss_rationalization_index",
        ],
        "OC": [
            "OC",
            "opening_complexity_index",
            "opening_complexity",
            "OCI",
        ],
        "P_f": [
            "P_f",
            "Pf",
            "P_f_prefab_level",
            "prefabrication_level",
            "prefab_level",
        ],
    }

    for standard_name, possible_names in alias_map.items():
        if standard_name not in df.columns:
            for col in possible_names:
                if col in df.columns:
                    df[standard_name] = df[col]
                    break

    return df

def plot_pareto_variable_spread(pareto_df: pd.DataFrame, baseline, path):
    """
    Thesis figure:
    Shows how much each design variable spreads across the Pareto set.
    This is useful when the Pareto front is narrow or when variables collapse to bounds.
    """

    df = pareto_df.copy()

    required = [
        "R_p_panel_rationalisation",
        "R_t_truss_rationalisation",
        "opening_complexity_index",
        "P_f_prefab_level",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARNING] Cannot plot variable spread. Missing columns: {missing}")
        return

    # Normalise all variables to [0, 1] so they are comparable on one axis.
    rp = df["R_p_panel_rationalisation"].astype(float)
    rt = df["R_t_truss_rationalisation"].astype(float)

    oc = (
        df["opening_complexity_index"].astype(float)
        - baseline.opening_complexity_range[0]
    ) / (
        baseline.opening_complexity_range[1]
        - baseline.opening_complexity_range[0]
    )

    pf = (
        df["P_f_prefab_level"].astype(float)
        - baseline.P_f_range[0]
    ) / (
        baseline.P_f_range[1]
        - baseline.P_f_range[0]
    )

    data = [rp, rt, oc, pf]
    labels = [
        r"$R_p$ panel rationalisation",
        r"$R_t$ truss rationalisation",
        r"$OC$ opening complexity",
        r"$P_f$ prefabrication level",
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        patch_artist=False,
    )

    rng = np.random.default_rng(42)

    for i, values in enumerate(data, start=1):
        jitter = rng.normal(0.0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            s=14,
            alpha=0.35,
        )

    ax.set_ylabel("Normalised variable position within allowed range")
    ax.set_title("Design-variable spread across Pareto-optimal solutions")
    ax.grid(axis="y", alpha=0.30)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {path}")

def select_core_representative_solutions_with_s2(
    pareto_df: pd.DataFrame,
    baseline
) -> pd.DataFrame:
    """
    Select thesis-ready representative solutions:
    cost-min, carbon-min, time-min, knee/balanced, and S2 reference.

    Important:
    This version uses the current revised-code S2 reference attributes:
        baseline.C_S2_reference
        baseline.CO2_S2_reference
        baseline.Time_S2_reference

    It does NOT use the older names:
        baseline.C_model_base
        baseline.CO2_model_base
        baseline.Time_model_base
    """

    df = pareto_df.copy()

    obj_cols = ["Cost_NZD", "Carbon_kgCO2e", "Time_hours"]

    for c in obj_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required objective column: {c}")

    # ------------------------------------------------------------------
    # 1. Objective-specific representative solutions
    # ------------------------------------------------------------------
    idx_cost = df["Cost_NZD"].idxmin()
    idx_carbon = df["Carbon_kgCO2e"].idxmin()
    idx_time = df["Time_hours"].idxmin()

    # ------------------------------------------------------------------
    # 2. Knee / balanced solution using distance to utopia point
    # ------------------------------------------------------------------
    F = df[obj_cols].astype(float).to_numpy()

    ideal = F.min(axis=0)
    nadir = F.max(axis=0)

    F_norm = (F - ideal) / (nadir - ideal + 1e-12)

    idx_knee_local = int(np.argmin(np.linalg.norm(F_norm, axis=1)))
    idx_knee = df.index[idx_knee_local]

    selected = [
        ("Cost-min", idx_cost),
        ("Carbon-min", idx_carbon),
        ("Time-min", idx_time),
        ("Knee/balanced", idx_knee),
    ]

    rows = []

    used_indices = set()

    for case_name, idx in selected:
        row = df.loc[idx].copy()
        row["Case"] = case_name
        row["Label"] = case_name

        if "Solution_ID" not in row.index or pd.isna(row.get("Solution_ID", np.nan)):
            row["Solution_ID"] = f"{case_name}"

        rows.append(row)
        used_indices.add(idx)

    # ------------------------------------------------------------------
    # 3. Build S2 reference row using current baseline attribute names
    # ------------------------------------------------------------------
    s2 = {}

    for col in df.columns:
        s2[col] = np.nan

    s2["Case"] = "S2 reference"
    s2["Label"] = "S2 reference"
    s2["Solution_ID"] = "S2"

    # Design vector at S2
    if hasattr(baseline, "x_ref"):
        s2["R_p_panel_rationalisation"] = baseline.x_ref[0]
        s2["R_t_truss_rationalisation"] = baseline.x_ref[1]
        s2["opening_complexity_index"] = baseline.x_ref[2]
        s2["P_f_prefab_level"] = baseline.x_ref[3]
    else:
        s2["R_p_panel_rationalisation"] = 0.0
        s2["R_t_truss_rationalisation"] = 0.0
        s2["opening_complexity_index"] = 1.0
        s2["P_f_prefab_level"] = 0.72

    # Reporting categories for S2
    if "panel_scheme" in df.columns:
        s2["panel_scheme"] = 0

    if "truss_scheme" in df.columns:
        s2["truss_scheme"] = 0

    # Current revised-code S2 baseline values
    if not hasattr(baseline, "C_S2_reference"):
        raise AttributeError("baseline.C_S2_reference is missing.")

    if not hasattr(baseline, "CO2_S2_reference"):
        raise AttributeError("baseline.CO2_S2_reference is missing.")

    if not hasattr(baseline, "Time_S2_reference"):
        raise AttributeError("baseline.Time_S2_reference is missing.")

    s2["Cost_NZD"] = float(baseline.C_S2_reference)
    s2["Carbon_kgCO2e"] = float(baseline.CO2_S2_reference)
    s2["Time_hours"] = float(baseline.Time_S2_reference)

    s2["Cost_normalized"] = 1.0
    s2["Carbon_normalized"] = 1.0
    s2["Time_normalized"] = 1.0

    s2["Cost_vs_S2_%"] = 0.0
    s2["Carbon_vs_S2_%"] = 0.0
    s2["Time_vs_S2_%"] = 0.0

    if "NZ_feasible" in df.columns:
        s2["NZ_feasible"] = True

    rows.append(pd.Series(s2))

    rep_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4. Clean column ordering
    # ------------------------------------------------------------------
    preferred_cols = [
        "Case",
        "Label",
        "Solution_ID",
        "R_p_panel_rationalisation",
        "R_t_truss_rationalisation",
        "panel_scheme",
        "truss_scheme",
        "opening_complexity_index",
        "P_f_prefab_level",
        "Cost_NZD",
        "Carbon_kgCO2e",
        "Time_hours",
        "Cost_normalized",
        "Carbon_normalized",
        "Time_normalized",
        "Cost_vs_S2_%",
        "Carbon_vs_S2_%",
        "Time_vs_S2_%",
        "NZ_feasible",
    ]

    existing_cols = [c for c in preferred_cols if c in rep_df.columns]
    remaining_cols = [c for c in rep_df.columns if c not in existing_cols]

    rep_df = rep_df[existing_cols + remaining_cols]

    return rep_df

def build_run_manifest_table(ts: str) -> pd.DataFrame:
    """
    Records the reported model run settings for Chapter 4 reproducibility.
    """

    rows = [
        {"Field": "Run_timestamp", "Value": ts},
        {"Field": "Run_mode", "Value": RUN_MODE},
        {"Field": "Initialisation_method", "Value": INIT_METHOD},
        {"Field": "Main_seed", "Value": MAIN_SEED},
        {"Field": "Population_size", "Value": POP_SIZE},
        {"Field": "Generations", "Value": GENERATIONS},
        {"Field": "Hypervolume_samples", "Value": HV_SAMPLES},
        {"Field": "Hypervolume_frequency", "Value": HV_EVERY},
        {"Field": "Grid_reference_step", "Value": GRID_PF_STEP},
        {"Field": "Monte_Carlo_enabled", "Value": RUN_MONTE_CARLO},
        {"Field": "Monte_Carlo_runs", "Value": MC_RUNS},
        {"Field": "Multi_seed_enabled", "Value": RUN_MULTI_SEED},
        {"Field": "Multi_seed_set", "Value": str(MULTI_SEED_SEEDS)},
        {"Field": "LHS_global_sensitivity_enabled", "Value": RUN_LHS_GLOBAL_SENSITIVITY},
        {"Field": "LHS_sensitivity_samples", "Value": LHS_DOE_SAMPLES},
        {"Field": "Tables_output_folder", "Value": str(TABLE_DIR)},
        {"Field": "Figures_output_folder", "Value": str(FIG_DIR)},
        {"Field": "Dashboard_data_folder", "Value": str(DASHBOARD_DATA_DIR)},
    ]

    out = pd.DataFrame(rows)
    out_path = TABLE_DIR / f"run_manifest_{ts}.csv"
    out.to_csv(out_path, index=False)

    print("\n[RUN MANIFEST]")
    print(out.to_string(index=False))
    print(f"[SAVED] {out_path}")

    return out
def export_dashboard_latest_data_bundle(ts: str, **tables):
    """
    Exports stable 'latest' dashboard files.

    Reason:
        A standalone HTML dashboard cannot scan results_fixed/tables and choose
        the newest timestamped CSV. Therefore the Python model must copy the
        latest run outputs into dashboard/data using fixed filenames.

    Output:
        dashboard/data/latest_manifest.json
        dashboard/data/pareto_front_latest.csv
        dashboard/data/representative_solutions_latest.csv
        dashboard/data/monte_carlo_robustness_latest.csv
        ...
    """

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "latest-dashboard-bundle-v1",
        "run_timestamp": str(ts),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_folder": str(DASHBOARD_DATA_DIR),
        "files": {}
    }

    for name, df in tables.items():
        if df is None:
            continue

        if not isinstance(df, pd.DataFrame):
            continue

        if df.empty:
            continue

        safe_name = (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        out_name = f"{safe_name}_latest.csv"
        out_path = DASHBOARD_DATA_DIR / out_name

        df.to_csv(out_path, index=False)

        manifest["files"][safe_name] = {
            "filename": out_name,
            "rows": int(len(df)),
            "columns": list(df.columns)
        }

        print(f"[DASHBOARD LATEST] {safe_name}: {out_path}")

    manifest_path = DASHBOARD_DATA_DIR / "latest_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[DASHBOARD LATEST] Manifest: {manifest_path}")

    return manifest

def main():
    random.seed(42)
    np.random.seed(42)
    #------------------------------------------------------------------
    # 1. Run optimisation and obtain reported-run objects
    #------------------------------------------------------------------
    final_archive, logbook, tb, baseline, obj, hv_df, ts, eval_audit = run_optimization(
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        seed=MAIN_SEED
    )
    #------------------------------------------------------------------
    # 2. Reported-run manifest
    #------------------------------------------------------------------
    run_manifest_df = build_run_manifest_table(ts)
    #------------------------------------------------------------------
    # 3. S2 reference values
    #------------------------------------------------------------------
    s2_reference_table_df = build_s2_reference_table(
        baseline=baseline,
        obj_func=obj,
        ts=ts
    )

    s2_reference = {
        "cost": obj.calculate_cost_abs(baseline.x_ref),
        "carbon": obj.calculate_carbon_abs(baseline.x_ref),
        "time": obj.calculate_time_abs(baseline.x_ref),
    }

    current_params = build_current_parameter_dictionary(baseline)

    calibration_audit_df = build_calibration_audit_table(
        params=current_params,
        s2_reference=s2_reference,
        output_path=TABLE_DIR / f"calibration_audit_{ts}.csv"
    )

    print(f"[SAVED] Calibration audit: {TABLE_DIR / f'calibration_audit_{ts}.csv'}")

    scenario_df = evaluate_prefab_scenarios(obj, baseline, ts)

    design_var_df = build_design_variable_table(baseline)
    design_var_path = TABLE_DIR / f"design_variable_table_{ts}.csv"
    design_var_df.to_csv(design_var_path, index=False)
    print(f"[SAVED] Design variable table: {design_var_path}")

    boundary_condition_df = build_fixed_boundary_condition_table(baseline, ts)

    validation_status_df = build_validation_status_table(baseline, ts)

    active_input_inventory_df = build_active_input_inventory_table(baseline, ts)


    #------------------------------------------------------------------
    # 4. Extract final nondominated front from external archive
    # ------------------------------------------------------------------
    front = tools.sortNondominated(
        final_archive,
        len(final_archive),
        first_front_only=True
    )[0]

    pdf = analyze_pareto_front(front, obj, baseline)

   
    pdf = get_pareto_front_df(
        pdf,
        obj_cols=("Cost_normalized", "Carbon_normalized", "Time_normalized")
    )

    pdf = clean_finite_objective_rows(
        pdf,
        name="NSGA_Pareto_before_IGD",
        ts=ts
    )

    diagnose_objective_dataframe(
        pdf,
        "NSGA_Pareto_after_cleaning",
        ts=ts
    )

    print("\n[OPTIMIZER VARIABLE DISTRIBUTION ON FINAL NONDOMINATED PARETO FRONT]")

    for col in ["panel_scheme", "truss_scheme"]:
        print(f"  {col}: {dict(pdf[col].value_counts().sort_index())}")

    print(
        "  opening_complexity_index: "
        f"min={pdf['opening_complexity_index'].min():.3f}, "
        f"max={pdf['opening_complexity_index'].max():.3f}, "
        f"mean={pdf['opening_complexity_index'].mean():.3f}"
    )

    print(
        "  P_f_prefab_level: "
        f"min={pdf['P_f_prefab_level'].min():.3f}, "
        f"max={pdf['P_f_prefab_level'].max():.3f}, "
        f"mean={pdf['P_f_prefab_level'].mean():.3f}"
    )

    print(f"\n[RESULT] Final nondominated Pareto front: {len(pdf)} solutions")
    print(f"  Cost:   {pdf['Cost_NZD'].min():.0f}–{pdf['Cost_NZD'].max():.0f} NZD")
    print(f"  Carbon: {pdf['Carbon_kgCO2e'].min():.0f}–{pdf['Carbon_kgCO2e'].max():.0f} kgCO2e")
    print(f"  Time:   {pdf['Time_hours'].min():.1f}–{pdf['Time_hours'].max():.1f} hrs")

    print("\n[PARETO STRUCTURE — DISCRETE VARIABLE DISTRIBUTION]")
    print(
        pdf.groupby(["panel_scheme", "truss_scheme"])[
            ["Cost_NZD", "Carbon_kgCO2e", "Time_hours"]
        ].agg(["min", "max", "count"])
    )

    # ------------------------------------------------------------------
    # 5. Grid validation and IGD
    # ------------------------------------------------------------------
    gdf, gpdf = run_grid_validation(
        obj,
        baseline,
        pf_step=GRID_PF_STEP
    )

    gdf = clean_finite_objective_rows(
        gdf,
        name="Grid_All_before_IGD",
        ts=ts
    )

    gpdf = clean_finite_objective_rows(
        gpdf,
        name="Grid_Pareto_before_IGD",
        ts=ts
    )

    nsga_pts = pdf[
        ["Cost_normalized", "Carbon_normalized", "Time_normalized"]
    ].values

    grid_pts = gpdf[
        ["Cost_normalized", "Carbon_normalized", "Time_normalized"]
    ].values

    igd_grid = compute_igd(nsga_pts, grid_pts)
    ablation_summary_df, ablation_all_df = run_ablation_grid_suite(
        obj,
        baseline,
        ts
    )
    if np.isfinite(igd_grid):
        print(f"\n[METRIC] IGD_vs_grid_reference = {igd_grid:.5f}")
        print("[NOTE] Grid reference front is an approximated reference, not the analytical true Pareto front.")
    else:
        print("\n[METRIC] IGD_vs_grid_reference = NaN")
        print("[DEBUG] Inspect finite-check CSVs for NSGA/Grid objective rows.")

    # ------------------------------------------------------------------
    # 6. Pareto structure, degeneracy, variable dominance
    # ------------------------------------------------------------------
    pdf = kmeans_archetypes(pdf, n_clusters=4)

    degeneracy_df = build_pareto_degeneracy_report(pdf, ts)

    variable_dominance_df = build_variable_dominance_finding(
        pdf,
        baseline,
        ts
    )

    if variable_dominance_df is not None and not variable_dominance_df.empty:
        collapsed_vars = variable_dominance_df[
            variable_dominance_df["Status"].astype(str).str.contains("Pareto-collapsed", na=False)
        ].copy()

        collapsed_vars.to_csv(
            TABLE_DIR / f"collapsed_variables_{ts}.csv",
            index=False
        )

        print(f"[SAVED] Collapsed variable diagnostic: {TABLE_DIR / f'collapsed_variables_{ts}.csv'}")
    else:
        collapsed_vars = pd.DataFrame()
        print("[WARNING] variable_dominance_df is empty. Collapsed-variable diagnostic skipped.")

    prefab_boundary_df = build_prefab_boundary_diagnostic(
        pdf,
        baseline,
        ts
    )

    if not collapsed_vars.empty:
        print("\n[WARNING] Some variables collapsed on the Pareto front:")
        for v in collapsed_vars["Variable"].tolist():
            print(f"  - {v}")

        print(
            "\n[INTERPRETATION] This is not automatically a coding error, but it must be "
            "reported as a model finding. If a variable should be active physically, "
            "check whether its benefit and penalty terms are balanced."
        )

    # ------------------------------------------------------------------
    # 7. Representative solutions and audits
    # ------------------------------------------------------------------
    rep_df = select_core_representative_solutions_with_s2(
        pareto_df=pdf,
        baseline=baseline
    )

    # Add carbon naming aliases BEFORE saving
    rep_df = add_carbon_scope_aliases(rep_df)

    rep_df.to_csv(
        TABLE_DIR / f"representative_solutions_{ts}.csv",
        index=False
    )

    print(f"[SAVED] {TABLE_DIR / f'representative_solutions_{ts}.csv'}")

    print(f"[SAVED] Representative solutions: {TABLE_DIR / f'representative_solutions_{ts}.csv'}")

    # Monte Carlo should only test optimised Pareto representative solutions.
    # The S2 row is kept in the CSV for comparison/reporting, but excluded from MC.
    rep_df_for_mc = rep_df[
        rep_df["Case"].astype(str).str.lower() != "s2 reference"
    ].copy()

    if rep_df_for_mc.empty:
        raise ValueError(
            "rep_df_for_mc is empty. Check the 'Case' column in representative_solutions."
        )

    solution_count_audit_df = build_solution_count_audit(
        eval_audit=eval_audit,
        final_archive=final_archive,
        pdf=pdf,
        rep_df=rep_df,
        ts=ts
    )

    improvement_df = compute_improvements_vs_s2(
        rep_df,
        ts
    )

    honest_improvement_df = compute_honest_improvements(
        rep_df,
        baseline,
        ts
    )

    topsis_df = build_topsis_ranking(
        pdf,
        ts
    )

    nz_constraint_df = build_nz_constraint_summary(
        pdf,
        ts
    )

    branch_df = build_pareto_branch_summary(
        pdf,
        ts
    )

    cost_comp_df, carbon_comp_df, time_comp_df = build_component_breakdown_tables(
        rep_df,
        obj,
        baseline,
        ts
    )

    # ------------------------------------------------------------------
    # 8. Local OAT sensitivity
    # ------------------------------------------------------------------
    sens_df = run_oat_sensitivity(
        obj,
        baseline,
        ts
    )

    elasticity_df = build_sensitivity_elasticity_table(
        sens_df,
        baseline,
        ts
    )

    # ------------------------------------------------------------------
    # 9. LHS global sensitivity analysis
    # ------------------------------------------------------------------
    if RUN_LHS_GLOBAL_SENSITIVITY:
        lhs_doe_df = evaluate_lhs_design_space(
            obj_func=obj,
            baseline=baseline,
            ts=ts,
            n_samples=LHS_DOE_SAMPLES,
            seed=42
        )

        lhs_spearman_df, lhs_prcc_df, lhs_group_effect_df = run_lhs_global_sensitivity(
            lhs_doe_df,
            ts
        )
    else:
        print("\n[INFO] LHS global sensitivity skipped.")
        lhs_doe_df = pd.DataFrame()
        lhs_spearman_df = pd.DataFrame()
        lhs_prcc_df = pd.DataFrame()
        lhs_group_effect_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # 10. Monte Carlo robustness
    # ------------------------------------------------------------------
    if RUN_MONTE_CARLO:
        print(f"\n[INFO] Monte Carlo uncertainty analysis (n={MC_RUNS}) ...")

        rep_df_for_mc = rep_df[
            rep_df["Case"].astype(str).str.lower() != "s2 reference"
        ].copy()

        if rep_df_for_mc.empty:
            raise ValueError(
                "rep_df_for_mc is empty. Check the 'Case' column in representative_solutions."
            )

        mc = run_monte_carlo(
            rep_df_for_mc,
            baseline,
            n_runs=MC_RUNS,
            seed=42
        )

        mc_df = build_mc_table(
            mc,
            baseline
        )

        mc_df.to_csv(
            TABLE_DIR / f"monte_carlo_robustness_{ts}.csv",
            index=False
        )

        print(f"[SAVED] {TABLE_DIR / f'monte_carlo_robustness_{ts}.csv'}")

        print("\n[ROBUSTNESS RANKING]")
        print(
            mc_df[
                [
                    "Robust_rank",
                    "Solution",
                    "Robust_score",
                    "Cost_CV_%",
                    "Carbon_CV_%",
                    "Time_CV_%"
                ]
            ].to_string(index=False)
        )

    else:
        print("\n[INFO] Monte Carlo skipped.")
        mc = {}
        mc_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # 11. Save main result tables
    # ------------------------------------------------------------------
    pdf_out = add_carbon_scope_aliases(pdf)
    gdf_out = add_carbon_scope_aliases(gdf)
    gpdf_out = add_carbon_scope_aliases(gpdf)

    # ------------------------------------------------------------------
    # OC-fixed diagnostic:
    # Checks whether Pareto dominance is mainly caused by low opening complexity.
    # ------------------------------------------------------------------
    oc_fixed_df = compare_at_fixed_oc(
        obj_func=obj,
        baseline=baseline,
        pareto_df=pdf,
        fixed_oc=1.0
    )

    oc_fixed_path = TABLE_DIR / f"oc_fixed_comparison_{ts}.csv"
    oc_fixed_df.to_csv(oc_fixed_path, index=False)

    print("\n[OC-FIXED COMPARISON]")
    print(f"[SAVED] {oc_fixed_path}")
    print(oc_fixed_df["Dominates_S2_fixed"].value_counts().to_string())

    pdf_out.to_csv(
        TABLE_DIR / f"pareto_front_{ts}.csv",
        index=False
    )

    gdf_out.to_csv(
        TABLE_DIR / f"grid_all_{ts}.csv",
        index=False
    )

    gpdf_out.to_csv(
        TABLE_DIR / f"grid_pareto_{ts}.csv",
        index=False
    )

    hv_df.to_csv(
        TABLE_DIR / f"hypervolume_{ts}.csv",
        index=False
    )
    hv_progress_df = build_hypervolume_progress_table(
        hv_df,
        ts
    )
    pd.DataFrame(logbook).to_csv(
        TABLE_DIR / f"logbook_{ts}.csv",
        index=False
    )

    val_df = pd.DataFrame([
        {
            "Method": "NSGA-II",
            "Solutions": len(pdf),
            "Min_cost": pdf["Cost_normalized"].min(),
            "Max_cost": pdf["Cost_normalized"].max(),
            "Min_carbon": pdf["Carbon_normalized"].min(),
            "Max_carbon": pdf["Carbon_normalized"].max(),
            "Min_time": pdf["Time_normalized"].min(),
            "Max_time": pdf["Time_normalized"].max(),
            "IGD_vs_grid_reference": igd_grid
        },
        {
            "Method": "Grid",
            "Solutions": len(gpdf),
            "Min_cost": gpdf["Cost_normalized"].min(),
            "Max_cost": gpdf["Cost_normalized"].max(),
            "Min_carbon": gpdf["Carbon_normalized"].min(),
            "Max_carbon": gpdf["Carbon_normalized"].max(),
            "Min_time": gpdf["Time_normalized"].min(),
            "Max_time": gpdf["Time_normalized"].max(),
            "IGD_vs_grid_reference": 0.0
        },
    ])

    val_df.to_csv(
        TABLE_DIR / f"nsga_vs_grid_{ts}.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # 12. Dashboard export
    # ------------------------------------------------------------------
   
    export_dashboard_config(
        baseline=baseline,
        pareto_df=pdf_out,
        representative_df=rep_df,
        ts=ts,
        sensitivity_df=lhs_prcc_df if RUN_LHS_GLOBAL_SENSITIVITY else sens_df,
        monte_carlo_df=mc_df,
        topsis_df=topsis_df
    )
    # ------------------------------------------------------------------
    # 12B. Dashboard latest-data bundle
    # ------------------------------------------------------------------
    latest_dashboard_manifest = export_dashboard_latest_data_bundle(
        ts=ts,

        pareto_front=pdf_out,
        representative_solutions=rep_df,
        pareto_branch_summary=branch_df,
        honest_improvements_vs_s2=honest_improvement_df,

        component_breakdown_cost=cost_comp_df,
        component_breakdown_carbon=carbon_comp_df,
        component_breakdown_time=time_comp_df,

        monte_carlo_robustness=mc_df,

        lhs_prcc_sensitivity=lhs_prcc_df if "lhs_prcc_df" in locals() else pd.DataFrame(),
        lhs_spearman_sensitivity=lhs_spearman_df if "lhs_spearman_df" in locals() else pd.DataFrame(),
        sensitivity_oat=sens_df if "sens_df" in locals() else pd.DataFrame(),

        topsis_stakeholder_ranking=topsis_df if "topsis_df" in locals() else pd.DataFrame(),

        variable_dominance_finding=variable_dominance_df if "variable_dominance_df" in locals() else pd.DataFrame(),
        pareto_degeneracy_report=degeneracy_df if "degeneracy_df" in locals() else pd.DataFrame(),

        hypervolume_progress=hv_progress_df if "hv_progress_df" in locals() else pd.DataFrame(),
        multi_seed_summary=pd.DataFrame(),

        nsga_vs_grid=val_df if "val_df" in locals() else pd.DataFrame()
    )
    print("\n[CHECK] Dashboard latest files now in:")
    print(DASHBOARD_DATA_DIR)

    for p in sorted(DASHBOARD_DATA_DIR.glob("*latest*")):
        print("  -", p.name)
    # ------------------------------------------------------------------
    # 13. Generate figures
    # ------------------------------------------------------------------
    print("\n[INFO] Generating figures ...")

    plot_component_breakdown_comparison(
        cost_df=cost_comp_df,
        carbon_df=carbon_comp_df,
        time_df=time_comp_df,
        baseline=baseline,
        path=FIG_DIR / f"component_breakdown_representative_vs_s2_{ts}.png"
    )

    pdf_plot = deduplicate_pareto_for_plotting(pdf)

    plot_pareto_2d_by_panel_scheme(
        pdf_plot,
        FIG_DIR / f"pareto_2d_by_panel_scheme_{ts}.png"
    )

    plot_pareto_3d(
        pdf_plot,
        FIG_DIR / f"pareto_3d_{ts}.png"
    )

    plot_pareto_2d(
        pdf_plot,
        FIG_DIR / f"pareto_2d_{ts}.png"
    )

    plot_parallel_coordinates(
        pdf_plot,
        FIG_DIR / f"parallel_coordinates_{ts}.png"
    )

    plot_pareto_variable_spread(
        pdf,
        baseline,
        FIG_DIR / f"pareto_variable_spread_{ts}.png"
    )

    plot_opening_interaction(
        obj,
        baseline,
        FIG_DIR / f"opening_prefab_interaction_{ts}.png"
    )

    plot_cost_breakeven(
        obj,
        baseline,
        FIG_DIR / f"cost_breakeven_{ts}.png"
    )

    plot_nsga_vs_grid_overlay(
        nsga_df=pdf,
        grid_df=gpdf,
        path=FIG_DIR / f"nsga_vs_grid_overlay_cost_carbon_time_{ts}.png"
    )

    if RUN_MONTE_CARLO and len(mc) > 0:
        plot_monte_carlo_uncertainty(
            mc,
            baseline,
            FIG_DIR / f"monte_carlo_uncertainty_{ts}.png"
        )
   
    # ------------------------------------------------------------------
    # Standardise design-variable aliases for post-processing plots
    # ------------------------------------------------------------------
    gdf = add_standard_design_aliases(gdf)
    pdf = add_standard_design_aliases(pdf)
    gpdf = add_standard_design_aliases(gpdf)

    print("\n[GDF COLUMNS BEFORE DESIGN VARIABLE TORNADO]")
    print(list(gdf.columns))

    required_aliases = ["R_p", "R_t", "OC", "P_f"]

    missing_aliases = [c for c in required_aliases if c not in gdf.columns]

    if missing_aliases:
        raise KeyError(
            f"Missing required design-variable alias columns before tornado plot: {missing_aliases}. "
            f"Available columns are: {list(gdf.columns)}"
        )

    design_effect_df = plot_design_variable_tornado_from_grid(
        gdf,
        FIG_DIR / f"design_variable_tornado_{ts}.png"
    )

    design_effect_df.to_csv(
        TABLE_DIR / f"design_variable_effect_ranges_{ts}.csv",
        index=False
    )

   
    plot_repetition_index_heatmaps(
        gdf,
        FIG_DIR / f"Rp_Rt_objective_heatmaps_{ts}.png",
        aggregation="min"
    )

    plot_main_effects_from_grid(
        gdf,
        FIG_DIR / f"main_effects_design_variables_{ts}.png"
    )

   
    if RUN_MONTE_CARLO and len(mc) > 0:
        plot_monte_carlo_boxplots(
            mc,
            FIG_DIR / f"monte_carlo_boxplots_{ts}.png"
        )
    plt.figure(figsize=(8, 5))
    plt.plot(
        hv_df["generation"],
        hv_df["hypervolume"],
        marker="o",
        ms=3
    )
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"hypervolume_{ts}.png",
        dpi=300
    )
    plt.close()


    # ------------------------------------------------------------------
    # 14. Multi-seed convergence
    # ------------------------------------------------------------------
    if RUN_MULTI_SEED:
        print("\n[INFO] Multi-seed convergence and stability test.")

        seed_df, seed_summary, multiseed_reference_df, combined_seed_fronts_df = run_multi_seed_convergence(
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            seeds=MULTI_SEED_SEEDS
        )

        seed_runs_path = TABLE_DIR / f"multi_seed_runs_{ts}.csv"
        seed_summary_path = TABLE_DIR / f"multi_seed_summary_{ts}.csv"
        seed_reference_path = TABLE_DIR / f"multi_seed_reference_front_{ts}.csv"
        combined_seed_fronts_path = TABLE_DIR / f"multi_seed_combined_fronts_{ts}.csv"

        seed_df.to_csv(seed_runs_path, index=False)
        seed_summary.to_csv(seed_summary_path, index=False)
        multiseed_reference_df.to_csv(seed_reference_path, index=False)
        combined_seed_fronts_df.to_csv(combined_seed_fronts_path, index=False)

        print("\n[MULTI-SEED FILES SAVED]")
        print(f"[SAVED] {seed_runs_path}")
        print(f"[SAVED] {seed_summary_path}")
        print(f"[SAVED] {seed_reference_path}")
        print(f"[SAVED] {combined_seed_fronts_path}")

        print(f"\n[MULTI-SEED SUMMARY]\n{seed_summary.to_string(index=False)}")

    else:
        print("\n[INFO] Multi-seed skipped.")
    
    # ------------------------------------------------------------------
    # 15. Finish
    # ------------------------------------------------------------------
    print(f"\n[DONE] Outputs saved with timestamp {ts}")
    print(f"[TABLES] {TABLE_DIR}")
    print(f"[FIGURES] {FIG_DIR}")
    print(f"[DASHBOARD DATA] {DASHBOARD_DATA_DIR / 'dashboard_config.json'}")

    return pdf, rep_df, mc_df, sens_df, val_df, hv_df, ts, obj, baseline


if __name__ == "__main__":
    main()
