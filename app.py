"""
Horikx Analysis Application for Rubber Devulcanization & Polymer Degradation
===========================================================================
A publication-grade interactive Streamlit web application for Horikx plot
construction, automated degradation mechanism classification, and high-resolution
figure generation.

Based on Horikx's theoretical framework (1956):
- Crosslink Scission (Selective Devulcanization):
    1 - (v_f / v_i) = 1 - [ (1 - s_f^(1/2))^2 / (1 - s_0^(1/2))^2 ]
- Main-Chain Scission (Polymer Backbone Degradation):
    1 - (v_f / v_i) = 1 - [ (1 - s_f^(1/2)) / (1 - s_0^(1/2)) ]

Requirements:
    pip install streamlit numpy pandas plotly matplotlib openpyxl
"""

import io
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Horikx Plot Analyzer | Rubber Devulcanization",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean layout, metric cards, and high readability
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 12px 16px;
        border-radius: 8px;
    }
    .info-box {
        background-color: #f1f5f9;
        border-left: 4px solid #4f46e5;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. THEORETICAL HORIKX CALCULATION ENGINE
# ==============================================================================
def calculate_horikx_theoretical_curves(s0: float, num_points: int = 400):
    """
    Computes theoretical Horikx boundary curves for a given initial sol fraction (S0).
    
    Parameters:
        s0 (float): Initial sol fraction of the virgin rubber network (0 < s0 < 1).
        num_points (int): Number of numerical evaluation steps.
        
    Returns:
        dict: Arrays for crosslink scission, main-chain scission, and sol fractions.
    """
    s0_clamped = max(0.0001, min(0.9999, float(s0)))
    sqrt_s0 = math.sqrt(s0_clamped)
    denom_crosslink = (1.0 - sqrt_s0) ** 2
    denom_mainchain = 1.0 - sqrt_s0

    # Generate fine sol fraction space from S0 to 0.9999
    s_vals = np.linspace(s0_clamped, 0.9999, num_points)
    sqrt_s = np.sqrt(s_vals)

    # 1. Selective Crosslink Scission Curve
    # 1 - (v_f / v_i) = 1 - [ (1 - sqrt(s_f))^2 / (1 - sqrt(s_0))^2 ]
    crosslink_decrease_cl = 1.0 - ((1.0 - sqrt_s) ** 2) / denom_crosslink
    crosslink_decrease_cl = np.clip(crosslink_decrease_cl, 0.0, 1.0)

    # 2. Main-Chain Scission Curve
    # 1 - (v_f / v_i) = 1 - [ (1 - sqrt(s_f)) / (1 - sqrt(s_0)) ]
    crosslink_decrease_mc = 1.0 - (1.0 - sqrt_s) / denom_mainchain
    crosslink_decrease_mc = np.clip(crosslink_decrease_mc, 0.0, 1.0)

    return {
        "s_vals": s_vals,
        "x_crosslink": crosslink_decrease_cl,
        "x_mainchain": crosslink_decrease_mc,
        "s0": s0_clamped
    }


def get_theoretical_sol_fraction_cl(x_val: float, s0: float) -> float:
    """Calculates theoretical Sol Fraction on Crosslink Scission curve for a given X."""
    x_clamped = max(0.0, min(1.0, x_val))
    s0_clamped = max(0.0001, min(0.9999, s0))
    sqrt_s0 = math.sqrt(s0_clamped)
    # (1 - sqrt(s_f))^2 = (1 - x) * (1 - sqrt(s_0))^2
    term = math.sqrt(max(0.0, 1.0 - x_clamped)) * (1.0 - sqrt_s0)
    sqrt_sf = 1.0 - term
    return max(0.0, min(1.0, sqrt_sf ** 2))


def get_theoretical_sol_fraction_mc(x_val: float, s0: float) -> float:
    """Calculates theoretical Sol Fraction on Main-Chain Scission curve for a given X."""
    x_clamped = max(0.0, min(1.0, x_val))
    s0_clamped = max(0.0001, min(0.9999, s0))
    sqrt_s0 = math.sqrt(s0_clamped)
    # (1 - sqrt(s_f)) = (1 - x) * (1 - sqrt(s_0))
    sqrt_sf = 1.0 - (1.0 - x_clamped) * (1.0 - sqrt_s0)
    return max(0.0, min(1.0, sqrt_sf ** 2))


def get_theoretical_x_cl(sf: float, s0: float) -> float:
    """Calculates theoretical X (crosslink decrease) on Crosslink Scission curve for a given Sol (s_f)."""
    sf_clamped = max(0.0, min(0.9999, sf))
    s0_clamped = max(0.0001, min(0.9999, s0))
    if sf_clamped <= s0_clamped:
        return 0.0
    sqrt_sf = math.sqrt(sf_clamped)
    sqrt_s0 = math.sqrt(s0_clamped)
    denom = (1.0 - sqrt_s0) ** 2
    if denom == 0:
        return 0.0
    val = 1.0 - ((1.0 - sqrt_sf) ** 2) / denom
    return max(0.0, min(1.0, val))


def get_theoretical_x_mc(sf: float, s0: float) -> float:
    """Calculates theoretical X (crosslink decrease) on Main-Chain Scission curve for a given Sol (s_f)."""
    sf_clamped = max(0.0, min(0.9999, sf))
    s0_clamped = max(0.0001, min(0.9999, s0))
    if sf_clamped <= s0_clamped:
        return 0.0
    sqrt_sf = math.sqrt(sf_clamped)
    sqrt_s0 = math.sqrt(s0_clamped)
    denom = 1.0 - sqrt_s0
    if denom == 0:
        return 0.0
    val = 1.0 - (1.0 - sqrt_sf) / denom
    return max(0.0, min(1.0, val))


def evaluate_sample_mechanism(x_meas: float, s_meas: float, s0: float):
    """
    Evaluates polymer degradation mechanism by computing horizontal (Δx)
    distances to theoretical Horikx boundaries and scission selectivity.
    """
    x_val = max(0.0, min(1.0, float(x_meas)))
    s_val = max(0.0, min(1.0, float(s_meas)))
    s0_val = max(0.0001, min(0.9999, float(s0)))

    # Theoretical curve X positions at the measured sol fraction level
    x_cl_target = get_theoretical_x_cl(s_val, s0_val)
    x_mc_target = get_theoretical_x_mc(s_val, s0_val)

    # Horizontal offsets (Delta X)
    dx_cl = abs(x_val - x_cl_target)
    dx_mc = abs(x_val - x_mc_target)

    # Scission Ratio & Selectivity
    total_dist = dx_cl + dx_mc
    if total_dist < 1e-7:
        cl_ratio = 50.0
        mc_ratio = 50.0
    else:
        # Inverse distance weighting
        cl_ratio = (dx_mc / total_dist) * 100.0
        mc_ratio = (dx_cl / total_dist) * 100.0

    # Classification logic based on physical bounds
    if s_val < s0_val:
        classification = "Below Initial Baseline (S_f < S₀)"
        color = "#64748b"
        quality_rating = "Baseline Discrepancy"
    elif x_val > x_cl_target and (x_val - x_cl_target) > 0.05:
        classification = "Highly Selective Crosslink Scission"
        color = "#16a34a"
        quality_rating = "Superior Devulcanization"
    elif dx_cl <= 0.04:
        classification = "Dominant Crosslink Scission"
        color = "#10b981"
        quality_rating = "Ideal Devulcanization"
    elif dx_mc <= 0.04:
        classification = "Dominant Main-Chain Scission"
        color = "#ef4444"
        quality_rating = "Severe Degradation"
    elif x_val < x_mc_target and (x_mc_target - x_val) > 0.05:
        classification = "Extreme Chain Degradation"
        color = "#b91c1c"
        quality_rating = "Critical Backbone Breakdown"
    else:
        classification = "Mixed Scission Mechanism"
        color = "#f59e0b"
        quality_rating = "Moderate Selectivity"

    return {
        "x": x_val,
        "s": s_val,
        "s0": s0_val,
        "dx_cl": dx_cl,
        "dx_mc": dx_mc,
        "cl_ratio": cl_ratio,
        "mc_ratio": mc_ratio,
        "classification": classification,
        "color": color,
        "quality_rating": quality_rating
    }


# ==============================================================================
# 3. BUILT-IN REFERENCE DATASETS
# ==============================================================================
DEFAULT_DATASETS = {
    "Thermal Devulcanization (EPDM)": {
        "s0": 0.02,
        "description": "Devulcanization of sulfur-cured EPDM at varying autoclave temperatures (180°C - 260°C).",
        "data": pd.DataFrame({
            "Sample Name": ["EPDM-180C", "EPDM-200C", "EPDM-220C", "EPDM-240C", "EPDM-260C"],
            "Initial Sol (S₀)": [0.02, 0.02, 0.02, 0.02, 0.02],
            "1 - v/v0": [0.35, 0.58, 0.74, 0.88, 0.95],
            "Sol Fraction (s)": [0.04, 0.08, 0.14, 0.28, 0.52],
            "Condition": ["180°C / 10min", "200°C / 10min", "220°C / 10min", "240°C / 10min", "260°C / 10min"]
        })
    },
    "Chemical Mechano-Scission (Ground Tire Rubber)": {
        "s0": 0.035,
        "description": "Ambient twin-screw extrusion of truck tire GTR using diaryl disulfide chemical agent.",
        "data": pd.DataFrame({
            "Sample Name": ["GTR-Control", "GTR-MechOnly", "GTR-1.0phr-Agent", "GTR-2.5phr-Agent", "GTR-4.0phr-Agent"],
            "Initial Sol (S₀)": [0.035, 0.035, 0.035, 0.035, 0.035],
            "1 - v/v0": [0.12, 0.44, 0.68, 0.84, 0.92],
            "Sol Fraction (s)": [0.04, 0.16, 0.12, 0.18, 0.26],
            "Condition": ["Virgin GTR", "Twin-Screw Mechanical", "1.0 phr Disulfide", "2.5 phr Disulfide", "4.0 phr Disulfide"]
        })
    },
    "High-Shear Degradation (Natural Rubber)": {
        "s0": 0.015,
        "description": "Aggressive mechanical mastication of sulfur-vulcanized NR demonstrating main-chain scission.",
        "data": pd.DataFrame({
            "Sample Name": ["NR-Shear-1", "NR-Shear-2", "NR-Shear-3", "NR-Shear-4", "NR-Shear-5"],
            "Initial Sol (S₀)": [0.015, 0.015, 0.015, 0.015, 0.015],
            "1 - v/v0": [0.25, 0.48, 0.65, 0.78, 0.89],
            "Sol Fraction (s)": [0.18, 0.38, 0.54, 0.68, 0.82],
            "Condition": ["200 rpm / 2min", "400 rpm / 4min", "600 rpm / 6min", "800 rpm / 8min", "1000 rpm / 10min"]
        })
    }
}


# ==============================================================================
# 4. HEADER & SIDEBAR CONTROLS
# ==============================================================================
st.markdown('<div class="main-header">⚗️ Horikx Analysis & Polymer Devulcanization Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Distinguish selective sulfur crosslink cleavage from main-chain polymer backbone degradation.</div>',
    unsafe_allow_html=True
)

st.sidebar.header("🛠️ Data Input & Parameters")

input_mode = st.sidebar.radio(
    "Data Source Mode",
    ["📁 Upload CSV / Excel File", "🧪 Built-in Benchmark Datasets", "✏️ Interactive Manual Entry"],
    index=0
)

# Initialize Session State
if "active_df" not in st.session_state:
    st.session_state["active_df"] = DEFAULT_DATASETS["Thermal Devulcanization (EPDM)"]["data"].copy()

# ------------------------------------------------------------------------------
# 4A. UPLOAD HANDLING WITH DYNAMIC COLUMN PARSING & STRICT S0 ISOLATION
# ------------------------------------------------------------------------------
if input_mode == "📁 Upload CSV / Excel File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Experimental Data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Upload tabular data containing Sample Name, Initial Sol (S₀), 1 - ν/ν₀, and measured Sol Fraction."
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)

            all_cols = list(raw_df.columns)
            
            # Helper to check if a series is truly numeric
            def is_col_numeric(series):
                converted = pd.to_numeric(series.dropna(), errors='coerce')
                return converted.notna().sum() > 0 and (converted.notna().sum() / max(1, len(series.dropna()))) > 0.6

            numeric_cols = [c for c in all_cols if is_col_numeric(raw_df[c])]
            text_cols = [c for c in all_cols if c not in numeric_cols]

            if not numeric_cols:
                st.error("⚠️ No numeric columns found in the uploaded file. Please ensure crosslink decrease and sol fraction contain numeric values.")
            else:
                # Helper to detect if a column represents initial sol / s0 baseline
                def is_s0_col(col_name):
                    clean = str(col_name).lower().replace(" ", "").replace("_", "").replace("-", "").replace("/", "").replace("(", "").replace(")", "")
                    s0_markers = ["initial", "s0", "s_0", "si", "s_i", "virgin", "baseline", "control", "unvulcanized", "raw"]
                    return any(m in clean for m in s0_markers)

                # Prioritized keyword matching helper
                def find_prioritized_col(candidates, keywords):
                    for k in keywords:
                        clean_k = k.lower().replace(" ", "").replace("_", "").replace("-", "").replace("/", "").replace("(", "").replace(")", "")
                        for c in candidates:
                            clean_c = str(c).lower().replace(" ", "").replace("_", "").replace("-", "").replace("/", "").replace("(", "").replace(")", "")
                            if clean_k in clean_c:
                                return c
                    return None

                # 1. Identify S0 candidates
                s0_candidates = [c for c in numeric_cols if is_s0_col(c)]
                auto_s0_col = s0_candidates[0] if s0_candidates else None
                auto_s0_idx = numeric_cols.index(auto_s0_col) if auto_s0_col else -1

                # 2. Identify X-Axis (1 - v/v0): Crosslink Density Decrease (prefer non-S0)
                non_s0_numeric = [c for c in numeric_cols if not is_s0_col(c)]
                x_pool = non_s0_numeric if non_s0_numeric else numeric_cols
                x_keywords = [
                    "1vv0", "1v", "1nu", "1vvi", "crosslinkdensitydecrease", "crosslinkdecrease",
                    "decreaseincrosslink", "densitydecrease", "crosslinkloss", "loss", "vdv0",
                    "crosslinkdensity", "crosslink", "decrease"
                ]
                auto_x_col = find_prioritized_col(x_pool, x_keywords) or (x_pool[0] if x_pool else numeric_cols[0])
                auto_x_idx = numeric_cols.index(auto_x_col) if auto_x_col in numeric_cols else 0

                # 3. Identify Y-Axis (Sol Fraction S_f): MUST exclude X-axis and STRICTLY exclude S0
                y_candidates = [c for c in numeric_cols if c != auto_x_col and not is_s0_col(c) and c != auto_s0_col]
                measured_s_keywords = [
                    "finalsolfraction", "finalsol", "measuredsolfraction", "measuredsol",
                    "devulcanizedsol", "sokfraction", "measuredfraction", "measure",
                    "solfractions", "solfractionsf", "sol(s)", "sol(sf)", "solcontent",
                    "solublefraction", "solfraction", "solpercentage", "extractables",
                    "sf", "sol_f", "solf", "sols", "sol", "s"
                ]
                auto_y_col = find_prioritized_col(y_candidates, measured_s_keywords)
                if not auto_y_col and y_candidates:
                    auto_y_col = y_candidates[0]

                # Safeguard: never let Y-axis map to an S0 column
                if auto_y_col and is_s0_col(auto_y_col):
                    auto_y_col = None

                auto_s_idx = numeric_cols.index(auto_y_col) if auto_y_col in numeric_cols else (1 if len(numeric_cols) > 1 and numeric_cols[1] != auto_x_col else 0)

                # 4. Identify Sample Name column (text column)
                name_keywords = ["samplename", "sample", "name", "id", "condition", "treatment", "trial", "batch"]
                auto_name_col = find_prioritized_col(all_cols, name_keywords)
                if not auto_name_col and text_cols:
                    auto_name_col = text_cols[0]

                st.sidebar.markdown("### 🎛️ Dynamic Column Mapping")
                name_options = ["<Auto-generate ID>"] + all_cols
                name_idx = name_options.index(auto_name_col) if auto_name_col in name_options else 0
                sample_col_mapped = st.sidebar.selectbox("Sample Name / ID Column", name_options, index=name_idx)

                s0_options = ["<Global Fixed S₀ from Slider>"] + numeric_cols
                s0_idx = (numeric_cols.index(auto_s0_col) + 1) if auto_s0_col else 0
                s0_col_mapped = st.sidebar.selectbox("Initial Sol (S₀) Column", s0_options, index=s0_idx)

                x_col_mapped = st.sidebar.selectbox("Decrease in Crosslink Density (X-axis)", numeric_cols, index=auto_x_idx)
                s_col_mapped = st.sidebar.selectbox("Measured Sol Fraction (Y-axis)", numeric_cols, index=auto_s_idx)

                # Build sanitized dataframe
                parsed_df = pd.DataFrame()
                if sample_col_mapped == "<Auto-generate ID>":
                    parsed_df["Sample Name"] = [f"Sample-{i+1}" for i in range(len(raw_df))]
                else:
                    parsed_df["Sample Name"] = raw_df[sample_col_mapped].astype(str).fillna("Unnamed")

                parsed_df["1 - v/v0"] = pd.to_numeric(raw_df[x_col_mapped], errors='coerce').fillna(0.0)
                parsed_df["Sol Fraction (s)"] = pd.to_numeric(raw_df[s_col_mapped], errors='coerce').fillna(0.0)

                if s0_col_mapped != "<Global Fixed S₀ from Slider>":
                    parsed_df["Initial Sol (S₀)"] = pd.to_numeric(raw_df[s0_col_mapped], errors='coerce').fillna(0.02)
                else:
                    parsed_df["Initial Sol (S₀)"] = np.nan

                # Detect percentage vs fraction scale (if values > 1.0, convert to decimal fractions)
                if parsed_df["1 - v/v0"].max() > 1.0 or parsed_df["Sol Fraction (s)"].max() > 1.0:
                    st.sidebar.info("ℹ️ Detected percentage values (> 1.0). Automatically converted to [0.0 - 1.0] fractions.")
                    if parsed_df["1 - v/v0"].max() > 1.0:
                        parsed_df["1 - v/v0"] = parsed_df["1 - v/v0"] / 100.0
                    if parsed_df["Sol Fraction (s)"].max() > 1.0:
                        parsed_df["Sol Fraction (s)"] = parsed_df["Sol Fraction (s)"] / 100.0
                    if "Initial Sol (S₀)" in parsed_df and parsed_df["Initial Sol (S₀)"].max() > 1.0:
                        parsed_df["Initial Sol (S₀)"] = parsed_df["Initial Sol (S₀)"] / 100.0

                parsed_df["1 - v/v0"] = parsed_df["1 - v/v0"].clip(0.0, 1.0)
                parsed_df["Sol Fraction (s)"] = parsed_df["Sol Fraction (s)"].clip(0.0, 1.0)

                st.session_state["active_df"] = parsed_df
                st.sidebar.success(f"✓ Loaded {len(parsed_df)} samples successfully!")
        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded file: {e}")

elif input_mode == "🧪 Built-in Benchmark Datasets":
    chosen_dataset_key = st.sidebar.selectbox("Choose Benchmark Dataset", list(DEFAULT_DATASETS.keys()))
    dataset_info = DEFAULT_DATASETS[chosen_dataset_key]
    st.sidebar.info(dataset_info["description"])
    st.session_state["active_df"] = dataset_info["data"].copy()

else:  # Manual Interactive Data Entry
    st.sidebar.markdown("### Manual Data Editor")
    manual_default = pd.DataFrame({
        "Sample Name": ["Trial-A1", "Trial-A2", "Trial-A3", "Trial-A4"],
        "Initial Sol (S₀)": [0.02, 0.02, 0.02, 0.02],
        "1 - v/v0": [0.40, 0.65, 0.82, 0.94],
        "Sol Fraction (s)": [0.05, 0.10, 0.19, 0.35]
    })
    edited_df = st.sidebar.data_editor(manual_default, num_rows="dynamic", use_container_width=True)
    st.session_state["active_df"] = edited_df

# Global S0 Slider for baseline comparison
global_s0 = st.sidebar.slider(
    "Global Baseline Initial Sol (S₀)",
    min_value=0.001,
    max_value=0.200,
    value=0.020,
    step=0.001,
    format="%.3f",
    help="Default virgin rubber extractable fraction used when per-sample S₀ is not provided."
)

active_df = st.session_state["active_df"].copy()

# Fill missing S0 with global slider value
if "Initial Sol (S₀)" not in active_df.columns or active_df["Initial Sol (S₀)"].isna().all():
    active_df["Initial Sol (S₀)"] = global_s0
else:
    active_df["Initial Sol (S₀)"] = active_df["Initial Sol (S₀)"].fillna(global_s0).clip(0.0001, 0.9999)


# ==============================================================================
# 5. AUTOMATED MECHANISTIC ANALYSIS & DISTANCE CALCULATIONS
# ==============================================================================
analysis_records = []
for _, row in active_df.iterrows():
    sname = str(row.get("Sample Name", "Sample"))
    x_val = float(row.get("1 - v/v0", 0.0))
    s_val = float(row.get("Sol Fraction (s)", 0.0))
    s0_val = float(row.get("Initial Sol (S₀)", global_s0))

    eval_result = evaluate_sample_mechanism(x_val, s_val, s0_val)
    analysis_records.append({
        "Sample Name": sname,
        "Initial Sol (s0)": s0_val,
        "1 - v/v0": x_val,
        "Sol Fraction (s)": s_val,
        "Distance to CL Scission (Δx_CL)": eval_result["dx_cl"],
        "Distance to MC Scission (Δx_MC)": eval_result["dx_mc"],
        "Crosslink Scission (%)": eval_result["cl_ratio"],
        "Chain Scission (%)": eval_result["mc_ratio"],
        "Classified Mechanism": eval_result["classification"],
        "Quality Rating": eval_result["quality_rating"],
        "Color": eval_result["color"]
    })

analysis_df = pd.DataFrame(analysis_records)


# ==============================================================================
# 6. HIGH-LEVEL KPI METRICS BAR
# ==============================================================================
avg_cl_scission = analysis_df["Crosslink Scission (%)"].mean()
cl_dominated_count = sum(1 for c in analysis_df["Classified Mechanism"] if "Crosslink" in c)
mc_dominated_count = sum(1 for c in analysis_df["Classified Mechanism"] if "Chain" in c or "Degradation" in c)
mixed_count = len(analysis_df) - cl_dominated_count - mc_dominated_count

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Samples Evaluated", f"{len(analysis_df)}")
m2.metric("Mean Crosslink Selectivity", f"{avg_cl_scission:.1f}%")
m3.metric("Selective Devulcanization", f"{cl_dominated_count} samples", delta="Ideal" if cl_dominated_count > 0 else None)
m4.metric("Degraded / Main-Chain", f"{mc_dominated_count} samples", delta="-Degraded" if mc_dominated_count > 0 else None, delta_color="inverse")


# ==============================================================================
# 7. INTERACTIVE PLOTLY HORIKX CHART WITH INTERNAL LEGEND & BALANCED MARGINS
# ==============================================================================
st.markdown("### 📈 Interactive Horikx Diagram")

fig = go.Figure()

# Distinct S0 curves to render
unique_s0_list = sorted(list(analysis_df["Initial Sol (s0)"].unique()))
palette_cl = ["#10b981", "#059669", "#047857", "#065f46"]
palette_mc = ["#ef4444", "#dc2626", "#b91c1c", "#991b1b"]

for idx, s0_val in enumerate(unique_s0_list):
    curves = calculate_horikx_theoretical_curves(s0_val, num_points=400)
    cl_color = palette_cl[idx % len(palette_cl)]
    mc_color = palette_mc[idx % len(palette_mc)]
    suffix = f" (S₀={s0_val:.3f})" if len(unique_s0_list) > 1 else ""

    # 1. Theoretical Crosslink Scission Curve
    fig.add_trace(go.Scatter(
        x=curves["x_crosslink"],
        y=curves["s_vals"],
        mode="lines",
        name=f"Crosslink Scission{suffix}",
        line=dict(color=cl_color, width=2.5, dash="solid"),
        hovertemplate=f"<b>Crosslink Scission (S₀={s0_val:.3f})</b><br>1 - ν/ν₀: %{{x:.3f}}<br>Sol (s): %{{y:.3f}}<extra></extra>"
    ))

    # 2. Theoretical Main-Chain Scission Curve
    fig.add_trace(go.Scatter(
        x=curves["x_mainchain"],
        y=curves["s_vals"],
        mode="lines",
        name=f"Main-Chain Scission{suffix}",
        line=dict(color=mc_color, width=2.5, dash="dash"),
        hovertemplate=f"<b>Main-Chain Scission (S₀={s0_val:.3f})</b><br>1 - ν/ν₀: %{{x:.3f}}<br>Sol (s): %{{y:.3f}}<extra></extra>"
    ))

# 3. Experimental Data Points
for _, row in analysis_df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["1 - v/v0"]],
        y=[row["Sol Fraction (s)"]],
        mode="markers+text",
        name=f"{row['Sample Name']}",
        text=[row["Sample Name"]],
        textposition="top center",
        textfont=dict(size=10, color="#1e293b"),
        marker=dict(
            size=11,
            color=row["Color"],
            line=dict(width=1.5, color="#ffffff"),
            symbol="circle"
        ),
        hovertemplate=(
            f"<b>{row['Sample Name']}</b><br>"
            f"1 - ν/ν₀: %{{x:.4f}}<br>"
            f"Sol Fraction (s): %{{y:.4f}}<br>"
            f"Initial Sol (S₀): {row['Initial Sol (s0)']:.4f}<br>"
            f"Δx(CL): {row['Distance to CL Scission (Δx_CL)']:.4f}<br>"
            f"Δx(MC): {row['Distance to MC Scission (Δx_MC)']:.4f}<br>"
            f"Crosslink Scission: {row['Crosslink Scission (%)']:.1f}%<br>"
            f"Mechanism: {row['Classified Mechanism']}<extra></extra>"
        ),
        legendgroup="experimental_samples"
    ))

# Configure Plotly layout with fixed margins (l=80, r=80, t=80, b=80) and internal legend
fig.update_layout(
    title=dict(
        text="<b>Horikx Plot Analysis for Rubber Devulcanization</b>",
        x=0.5,
        xanchor="center",
        font=dict(size=17, color="#1e293b")
    ),
    xaxis=dict(
        title=dict(
            text="<b>Relative Decrease in Crosslink Density (1 - ν<sub>f</sub> / ν<sub>i</sub>)</b>",
            font=dict(size=12, color="#334155")
        ),
        range=[0.0, 1.02],
        tickformat=".2f",
        dtick=0.1,
        gridcolor="#e2e8f0",
        showline=True,
        linewidth=1.2,
        linecolor="#64748b",
        zeroline=False,
        automargin=True
    ),
    yaxis=dict(
        title=dict(
            text="<b>Measured Sol Fraction (S<sub>f</sub>)</b>",
            font=dict(size=12, color="#334155")
        ),
        range=[0.0, 1.02],
        tickformat=".2f",
        dtick=0.1,
        gridcolor="#e2e8f0",
        showline=True,
        linewidth=1.2,
        linecolor="#64748b",
        zeroline=False,
        automargin=True
    ),
    legend=dict(
        title=dict(text="<b>Legend</b>", font=dict(size=10.5, color="#1e293b")),
        orientation="v",
        y=0.96,
        x=0.03,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255, 255, 255, 0.88)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        font=dict(size=9.5),
        tracegroupgap=2
    ),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    height=720,
    margin=dict(l=80, r=80, t=80, b=80)
)

plotly_config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'horikx_plot_highres_complete',
        'height': 900,
        'width': 1200,
        'scale': 2
    },
    'displaylogo': False,
    'responsive': True
}

st.plotly_chart(fig, use_container_width=True, config=plotly_config)


# ==============================================================================
# 8. PUBLICATION EXPORT BUTTONS (1200x900 PNG, 300 DPI Matplotlib, HTML)
# ==============================================================================
st.markdown("### 💾 Export Publication Figures & Data")
col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    # Matplotlib 300 DPI figure generator
    def generate_matplotlib_figure():
        mpl_fig, ax = plt.subplots(figsize=(10.0, 7.5), dpi=300)
        
        for idx_s0, s0_val in enumerate(unique_s0_list):
            curves = calculate_horikx_theoretical_curves(s0_val, num_points=400)
            suffix = f" ($S_0={s0_val:.3f}$)" if len(unique_s0_list) > 1 else ""
            ax.plot(
                curves["x_crosslink"], curves["s_vals"],
                label=f"Crosslink Scission{suffix}",
                color="#10b981", lw=2.2, linestyle="-"
            )
            ax.plot(
                curves["x_mainchain"], curves["s_vals"],
                label=f"Main-Chain Scission{suffix}",
                color="#ef4444", lw=2.2, linestyle="--"
            )

        for _, row in analysis_df.iterrows():
            ax.scatter(
                row["1 - v/v0"], row["Sol Fraction (s)"],
                color=row["Color"], s=65, edgecolor="#1e293b", zorder=5
            )
            ax.annotate(
                row["Sample Name"],
                (row["1 - v/v0"], row["Sol Fraction (s)"]),
                textcoords="offset points", xytext=(0, 7),
                ha='center', fontsize=8, fontweight='semibold'
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"Relative Decrease in Crosslink Density $\left(1 - \frac{\nu_f}{\nu_i}\right)$", fontsize=11, fontweight="semibold")
        ax.set_ylabel(r"Sol Fraction ($S_f$)", fontsize=11, fontweight="semibold")
        ax.set_title("Horikx Plot Analysis for Rubber Devulcanization", fontsize=13, fontweight="bold", pad=14)
        ax.grid(True, linestyle=":", alpha=0.5, color="gray")

        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.90,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            fontsize=8.5,
            borderaxespad=0.8
        )
        plt.tight_layout()
        mpl_fig.subplots_adjust(bottom=0.12, top=0.92, left=0.10, right=0.95)
        return mpl_fig

    try:
        mpl_figure = generate_matplotlib_figure()
        buf_mpl = io.BytesIO()
        mpl_figure.savefig(buf_mpl, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        buf_mpl.seek(0)
        plt.close(mpl_figure)

        st.download_button(
            label="📷 Download 300 DPI Matplotlib PNG",
            data=buf_mpl,
            file_name="horikx_plot_publication_300dpi.png",
            mime="image/png",
            help="Crisp publication raster with internal legend and pad_inches=0.2 ensuring zero truncation."
        )
    except Exception as e:
        st.warning(f"Matplotlib export notice: {e}")

with col_exp2:
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=900, scale=2)
        st.download_button(
            label="📊 Download Plotly PNG (1200x900, 2x)",
            data=img_bytes,
            file_name="horikx_plotly_highres_complete.png",
            mime="image/png",
            help="Vector rasterization with fixed margins (l=80, r=80, t=80, b=80) and internal legend."
        )
    except Exception:
        st.download_button(
            label="📊 Download Interactive HTML",
            data=fig.to_html(include_plotlyjs="cdn"),
            file_name="horikx_plotly_interactive.html",
            mime="text/html",
            help="Download interactive HTML containing Plotly chart and toggles."
        )

with col_exp3:
    csv_export = analysis_df.drop(columns=["Color"]).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download Analysis Table (CSV)",
        data=csv_export,
        file_name="horikx_mechanistic_analysis_report.csv",
        mime="text/csv"
    )


# ==============================================================================
# 9. AUTOMATED ENGINEERING INSIGHTS & FORMATTED SUMMARY TABLE
# ==============================================================================
st.markdown("### 📋 Mechanistic Classification & Distance Table")

# Format display dataframe with clean LaTeX / Unicode headers
display_df = pd.DataFrame({
    "Sample Name": analysis_df["Sample Name"],
    "Initial Sol (S₀)": analysis_df["Initial Sol (s0)"].apply(lambda v: f"{v:.4f}"),
    "1 - ν/ν₀": analysis_df["1 - v/v0"].apply(lambda v: f"{v:.4f}"),
    "Sol Fraction (S_f)": analysis_df["Sol Fraction (s)"].apply(lambda v: f"{v:.4f}"),
    r"Distance to CL Scission (Δx_CL)": analysis_df["Distance to CL Scission (Δx_CL)"].apply(lambda v: f"{v:.4f}"),
    r"Distance to MC Scission (Δx_MC)": analysis_df["Distance to MC Scission (Δx_MC)"].apply(lambda v: f"{v:.4f}"),
    "Crosslink Scission (%)": analysis_df["Crosslink Scission (%)"].apply(lambda v: f"{v:.1f}%"),
    "Chain Scission (%)": analysis_df["Chain Scission (%)"].apply(lambda v: f"{v:.1f}%"),
    "Classified Mechanism": analysis_df["Classified Mechanism"],
    "Quality Assessment": analysis_df["Quality Rating"]
})

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Mechanistic Insights Summary
with st.expander("🔍 Mechanistic Insights & Theoretical Background", expanded=True):
    st.markdown("""
    #### Theoretical Basis of the Horikx Method (1956)
    The Horikx diagnostic curve relates the **relative decrease in crosslink density** ($1 - \\nu_f / \\nu_i$) to the **increase in sol fraction** ($S_f$).
    
    1. **Selective Crosslink Scission ($1 - \\nu_f / \\nu_i = 1 - \\frac{(1 - S_f^{1/2})^2}{(1 - S_0^{1/2})^2}$)**:
       Cleavage occurs strictly at the vulcanization crosslink bonds (e.g., monosulfidic, disulfidic, polysulfidic linkages). The sol fraction remains low until very high levels of network breakdown ($>85\\%$) are reached. This represents **ideal devulcanization** where polymer molecular weight is preserved.
       
    2. **Main-Chain Scission ($1 - \\nu_f / \\nu_i = 1 - \\frac{1 - S_f^{1/2}}{1 - S_0^{1/2}}$)**:
       Cleavage occurs randomly along the primary polymer carbon-carbon backbone. Sol fraction increases rapidly even at moderate crosslink loss, yielding low-molecular-weight oligomers and severe elastomeric degradation.
       
    3. **Distance Metrics ($\\Delta x_{CL}$ and $\\Delta x_{MC}$)**:
       Represent the orthogonal horizontal distances from each experimental point to the respective theoretical Horikx boundary at that sol fraction level.
    """)
