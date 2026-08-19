"""
Horikx Plot Analysis for Rubber Devulcanization
A fully standalone, interactive Streamlit web application with:
  - Interactive Plotly Graph Integration with native legend toggling (st.plotly_chart)
  - Fixed PNG Export Functions (Zero truncation with bbox_inches='tight', pad_inches=0.3, dpi=300, width=1000, height=600, scale=2)
  - Interactive Sidebar Controls to declutter multi-S0 and multi-trial plots:
      * Multiselect to toggle active S0 theoretical curves
      * Independent checkboxes for Main-Chain and Crosslink scission curves
      * Multiselect filter for Visible Experimental Trials
  - Dual Dataset Input via Radio Selection:
      Option 1: Upload Excel/CSV File (with st.file_uploader & column mapping)
      Option 2: Manual Data Entry (Interactive spreadsheet with st.data_editor)
  - Per-Trial / Per-Sample Initial Sol Fraction (S_0) Support
  - Dynamic Curve Rendering & Multi-Trial Plotting for Multiple Unique S_0 Values
  - Individual Distance Calculation in Automated Mechanism Analysis using each Sample's S_0
  - High-resolution publication exports and automated insights

Requirements:
    pip install streamlit pandas numpy plotly matplotlib openpyxl kaleido

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

# ---------------------------------------------------------
# 1. Page Configuration & Header
# ---------------------------------------------------------
st.set_page_config(
    page_title="Horikx Plot Analysis - Rubber Devulcanization",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Horikx Plot Analysis for Rubber Devulcanization")
st.markdown(
    """
    Evaluate devulcanization efficiency by comparing experimental sol fraction ($S_f$ or $s$) 
    and crosslink density decrease ($1 - \nu_f/\nu_i$) against theoretical **Horikx (1956)** scission curves.
    **Features interactive Plotly rendering with multi-S₀ curves, per-sample baseline analysis, and truncation-free high-res PNG export.**
    """
)

# ---------------------------------------------------------
# 2. Calculation Engine
# ---------------------------------------------------------
def calculate_horikx_curves(s0_val, n_points=300):
    """
    Computes theoretical Horikx curve coordinates (Horikx, 1956) for a given S_0 value.
    1. Main-Chain Degradation:
       1 - (nu_f / nu_i) = 1 - [(1 - sqrt(S_f))^2 / (1 - sqrt(S_0))^2]
       Solved for S_f: S_f = [1 - (1 - sqrt(S_0)) * sqrt(1 - x)]^2
    2. Selective Crosslink Cleavage:
       1 - (nu_f / nu_i) = 1 - [gamma_f * (1 - sqrt(S_f))^2] / [gamma_i * (1 - sqrt(S_0))^2]
       where gamma(s) = 1 / (s + sqrt(s))
    """
    s0_val = float(max(0.0001, min(0.3, s0_val)))
    
    # 1. Main-chain scission curve (x from 0 to 1)
    x_mainchain = np.linspace(0.0, 1.0, n_points)
    term = (1.0 - np.sqrt(s0_val)) * np.sqrt(np.maximum(0.0, 1.0 - x_mainchain))
    s_mainchain = (1.0 - term) ** 2
    s_mainchain = np.clip(s_mainchain, 0.0, 1.0)
    
    # 2. Crosslink scission curve (parameterized by s from s0_val to 1.0)
    s_crosslink = np.linspace(s0_val, 1.0, n_points)
    denom = (1.0 - np.sqrt(s0_val)) ** 2
    numerator = (1.0 - np.sqrt(s_crosslink)) ** 2
    gamma_ratio = (s0_val + np.sqrt(s0_val)) / (s_crosslink + np.sqrt(s_crosslink))
    ratio = gamma_ratio * (numerator / denom)
    x_crosslink = np.clip(1.0 - ratio, 0.0, 1.0)
    
    return x_crosslink, s_crosslink, x_mainchain, s_mainchain

def evaluate_data_points(df_points, fallback_s0=0.02, scale_factor=1.0):
    """
    Quantitatively analyzes each experimental point against theoretical curves
    using that specific sample's own S_0 (S_{0, sample}).
    """
    results = []
    
    for idx, row in df_points.iterrows():
        sample_name = str(row.get("Sample Name", f"Sample {idx+1}"))
        
        # 1. Extract sample-specific S_0 value
        raw_s0 = None
        for col_name in ["Initial Sol Fraction (s0)", "Initial Sol Fraction", "s0", "S0", "S_0", "Si", "S_i"]:
            if col_name in row and pd.notna(row[col_name]):
                raw_s0 = row[col_name]
                break
        
        if raw_s0 is not None:
            try:
                parsed_s0 = float(str(raw_s0).replace("%", "").strip())
                if parsed_s0 > 1.0 or (scale_factor > 1.0 and parsed_s0 > 0.3):
                    parsed_s0 = parsed_s0 / 100.0
                sample_s0 = float(np.clip(parsed_s0, 0.0001, 0.3000))
            except (ValueError, TypeError):
                sample_s0 = float(fallback_s0)
        else:
            sample_s0 = float(fallback_s0)

        # 2. Extract Crosslink Density Decrease (1 - v/v0)
        raw_x = None
        for col_name in ["Crosslink Density Decrease (1 - v/v0)", "Crosslink Decrease (1 - v/v0)", "1 - v/v0", "Crosslink Density Decrease"]:
            if col_name in row and pd.notna(row[col_name]):
                raw_x = row[col_name]
                break
        if raw_x is None:
            raw_x = row.iloc[2] if len(row) > 2 else 0.0

        # 3. Extract Sol Fraction (s)
        raw_s = None
        for col_name in ["Sol Fraction (s)", "Sol Fraction (Sf)", "Sol Fraction", "s", "Sf"]:
            if col_name in row and pd.notna(row[col_name]):
                raw_s = row[col_name]
                break
        if raw_s is None:
            raw_s = row.iloc[3] if len(row) > 3 else 0.0

        try:
            val_x = float(str(raw_x).replace("%", "").strip())
            val_s = float(str(raw_s).replace("%", "").strip())
        except (ValueError, TypeError):
            continue
            
        x_val = val_x / scale_factor
        s_val = val_s / scale_factor
        
        x_val = float(np.clip(x_val, 0.0, 1.0))
        s_val = float(np.clip(s_val, sample_s0, 1.0))
        
        # Calculate theoretical values using THIS sample's own S_0 (S_{0, sample})
        denom_mc = (1.0 - np.sqrt(sample_s0)) ** 2
        num_s = (1.0 - np.sqrt(s_val)) ** 2
        x_mc_th = float(np.clip(1.0 - (num_s / denom_mc), 0.0, 1.0))
        
        gamma_r = (sample_s0 + np.sqrt(sample_s0)) / (s_val + np.sqrt(s_val))
        x_cl_th = float(np.clip(1.0 - (gamma_r * (num_s / denom_mc)), 0.0, 1.0))
        
        # Individual Distances to theoretical curves for this specific sample's S_0
        dist_cl = abs(x_val - x_cl_th)
        dist_mc = abs(x_val - x_mc_th)
        
        # Selectivity Calculation (Crosslink Scission Fraction)
        if x_cl_th > x_mc_th:
            selectivity = float(np.clip(((x_val - x_mc_th) / (x_cl_th - x_mc_th)) * 100.0, 0.0, 100.0))
        else:
            selectivity = 50.0
            
        chain_scission = 100.0 - selectivity
        
        # Mechanism Classification
        if selectivity >= 80.0:
            classification = "Selective Crosslink Cleavage"
            rating = "⭐⭐⭐⭐⭐ (Ideal Devulcanization)"
        elif selectivity >= 60.0:
            classification = "Predominantly Crosslink Scission"
            rating = "⭐⭐⭐⭐ (Good Devulcanization)"
        elif selectivity >= 40.0:
            classification = "Mixed Scission Mechanism"
            rating = "⭐⭐⭐ (Moderate Cleavage)"
        elif selectivity >= 20.0:
            classification = "Predominantly Main-Chain Scission"
            rating = "⭐⭐ (High Degradation)"
        else:
            classification = "Severe Main-Chain Degradation"
            rating = "⭐ (Severe Backbone Cleavage)"
            
        results.append({
            "Sample Name": sample_name,
            "Initial Sol Fraction (s0)": sample_s0,
            "1 - v/v0": x_val,
            "Sol Fraction (s)": s_val,
            "Theoretical X (Crosslink)": x_cl_th,
            "Theoretical X (Main-Chain)": x_mc_th,
            "Dist to Crosslink": dist_cl,
            "Dist to Main-Chain": dist_mc,
            "Crosslink Scission (%)": selectivity,
            "Chain Scission (%)": chain_scission,
            "Classification": classification,
            "Rating": rating,
            "Notes": str(row.get("Notes", ""))
        })
        
    return pd.DataFrame(results)

# ---------------------------------------------------------
# 3. Sidebar Controls: Model Parameters & Display Filters
# ---------------------------------------------------------
st.sidebar.header("⚙️ Model Parameters")

default_s0 = st.sidebar.number_input(
    "Default Initial Sol Fraction ($S_0$ / $S_i$)",
    min_value=0.0001,
    max_value=0.3000,
    value=0.0200,
    step=0.0050,
    format="%.4f",
    help="Default baseline sol fraction of virgin rubber vulcanizate. Applied when sample-level S_0 is not specified."
)

st.sidebar.caption("💡 Typical Rubber Vulcanizate $S_0$ Presets:")
preset_cols = st.sidebar.columns(4)
if preset_cols[0].button("NR: 0.015"):
    default_s0 = 0.015
if preset_cols[1].button("EPDM: 0.025"):
    default_s0 = 0.025
if preset_cols[2].button("SBR: 0.035"):
    default_s0 = 0.035
if preset_cols[3].button("Reclaim: 0.050"):
    default_s0 = 0.050

unit_format = st.sidebar.radio(
    "Data Scale in Table",
    options=["Fraction (0.0 to 1.0)", "Percentage (0% to 100%)"],
    index=0,
    help="Choose whether experimental values are entered as fractions (0.0 - 1.0) or percentages (0% - 100%)."
)

# ---------------------------------------------------------
# 4. Dataset Input (Radio: Upload vs Manual Entry)
# ---------------------------------------------------------
st.subheader("📋 Dataset Input & Multi-Sample Setup")

# Initialize default experimental dataset in session state with per-sample S_0 values
if "manual_data" not in st.session_state:
    st.session_state["manual_data"] = pd.DataFrame([
        {"Sample Name": "Trial 1 (NR 180°C)", "Initial Sol Fraction (s0)": 0.015, "Crosslink Density Decrease (1 - v/v0)": 0.35, "Sol Fraction (s)": 0.06, "Notes": "NR vulcanizate matrix"},
        {"Sample Name": "Trial 2 (EPDM 200°C)", "Initial Sol Fraction (s0)": 0.025, "Crosslink Density Decrease (1 - v/v0)": 0.58, "Sol Fraction (s)": 0.12, "Notes": "EPDM microwave devulc"},
        {"Sample Name": "Trial 3 (SBR 220°C)", "Initial Sol Fraction (s0)": 0.035, "Crosslink Density Decrease (1 - v/v0)": 0.74, "Sol Fraction (s)": 0.22, "Notes": "SBR thermo-chemical"},
        {"Sample Name": "Trial 4 (SBR 240°C)", "Initial Sol Fraction (s0)": 0.035, "Crosslink Density Decrease (1 - v/v0)": 0.88, "Sol Fraction (s)": 0.42, "Notes": "Onset of chain scission"},
        {"Sample Name": "Trial 5 (Reclaim 260°C)", "Initial Sol Fraction (s0)": 0.050, "Crosslink Density Decrease (1 - v/v0)": 0.95, "Sol Fraction (s)": 0.68, "Notes": "Tire reclaim feedstock"},
    ])

# Radio button to select dataset input method
input_option = st.radio(
    "Choose Dataset Input Option:",
    options=["Option 1: Upload Excel/CSV File", "Option 2: Manual Data Entry"],
    index=1,
    horizontal=True,
    help="Select Option 1 to upload an existing spreadsheet, or Option 2 to edit data points directly in the interactive table."
)

current_active_df = None

if input_option == "Option 1: Upload Excel/CSV File":
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Spreadsheet File (.xlsx, .xls, .csv)",
            type=["csv", "xlsx", "xls"],
            help="Upload an Excel or CSV file containing Initial Sol Fraction (s0), Crosslink Density Decrease, and Sol Fraction columns."
        )
    with col_up2:
        st.write("")
        st.write("")
        sample_csv = st.session_state["manual_data"].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV Template with s0",
            data=sample_csv,
            file_name="horikx_multi_s0_template.csv",
            mime="text/csv"
        )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                imported_df = pd.read_excel(uploaded_file)
            else:
                imported_df = pd.read_csv(uploaded_file)
                
            st.success(f"Loaded **{uploaded_file.name}** ({len(imported_df)} rows)")
            
            # Smart column mapping
            cols = list(imported_df.columns)
            def find_match(keys, default_idx=0):
                for i, c in enumerate(cols):
                    clean = str(c).lower().replace(" ", "").replace("_", "").replace("-", "")
                    if any(k in clean for k in keys):
                        return i
                return min(default_idx, len(cols) - 1)
                
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                name_match = st.selectbox("Sample Name Column", ["Auto Index"] + cols, index=find_match(["samplename", "sample", "name", "id"], 0) + 1 if "Sample Name" in cols else 0)
            with m2:
                s0_match = st.selectbox("Initial Sol (s0) Column", [f"Default s0 ({default_s0:.4f})"] + cols, index=find_match(["initialsol", "s0", "si", "initials0"], 0) + 1 if any(k in str(cols).lower() for k in ["s0", "si", "initial"]) else 0)
            with m3:
                x_match = st.selectbox("Crosslink Decrease Column (X)", cols, index=find_match(["crosslink", "decrease", "1v", "xd", "loss", "density"], 1))
            with m4:
                s_match = st.selectbox("Sol Fraction Column (Y)", cols, index=find_match(["sol", "fraction", "soluble", "s"], 2))
                
            mapped_df = pd.DataFrame()
            mapped_df["Sample Name"] = imported_df[name_match] if name_match != "Auto Index" else [f"Sample {i+1}" for i in range(len(imported_df))]
            
            if s0_match.startswith("Default s0"):
                mapped_df["Initial Sol Fraction (s0)"] = default_s0
            else:
                mapped_df["Initial Sol Fraction (s0)"] = pd.to_numeric(imported_df[s0_match].astype(str).str.replace("%", "").str.strip(), errors='coerce').fillna(default_s0)

            mapped_df["Crosslink Density Decrease (1 - v/v0)"] = pd.to_numeric(imported_df[x_match].astype(str).str.replace("%", "").str.strip(), errors='coerce')
            mapped_df["Sol Fraction (s)"] = pd.to_numeric(imported_df[s_match].astype(str).str.replace("%", "").str.strip(), errors='coerce')
            mapped_df["Notes"] = [f"Imported from {uploaded_file.name}"] * len(imported_df)
            current_active_df = mapped_df.dropna(subset=["Crosslink Density Decrease (1 - v/v0)", "Sol Fraction (s)"])
            
            st.dataframe(current_active_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")
            current_active_df = st.session_state["manual_data"]
    else:
        st.info("ℹ️ No file uploaded yet. Showing default sample dataset below. Upload an Excel/CSV file above or switch to Option 2 for manual entry.")
        current_active_df = st.session_state["manual_data"]

else:  # Option 2: Manual Data Entry
    st.markdown(
        "Directly **add**, **edit**, or **delete** experimental rows in the interactive table below. "
        "Each row can have its own **Initial Sol Fraction ($S_0$)** value. The Horikx Plotly graph and analysis update in real time."
    )
    
    edited_df = st.data_editor(
        st.session_state["manual_data"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Sample Name": st.column_config.TextColumn(
                "Sample Name",
                default="Sample",
                required=True,
                help="Name / ID of the vulcanizate or devulcanized sample"
            ),
            "Initial Sol Fraction (s0)": st.column_config.NumberColumn(
                "Initial Sol Fraction (s0)",
                min_value=0.0001,
                max_value=0.3000 if "Fraction" in unit_format else 30.0,
                step=0.0050 if "Fraction" in unit_format else 0.5,
                format="%.4f" if "Fraction" in unit_format else "%.2f%%",
                default=default_s0,
                required=False,
                help="Per-sample virgin rubber initial sol fraction (S_0). If left blank, falls back to default."
            ),
            "Crosslink Density Decrease (1 - v/v0)": st.column_config.NumberColumn(
                "Crosslink Density Decrease (1 - v/v0)",
                min_value=0.0,
                max_value=1.0 if "Fraction" in unit_format else 100.0,
                step=0.01 if "Fraction" in unit_format else 1.0,
                format="%.4f" if "Fraction" in unit_format else "%.1f%%",
                required=True,
                help="Relative reduction in crosslink density: 1 - (nu_f / nu_i)"
            ),
            "Sol Fraction (s)": st.column_config.NumberColumn(
                "Sol Fraction (s)",
                min_value=0.0,
                max_value=1.0 if "Fraction" in unit_format else 100.0,
                step=0.01 if "Fraction" in unit_format else 1.0,
                format="%.4f" if "Fraction" in unit_format else "%.1f%%",
                required=True,
                help="Soluble polymer fraction (S_f)"
            ),
            "Notes": st.column_config.TextColumn("Notes / Conditions (Optional)")
        },
        key="manual_table_editor"
    )
    st.session_state["manual_data"] = edited_df
    current_active_df = edited_df

# Process & Evaluate Data Points
scale = 100.0 if "Percentage" in unit_format else 1.0
evaluated_df = evaluate_data_points(current_active_df, fallback_s0=default_s0, scale_factor=scale)

# ---------------------------------------------------------
# 5. Interactive Display Controls (Sidebar: Avoid Plot Clutter)
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Plot Filter & Declutter Controls")

# Extract all unique S_0 values present in active data
if not evaluated_df.empty:
    dataset_unique_s0 = sorted([round(float(s), 4) for s in evaluated_df["Initial Sol Fraction (s0)"].unique()])
else:
    dataset_unique_s0 = [round(float(default_s0), 4)]

# 1. Multiselect for active S_0 theoretical curves
selected_s0_curves = st.sidebar.multiselect(
    "Select Active S₀ Theoretical Curves to Display:",
    options=dataset_unique_s0,
    default=dataset_unique_s0,
    help="Filter which S_0 theoretical baseline curves appear on the plot to prevent visual clutter."
)

# 2. Independent Checkboxes for Main-Chain and Crosslink Curves
col_chk1, col_chk2 = st.sidebar.columns(2)
with col_chk1:
    show_crosslink_curves = st.checkbox("Selective Crosslink", value=True, help="Toggle Selective Crosslink Scission (solid) curves")
with col_chk2:
    show_mainchain_curves = st.checkbox("Main-Chain Scission", value=True, help="Toggle Main-Chain Degradation (dashed) curves")

show_devulc_fill = st.sidebar.checkbox("Shade Devulcanization Corridor", value=True, help="Highlight the selective devulcanization zone between curves")
show_s0_baselines = st.sidebar.checkbox("Show Horizontal S₀ Reference Baselines", value=False, help="Display horizontal reference lines at y = S_0")

# 3. Multiselect for Experimental Trials Filter
if not evaluated_df.empty:
    all_trials = evaluated_df["Sample Name"].tolist()
    visible_trials = st.sidebar.multiselect(
        "Filter Visible Experimental Trials on Plot:",
        options=all_trials,
        default=all_trials,
        help="Select which experimental data points to render. Unselected trials will be hidden from the graph."
    )
else:
    visible_trials = []

# ---------------------------------------------------------
# 6. Interactive Plotly Graph Integration (with Legend Toggling & Margin Padding)
# ---------------------------------------------------------
st.subheader("📊 Interactive Horikx Diagram (Plotly Engine)")
st.caption("💡 **Tip:** Click any curve or sample in the plot legend on the right to instantly hide/show it dynamically!")

fig = go.Figure()

# Distinct color palette for multiple S_0 curves
cl_colors = ["#16A34A", "#0D9488", "#2563EB", "#7C3AED", "#D97706", "#059669"]
mc_colors = ["#DC2626", "#EA580C", "#DB2777", "#9333EA", "#475569", "#E11D48"]

# 1. Render Theoretical Curves for Selected S_0 values
for i, cur_s0 in enumerate(selected_s0_curves):
    x_cl, s_cl, x_mc, s_mc = calculate_horikx_curves(cur_s0)
    cl_col = cl_colors[i % len(cl_colors)]
    mc_col = mc_colors[i % len(mc_colors)]

    # Main-Chain Scission Curve (Upper Dashed)
    if show_mainchain_curves:
        fig.add_trace(go.Scatter(
            x=x_mc,
            y=s_mc,
            mode='lines',
            name=f'Main-Chain (S₀={cur_s0:.4f})',
            line=dict(color=mc_col, width=2.5, dash='dash'),
            customdata=np.full(len(x_mc), cur_s0),
            hovertemplate=(
                "<b>Main-Chain Scission Curve</b><br>"
                "Baseline S₀: %{customdata:.4f}<br>"
                "1 - ν/ν₀: %{x:.4f}<br>"
                "Sol Fraction (s): %{y:.4f}<extra></extra>"
            ),
            legendgroup=f"s0_{cur_s0}"
        ))

    # Selective Crosslink Scission Curve (Lower Solid)
    if show_crosslink_curves:
        fig.add_trace(go.Scatter(
            x=x_cl,
            y=s_cl,
            mode='lines',
            name=f'Selective CL (S₀={cur_s0:.4f})',
            line=dict(color=cl_col, width=2.5),
            customdata=np.full(len(x_cl), cur_s0),
            hovertemplate=(
                "<b>Selective Crosslink Scission Curve</b><br>"
                "Baseline S₀: %{customdata:.4f}<br>"
                "1 - ν/ν₀: %{x:.4f}<br>"
                "Sol Fraction (s): %{y:.4f}<extra></extra>"
            ),
            legendgroup=f"s0_{cur_s0}"
        ))

    # Devulcanization corridor fill (between Main-chain and Crosslink)
    if show_devulc_fill and show_crosslink_curves and show_mainchain_curves:
        # Interpolate crosslink curve onto mainchain x coordinates
        s_cl_interp = np.interp(x_mc, x_cl, s_cl, left=cur_s0, right=1.0)
        # Create polygon for clean filled area
        x_poly = np.concatenate([x_mc, x_mc[::-1]])
        y_poly = np.concatenate([s_mc, s_cl_interp[::-1]])
        
        fig.add_trace(go.Scatter(
            x=x_poly,
            y=y_poly,
            fill='toself',
            fillcolor=f'rgba({int(cl_col[1:3], 16)}, {int(cl_col[3:5], 16)}, {int(cl_col[5:7], 16)}, 0.08)',
            line=dict(width=0),
            hoverinfo='skip',
            showlegend=False,
            name=f'Devulcanization Zone (S₀={cur_s0:.4f})',
            legendgroup=f"s0_{cur_s0}"
        ))

    # Optional S_0 horizontal baseline
    if show_s0_baselines:
        fig.add_trace(go.Scatter(
            x=[0.0, 1.0],
            y=[cur_s0, cur_s0],
            mode='lines',
            name=f'Baseline S₀={cur_s0:.4f}',
            line=dict(color=cl_col, width=1.2, dash='dot'),
            hoverinfo='skip',
            showlegend=True,
            legendgroup=f"s0_{cur_s0}"
        ))

# Diagonal Reference Line (1:1 equivalence s = 1 - v/v0)
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Reference (y = x)',
    line=dict(color='#cbd5e1', width=1, dash='dot'),
    hoverinfo='skip',
    showlegend=False
))

# 2. Render Experimental Trials
if not evaluated_df.empty:
    filtered_points = evaluated_df[evaluated_df["Sample Name"].isin(visible_trials)]
    
    # Render each sample point as an individual trace for native legend toggling
    for _, pt in filtered_points.iterrows():
        # Color by selectivity rating
        sel = pt["Crosslink Scission (%)"]
        pt_color = "#16A34A" if sel >= 75 else ("#2563EB" if sel >= 50 else ("#D97706" if sel >= 30 else "#DC2626"))
        
        fig.add_trace(go.Scatter(
            x=[pt["1 - v/v0"]],
            y=[pt["Sol Fraction (s)"]],
            mode='markers+text',
            name=f"{pt['Sample Name']} (S₀={pt['Initial Sol Fraction (s0)']:.3f})",
            text=[f" {pt['Sample Name']}"],
            textposition="top right",
            textfont=dict(size=10, color="#1e293b"),
            marker=dict(
                size=12,
                color=pt_color,
                line=dict(width=1.5, color="#ffffff"),
                symbol="circle"
            ),
            customdata=[[
                pt["Sample Name"],
                pt["Initial Sol Fraction (s0)"],
                pt["Crosslink Scission (%)"],
                pt["Chain Scission (%)"],
                pt["Classification"],
                pt["Dist to Crosslink"],
                pt["Dist to Main-Chain"]
            ]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Initial Sol (S₀): %{customdata[1]:.4f}<br>"
                "1 - ν/ν₀: %{x:.4f}<br>"
                "Sol Fraction (s): %{y:.4f}<br>"
                "------------------------------<br>"
                "<b>Mechanism:</b> %{customdata[4]}<br>"
                "<b>Selectivity:</b> %{customdata[2]:.1f}% Crosslink Scission<br>"
                "Dist to Crosslink (Δx): %{customdata[5]:.4f}<br>"
                "Dist to Main-Chain (Δx): %{customdata[6]:.4f}<extra></extra>"
            ),
            legendgroup="experimental_trials"
        ))

# Configure Plotly Layout with Explicit Padding & Margins to Prevent Truncation
fig.update_layout(
    title=dict(
        text="<b>Horikx Plot Analysis for Rubber Devulcanization</b>",
        x=0.5,
        font=dict(size=16, color="#0f172a")
    ),
    xaxis=dict(
        title=dict(
            text="<b>Relative Decrease in Crosslink Density (1 - ν<sub>f</sub> / ν<sub>i</sub>)</b>",
            font=dict(size=12, color="#1e293b")
        ),
        range=[0.0, 1.02],
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=True,
        zerolinecolor="#cbd5e1",
        tickformat=".2f",
        automargin=True
    ),
    yaxis=dict(
        title=dict(
            text="<b>Sol Fraction (S<sub>f</sub>)</b>",
            font=dict(size=12, color="#1e293b")
        ),
        range=[0.0, 1.02],
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=True,
        zerolinecolor="#cbd5e1",
        tickformat=".2f",
        automargin=True
    ),
    legend=dict(
        title=dict(text="<b>Interactive Legend (Click to Toggle)</b>", font=dict(size=11)),
        itemclick="toggle",              # Clicking trace toggles its visibility
        itemdoubleclick="toggleothers",  # Double-clicking isolates that trace
        orientation="h",                 # Strictly horizontal layout below plot
        y=-0.30,                         # Positioned below X-axis at the bottom-center
        x=0.5,                           # Centered horizontally
        xanchor="center",
        yanchor="top",
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        font=dict(size=10)
    ),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    hovermode="closest",
    height=700,
    margin=dict(l=90, r=90, t=80, b=180)
)

# Configure Plotly interactive toolbar download options for high-resolution complete PNG capture
plotly_config = {
    'toImageButtonOptions': {
        'format': 'png',                           # High-quality PNG raster format
        'filename': 'horikx_plot_highres_complete',# Default downloaded file name
        'height': 800,                             # Explicit canvas height
        'width': 1200,                             # Explicit canvas width
        'scale': 2                                 # 2x scale for crisp 2400x1600 px export
    },
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'responsive': True
}

# Render Plotly interactive chart with high-res download configuration
st.plotly_chart(fig, use_container_width=True, config=plotly_config)

# ---------------------------------------------------------
# High-Resolution Publication-Quality Figure Export Section
# ---------------------------------------------------------
st.markdown("#### 💾 High-Resolution Figure Export (Truncation-Free)")

exp_col1, exp_col2, exp_col3 = st.columns(3)

# 1. Matplotlib Publication PNG Export (with bbox_inches='tight', pad_inches=0.3, dpi=300)
with exp_col1:
    def create_matplotlib_publication_figure():
        mpl_fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=300)
        
        # Render theoretical curves
        for idx_s0, s0_val in enumerate(selected_s0_curves):
            x_c, s_c, x_m, s_m = calculate_horikx_curves(s0_val)
            c_col = cl_colors[idx_s0 % len(cl_colors)]
            m_col = mc_colors[idx_s0 % len(mc_colors)]
            
            if show_mainchain_curves:
                ax.plot(x_m, s_m, color=m_col, linestyle='--', linewidth=2.2, label=f'Main-Chain (S₀={s0_val:.4f})')
            if show_crosslink_curves:
                ax.plot(x_c, s_c, color=c_col, linestyle='-', linewidth=2.4, label=f'Selective CL (S₀={s0_val:.4f})')
            if show_devulc_fill and show_crosslink_curves and show_mainchain_curves:
                s_c_int = np.interp(x_m, x_c, s_c, left=s0_val, right=1.0)
                ax.fill_between(x_m, s_m, s_c_int, color=c_col, alpha=0.08, label=f'Devulc Zone (S₀={s0_val:.4f})' if len(selected_s0_curves) <= 2 else None)
            if show_s0_baselines:
                ax.axhline(s0_val, color=c_col, linestyle=':', linewidth=1.0, alpha=0.6)
                
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], color='#cbd5e1', linestyle=':', linewidth=1)
        
        # Experimental Points
        if not evaluated_df.empty:
            pts_to_plot = evaluated_df[evaluated_df["Sample Name"].isin(visible_trials)]
            for _, r in pts_to_plot.iterrows():
                sel = r["Crosslink Scission (%)"]
                p_col = "#16A34A" if sel >= 75 else ("#2563EB" if sel >= 50 else ("#D97706" if sel >= 30 else "#DC2626"))
                ax.scatter(r["1 - v/v0"], r["Sol Fraction (s)"], color=p_col, s=85, edgecolors='white', linewidth=1.2, zorder=6, label=f"{r['Sample Name']} ({r['Initial Sol Fraction (s0)']:.3f})")
                ax.annotate(f" {r['Sample Name']}", (r["1 - v/v0"], r["Sol Fraction (s)"]), fontsize=8.5, xytext=(4, 4), textcoords='offset points')
                
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"Relative Decrease in Crosslink Density $\left(1 - \frac{\nu_f}{\nu_i}\right)$", fontsize=11, fontweight="semibold")
        ax.set_ylabel(r"Sol Fraction ($S_f$)", fontsize=11, fontweight="semibold")
        ax.set_title("Horikx Plot Analysis for Rubber Devulcanization", fontsize=13, fontweight="bold", pad=14)
        ax.grid(True, linestyle=":", alpha=0.5, color="gray")
        
        # Move legend strictly to bottom-center with multi-column alignment
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=3,
            frameon=True,
            fontsize=8.5,
            borderaxespad=0.0
        )
        
        # Wrap figure rendering with tight_layout and adjust bottom bounds to prevent legend cutoff
        plt.tight_layout()
        mpl_fig.subplots_adjust(bottom=0.28, top=0.92, left=0.10, right=0.95)
        return mpl_fig

    try:
        mpl_figure = create_matplotlib_publication_figure()
        buf_mpl = io.BytesIO()
        # Explicitly force matplotlib to include the bottom-centered legend and outer margins in the saved PNG image buffer
        mpl_figure.savefig(buf_mpl, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
        buf_mpl.seek(0)
        plt.close(mpl_figure)
        
        st.download_button(
            label="🖼️ Download Matplotlib PNG (300 DPI)",
            data=buf_mpl,
            file_name="horikx_plot_publication_300dpi.png",
            mime="image/png",
            help="High-resolution publication figure with bottom-centered legend, explicit bottom margin (0.28), and pad_inches=0.3 ensuring zero truncation."
        )
    except Exception as e:
        st.warning(f"Matplotlib export note: {e}")

# 2. Plotly High-Resolution PNG Export (width=1200, height=800, scale=2)
with exp_col2:
    try:
        # Export via Plotly to_image with explicit dimensions and 2x scale (2400x1600 px)
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        st.download_button(
            label="📊 Download Plotly PNG (1200x800, 2x)",
            data=img_bytes,
            file_name="horikx_plotly_highres_complete.png",
            mime="image/png",
            help="High-resolution Plotly vector rasterization (2400x1600 px, scale=2) with complete margins and automargin enabled."
        )
    except Exception:
        # Fallback if kaleido is not installed
        st.download_button(
            label="📊 Download Interactive HTML",
            data=fig.to_html(include_plotlyjs="cdn"),
            file_name="horikx_plotly_interactive.html",
            mime="text/html",
            help="Download interactive HTML containing Plotly chart and toggles."
        )

# 3. Interactive HTML Export
with exp_col3:
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode('utf-8')
    st.download_button(
        label="🌐 Download Standalone HTML Plot",
        data=html_bytes,
        file_name="horikx_plot_standalone.html",
        mime="text/html",
        help="Interactive HTML file viewable in any web browser without Python."
    )

# ---------------------------------------------------------
# 7. Automated Mechanism Classification & Individual S₀ Distance Analysis
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Automated Mechanism Classification & Individual S₀ Distance Analysis")

if not evaluated_df.empty:
    # Filter evaluated dataset based on visible trials if desired
    analysis_df = evaluated_df[evaluated_df["Sample Name"].isin(visible_trials)] if visible_trials else evaluated_df
    
    if not analysis_df.empty:
        # Top KPI Metrics
        avg_selectivity = analysis_df["Crosslink Scission (%)"].mean()
        best_row = analysis_df.loc[analysis_df["Crosslink Scission (%)"].idxmax()]
        worst_row = analysis_df.loc[analysis_df["Crosslink Scission (%)"].idxmin()]
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Visible Samples", f"{len(analysis_df)}")
        with kpi2:
            st.metric("Avg Crosslink Selectivity", f"{avg_selectivity:.1f}%")
        with kpi3:
            st.metric("Most Ideal Trial", str(best_row["Sample Name"]), f"{best_row['Crosslink Scission (%)']:.1f}% CL")
        with kpi4:
            ideal_count = sum(analysis_df["Crosslink Scission (%)"] >= 60.0)
            st.metric("High-Selectivity Trials", f"{ideal_count} / {len(analysis_df)}")

        # 1. Summary Table & Distance Breakdown with Explicit S_0 column
        st.markdown("#### 1. Summary Table & Distance Breakdown (Per-Sample S₀)")
        
        display_table = pd.DataFrame({
            "Sample Name": analysis_df["Sample Name"],
            "Initial Sol (S₀)": analysis_df["Initial Sol Fraction (s0)"].apply(lambda v: f"{v:.4f}"),
            "1 - ν/ν₀": analysis_df["1 - v/v0"].apply(lambda v: f"{v:.4f}" if "Fraction" in unit_format else f"{v*100:.1f}%"),
            "Sol Fraction (s)": analysis_df["Sol Fraction (s)"].apply(lambda v: f"{v:.4f}" if "Fraction" in unit_format else f"{v*100:.1f}%"),
            "Distance to CL Curve (Δx)": analysis_df["Dist to Crosslink"].apply(lambda v: f"{v:.4f}"),
            "Distance to MC Curve (Δx)": analysis_df["Dist to Main-Chain"].apply(lambda v: f"{v:.4f}"),
            "Crosslink Scission Ratio": analysis_df["Crosslink Scission (%)"].apply(lambda v: f"{v:.1f}%"),
            "Chain Scission Ratio": analysis_df["Chain Scission (%)"].apply(lambda v: f"{v:.1f}%"),
            "Classified Mechanism": analysis_df["Classification"],
            "Evaluation Rating": analysis_df["Rating"]
        })
        
        st.dataframe(display_table, use_container_width=True)

        # 2. Automated Insights & Recommendations
        st.markdown("#### 2. Automated Engineering Insights & Recommendations")
        
        with st.container():
            st.info(
                f"🎯 **Most Ideal Devulcanization Trial:** **{best_row['Sample Name']}** (evaluated with baseline $S_0={best_row['Initial Sol Fraction (s0)']:.4f}$) demonstrated the highest selectivity index (**{best_row['Crosslink Scission (%)']:.1f}%** crosslink scission vs **{best_row['Chain Scission (%)']:.1f}%** main-chain scission). "
                f"It achieved a **{best_row['1 - v/v0']*100:.1f}%** reduction in crosslink density while maintaining sol fraction at **{best_row['Sol Fraction (s)']*100:.1f}%**, minimizing rubber backbone degradation."
            )

            st.markdown("**Key Takeaways from Multi-Trial Horikx Trajectory:**")
            
            insights_bullets = []
            
            # Check unique s0 diversity
            if len(dataset_unique_s0) > 1:
                s0_str = ", ".join(f"{s:.4f}" for s in dataset_unique_s0)
                insights_bullets.append(
                    f"🔬 **Multi-Polymer / Multi-Feedstock Dataset Active:** Evaluated across {len(dataset_unique_s0)} unique initial sol baselines ($S_0 = {s0_str}$). Each sample's theoretical crosslink and main-chain limits were calculated dynamically against its respective virgin rubber network baseline."
                )

            # Check high selectivity points
            high_sel = analysis_df[analysis_df["Crosslink Scission (%)"] >= 75.0]
            if not high_sel.empty:
                names = ", ".join(f"*{n}*" for n in high_sel["Sample Name"])
                insights_bullets.append(
                    f"✅ **Selective Devulcanization Confirmed:** Samples ({names}) align closely with their respective lower Horikx crosslink curves (Δx < 0.08). Mono/di/polysulfidic crosslink bonds were preferentially cleaved without significant polymer backbone destruction."
                )
                
            # Check severe degradation points
            degraded = analysis_df[analysis_df["Crosslink Scission (%)"] < 35.0]
            if not degraded.empty:
                names = ", ".join(f"*{n}*" for n in degraded["Sample Name"])
                insights_bullets.append(
                    f"⚠️ **Severe Main-Chain Degradation Detected:** Samples ({names}) lie near the upper dashed degradation boundary. The processing conditions (excessive temperature, prolonged residence time, or extreme shear) led to uncontrolled cleavage of carbon-carbon backbones."
                )
                
            # Check intermediate / mixed points
            mixed = analysis_df[(analysis_df["Crosslink Scission (%)"] >= 35.0) & (analysis_df["Crosslink Scission (%)"] < 75.0)]
            if not mixed.empty:
                names = ", ".join(f"*{n}*" for n in mixed["Sample Name"])
                insights_bullets.append(
                    f"🔄 **Mixed Cleavage Regime:** Samples ({names}) fall inside the intermediate devulcanization corridor. Both crosslink scission and mild thermal-oxidative chain scission occurred simultaneously."
                )
                
            # Optimization guidance
            if worst_row["Crosslink Scission (%)"] < 50.0:
                insights_bullets.append(
                    f"💡 **Process Tuning Recommendation:** To shift lower-performing trials closer to ideal devulcanization, consider lowering devulcanization temperature by 15–30°C, increasing devulcanizing agent (e.g. diphenyl disulfide / DBD) concentration, or decreasing extruder barrel residence time."
                )
            else:
                insights_bullets.append(
                    "💡 **Process Quality:** Overall devulcanization quality is favorable across the test matrix with dominant crosslink network breakdown."
                )

            for b in insights_bullets:
                st.markdown(f"- {b}")

else:
    st.warning("No valid experimental points found in the table. Please add at least one row in the editor above.")
