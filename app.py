"""
Horikx Plot Analysis for Rubber Devulcanization
A complete Streamlit web application with:
  - Dual Dataset Input via Radio Selection:
      Option 1: Upload Excel/CSV File (with st.file_uploader & column mapping)
      Option 2: Manual Data Entry (Interactive spreadsheet with st.data_editor)
  - Full mathematical Horikx theoretical curve engine (1956)
  - High-resolution Matplotlib visualization with customizable styling
  - Automated quantitative mechanism classification & distance analysis
  - Dynamic automated insights and optimization takeaways

Requirements:
    pip install streamlit pandas numpy matplotlib openpyxl

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
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
    and crosslink density decrease ($1 - \\nu_f/\\nu_i$) against theoretical **Horikx (1956)** scission curves.
    """
)

# ---------------------------------------------------------
# 2. Sidebar Parameters & Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Model Parameters")

s0 = st.sidebar.number_input(
    "Initial Sol Fraction ($S_i$ or $s_0$)",
    min_value=0.0001,
    max_value=0.3000,
    value=0.0200,
    step=0.0050,
    format="%.4f",
    help="Sol fraction of the virgin cured rubber vulcanizate prior to devulcanization (typically 0.01 - 0.05)."
)

# Rubber presets
st.sidebar.caption("💡 Typical $S_i$ Presets:")
preset_cols = st.sidebar.columns(4)
if preset_cols[0].button("NR: 0.015"):
    s0 = 0.015
if preset_cols[1].button("EPDM: 0.025"):
    s0 = 0.025
if preset_cols[2].button("SBR: 0.035"):
    s0 = 0.035
if preset_cols[3].button("Reclaim: 0.050"):
    s0 = 0.050

unit_format = st.sidebar.radio(
    "Data Scale in Table",
    options=["Fraction (0.0 to 1.0)", "Percentage (0% to 100%)"],
    index=0,
    help="Choose whether your experimental values are entered as fractions (0.0 - 1.0) or percentages (0% - 100%)."
)

st.sidebar.subheader("🎨 Visualization Styling")
crosslink_color = st.sidebar.color_picker("Selective Crosslink Scission Curve", "#16A34A")
mainchain_color = st.sidebar.color_picker("Main-Chain Degradation Curve", "#DC2626")
point_color = st.sidebar.color_picker("Experimental Data Points", "#2563EB")
point_size = st.sidebar.slider("Scatter Point Size", min_value=30, max_value=200, value=75, step=5)
show_grid = st.sidebar.checkbox("Show Gridlines", value=True)
show_fill = st.sidebar.checkbox("Highlight Devulcanization Zone", value=True)
dpi_export = st.sidebar.selectbox("Figure Export Resolution (DPI)", options=[150, 300, 600], index=1)

with st.sidebar.expander("📖 Horikx Theoretical Equations", expanded=False):
    st.markdown("**1. Main-Chain Degradation (Upper Dashed):**")
    st.latex(r"1 - \frac{\nu_f}{\nu_i} = 1 - \frac{\left(1 - \sqrt{S_f}\right)^2}{\left(1 - \sqrt{S_i}\right)^2}")
    st.markdown("**2. Selective Crosslink Cleavage (Lower Solid):**")
    st.latex(r"1 - \frac{\nu_f}{\nu_i} = 1 - \frac{\gamma_f \left(1 - \sqrt{S_f}\right)^2}{\gamma_i \left(1 - \sqrt{S_i}\right)^2}")
    st.latex(r"\frac{\gamma_f}{\gamma_i} = \frac{S_i + \sqrt{S_i}}{S_f + \sqrt{S_f}}")

# ---------------------------------------------------------
# 3. Calculation Engine
# ---------------------------------------------------------
def calculate_horikx_curves(s0_val, n_points=300):
    """
    Computes theoretical Horikx curve coordinates (Horikx, 1956).
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

def evaluate_data_points(df_points, s0_val, scale_factor=1.0):
    """
    Quantitatively analyzes each experimental point against both theoretical curves.
    """
    s0_val = float(max(0.0001, min(0.3, s0_val)))
    denom_mc = (1.0 - np.sqrt(s0_val)) ** 2
    results = []
    
    for idx, row in df_points.iterrows():
        sample_name = str(row.get("Sample Name", f"Sample {idx+1}"))
        
        # Extract Crosslink Density Decrease
        raw_x = None
        for col_name in ["Crosslink Density Decrease (1 - v/v0)", "Crosslink Decrease (1 - v/v0)", "1 - v/v0", "Crosslink Density Decrease"]:
            if col_name in row and pd.notna(row[col_name]):
                raw_x = row[col_name]
                break
        if raw_x is None:
            raw_x = row.iloc[1] if len(row) > 1 else 0.0

        # Extract Sol Fraction
        raw_s = None
        for col_name in ["Sol Fraction (s)", "Sol Fraction (Sf)", "Sol Fraction", "s", "Sf"]:
            if col_name in row and pd.notna(row[col_name]):
                raw_s = row[col_name]
                break
        if raw_s is None:
            raw_s = row.iloc[2] if len(row) > 2 else 0.0

        try:
            val_x = float(str(raw_x).replace("%", "").strip())
            val_s = float(str(raw_s).replace("%", "").strip())
        except (ValueError, TypeError):
            continue
            
        x_val = val_x / scale_factor
        s_val = val_s / scale_factor
        
        x_val = float(np.clip(x_val, 0.0, 1.0))
        s_val = float(np.clip(s_val, s0_val, 1.0))
        
        # Theoretical x on Main-Chain curve at this s_val
        num_s = (1.0 - np.sqrt(s_val)) ** 2
        x_mc_th = float(np.clip(1.0 - (num_s / denom_mc), 0.0, 1.0))
        
        # Theoretical x on Crosslink curve at this s_val
        gamma_r = (s0_val + np.sqrt(s0_val)) / (s_val + np.sqrt(s_val))
        x_cl_th = float(np.clip(1.0 - (gamma_r * (num_s / denom_mc)), 0.0, 1.0))
        
        # Distances to theoretical curves
        dist_cl = abs(x_val - x_cl_th)
        dist_mc = abs(x_val - x_mc_th)
        
        # Selectivity Calculation (Crosslink Scission Fraction)
        if x_cl_th > x_mc_th:
            selectivity = float(np.clip(((x_val - x_mc_th) / (x_cl_th - x_mc_th)) * 100.0, 0.0, 100.0))
        else:
            selectivity = 50.0
            
        chain_scission = 100.0 - selectivity
        
        # Qualitative Mechanism Classification
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
# 4. Dataset Input Options (Radio: Upload vs Manual Entry)
# ---------------------------------------------------------
st.subheader("📋 Dataset Input")

# Initialize default experimental dataset in session state
if "manual_data" not in st.session_state:
    st.session_state["manual_data"] = pd.DataFrame([
        {"Sample Name": "Trial 1 (180°C / 5 min)", "Crosslink Density Decrease (1 - v/v0)": 0.35, "Sol Fraction (s)": 0.06, "Notes": "Mild thermal devulcanization"},
        {"Sample Name": "Trial 2 (200°C / 5 min)", "Crosslink Density Decrease (1 - v/v0)": 0.58, "Sol Fraction (s)": 0.12, "Notes": "Optimal balance"},
        {"Sample Name": "Trial 3 (220°C / 5 min)", "Crosslink Density Decrease (1 - v/v0)": 0.74, "Sol Fraction (s)": 0.22, "Notes": "Selective crosslink cleavage"},
        {"Sample Name": "Trial 4 (240°C / 5 min)", "Crosslink Density Decrease (1 - v/v0)": 0.88, "Sol Fraction (s)": 0.42, "Notes": "Onset of chain scission"},
        {"Sample Name": "Trial 5 (260°C / 5 min)", "Crosslink Density Decrease (1 - v/v0)": 0.95, "Sol Fraction (s)": 0.68, "Notes": "Severe thermal degradation"},
    ])

# Radio button to select dataset input method
input_option = st.radio(
    "Choose Dataset Input Option:",
    options=["Option 1: Upload Excel/CSV File", "Option 2: Manual Data Entry"],
    index=1,
    horizontal=True,
    help="Select Option 1 to upload an existing spreadsheet, or Option 2 to type/edit data points directly in the interactive table."
)

current_active_df = None

if input_option == "Option 1: Upload Excel/CSV File":
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Spreadsheet File (.xlsx, .xls, .csv)",
            type=["csv", "xlsx", "xls"],
            help="Upload an Excel or CSV file containing Crosslink Density Decrease and Sol Fraction columns."
        )
    with col_up2:
        st.write("")
        st.write("")
        sample_csv = st.session_state["manual_data"].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV Template",
            data=sample_csv,
            file_name="horikx_data_template.csv",
            mime="text/csv"
        )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                imported_df = pd.read_excel(uploaded_file)
            else:
                imported_df = pd.read_csv(uploaded_file)
                
            st.success(f"Loaded **{uploaded_file.name}** ({len(imported_df)} rows)")
            
            # Smart column mapping assistant
            cols = list(imported_df.columns)
            def find_match(keys, default_idx=0):
                for i, c in enumerate(cols):
                    clean = str(c).lower().replace(" ", "").replace("_", "").replace("-", "")
                    if any(k in clean for k in keys):
                        return i
                return min(default_idx, len(cols) - 1)
                
            m1, m2, m3 = st.columns(3)
            with m1:
                x_match = st.selectbox("Map Crosslink Density Decrease Column (X)", cols, index=find_match(["crosslink", "decrease", "1v", "xd", "loss", "density"], 1))
            with m2:
                s_match = st.selectbox("Map Sol Fraction Column (Y)", cols, index=find_match(["sol", "fraction", "soluble", "s"], 2))
            with m3:
                name_match = st.selectbox("Map Sample Name Column (Optional)", ["Auto Index"] + cols, index=find_match(["samplename", "sample", "name", "id"], 0) + 1 if "Sample Name" in cols else 0)
                
            mapped_df = pd.DataFrame()
            mapped_df["Sample Name"] = imported_df[name_match] if name_match != "Auto Index" else [f"Sample {i+1}" for i in range(len(imported_df))]
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
        "The Horikx plot, curve distances, and automated insights will update dynamically."
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

# ---------------------------------------------------------
# 5. Process & Calculate Analysis
# ---------------------------------------------------------
scale = 100.0 if "Percentage" in unit_format else 1.0
evaluated_df = evaluate_data_points(current_active_df, s0, scale_factor=scale)

# ---------------------------------------------------------
# 6. Matplotlib Horikx Diagram
# ---------------------------------------------------------
st.subheader("📊 Horikx Diagram")

x_cl, s_cl, x_mc, s_mc = calculate_horikx_curves(s0)

fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=dpi_export)

# Upper Dashed Line: Main-Chain Degradation
ax.plot(
    x_mc, s_mc,
    color=mainchain_color,
    linestyle='--',
    linewidth=2.2,
    label=f'Main-chain Degradation (Upper dashed line, $S_i={s0:.4f}$)'
)

# Lower Solid Line: Selective Crosslink Cleavage
ax.plot(
    x_cl, s_cl,
    color=crosslink_color,
    linestyle='-',
    linewidth=2.2,
    label=f'Selective Crosslink Cleavage (Lower line, $S_i={s0:.4f}$)'
)

# Shaded Selective Devulcanization Zone
if show_fill:
    s_cl_interp = np.interp(x_mc, x_cl, s_cl, left=s0, right=1.0)
    ax.fill_between(
        x_mc, s_mc, s_cl_interp,
        color=crosslink_color,
        alpha=0.10,
        label='Selective Devulcanization Zone'
    )

# Baseline s0 Reference line
ax.axhline(s0, color='#6366F1', linestyle=':', linewidth=1.2, alpha=0.7, label=f'Initial Sol Baseline ($S_i={s0:.4f}$)')

# Scatter Points
if not evaluated_df.empty:
    ax.scatter(
        evaluated_df["1 - v/v0"],
        evaluated_df["Sol Fraction (s)"],
        color=point_color,
        s=point_size,
        edgecolors='black',
        linewidth=0.9,
        zorder=6,
        label=f'Experimental Points ({len(evaluated_df)})'
    )
    
    # Point Annotations
    for _, row in evaluated_df.iterrows():
        ax.annotate(
            str(row["Sample Name"]),
            (row["1 - v/v0"], row["Sol Fraction (s)"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
            weight='500',
            alpha=0.85
        )

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel(r"Relative Decrease in Crosslink Density $\left(1 - \frac{\nu_f}{\nu_i}\right)$", fontsize=11, fontweight='semibold')
ax.set_ylabel(r"Sol Fraction $\left(S_f\right)$", fontsize=11, fontweight='semibold')
ax.set_title("Horikx Plot Analysis for Rubber Devulcanization", fontsize=13, fontweight='bold', pad=14)

if show_grid:
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')

ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False, fontsize=9.5)
plt.tight_layout()

# Render Matplotlib Figure
st.pyplot(fig)

# Figure Download Button
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=dpi_export, bbox_inches="tight")
buf.seek(0)
st.download_button(
    label=f"💾 Download Horikx Plot as PNG ({dpi_export} DPI)",
    data=buf,
    file_name=f"horikx_plot_s0_{s0:.4f}.png",
    mime="image/png"
)

# ---------------------------------------------------------
# 7. Automated Quantitative Mechanism Analysis & Insights
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔬 Automated Mechanism Classification & Distance Analysis")

if not evaluated_df.empty:
    # Top KPI Metrics
    avg_selectivity = evaluated_df["Crosslink Scission (%)"].mean()
    best_row = evaluated_df.loc[evaluated_df["Crosslink Scission (%)"].idxmax()]
    worst_row = evaluated_df.loc[evaluated_df["Crosslink Scission (%)"].idxmin()]
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Samples", f"{len(evaluated_df)}")
    with kpi2:
        st.metric("Average Crosslink Selectivity", f"{avg_selectivity:.1f}%")
    with kpi3:
        st.metric("Most Ideal Trial", str(best_row["Sample Name"]), f"{best_row['Crosslink Scission (%)']:.1f}% CL")
    with kpi4:
        ideal_count = sum(evaluated_df["Crosslink Scission (%)"] >= 60.0)
        st.metric("High-Selectivity Trials", f"{ideal_count} / {len(evaluated_df)}")

    # 1. Summary Table & Metrics
    st.markdown("#### 1. Summary Table & Distance Breakdown")
    
    display_table = pd.DataFrame({
        "Sample Name": evaluated_df["Sample Name"],
        "1 - ν/ν₀": evaluated_df["1 - v/v0"].apply(lambda v: f"{v:.4f}" if "Fraction" in unit_format else f"{v*100:.1f}%"),
        "Sol Fraction (s)": evaluated_df["Sol Fraction (s)"].apply(lambda v: f"{v:.4f}" if "Fraction" in unit_format else f"{v*100:.1f}%"),
        "Distance to CL Curve (Δx)": evaluated_df["Dist to Crosslink"].apply(lambda v: f"{v:.4f}"),
        "Distance to MC Curve (Δx)": evaluated_df["Dist to Main-Chain"].apply(lambda v: f"{v:.4f}"),
        "Crosslink Scission Ratio": evaluated_df["Crosslink Scission (%)"].apply(lambda v: f"{v:.1f}%"),
        "Chain Scission Ratio": evaluated_df["Chain Scission (%)"].apply(lambda v: f"{v:.1f}%"),
        "Classified Mechanism": evaluated_df["Classification"],
        "Evaluation Rating": evaluated_df["Rating"]
    })
    
    st.dataframe(display_table, use_container_width=True)

    # 2. Automated Insights & Recommendations
    st.markdown("#### 2. Automated Engineering Insights & Recommendations")
    
    with st.container():
        st.info(
            f"🎯 **Most Ideal Devulcanization Trial:** **{best_row['Sample Name']}** demonstrated the highest selectivity index (**{best_row['Crosslink Scission (%)']:.1f}%** crosslink scission vs **{best_row['Chain Scission (%)']:.1f}%** main-chain scission). "
            f"It achieved a **{best_row['1 - v/v0']*100:.1f}%** reduction in crosslink density while maintaining sol fraction at **{best_row['Sol Fraction (s)']*100:.1f}%**, minimizing molecular weight loss of the rubber matrix."
        )

        st.markdown("**Key Takeaways from Horikx Trajectory:**")
        
        insights_bullets = []
        
        # Check high selectivity points
        high_sel = evaluated_df[evaluated_df["Crosslink Scission (%)"] >= 75.0]
        if not high_sel.empty:
            names = ", ".join(f"*{n}*" for n in high_sel["Sample Name"])
            insights_bullets.append(
                f"✅ **Selective Devulcanization Confirmed:** Samples ({names}) align closely with the lower Horikx curve (Δx < 0.08). Mono/di/polysulfidic crosslink bonds were preferentially cleaved without significant polymer backbone destruction."
            )
            
        # Check severe degradation points
        degraded = evaluated_df[evaluated_df["Crosslink Scission (%)"] < 35.0]
        if not degraded.empty:
            names = ", ".join(f"*{n}*" for n in degraded["Sample Name"])
            insights_bullets.append(
                f"⚠️ **Severe Main-Chain Degradation Detected:** Samples ({names}) lie near the upper dashed line. The processing conditions (excessive temperature, prolonged residence time, or extreme shear) led to uncontrolled cleavage of carbon-carbon backbones."
            )
            
        # Check intermediate / mixed points
        mixed = evaluated_df[(evaluated_df["Crosslink Scission (%)"] >= 35.0) & (evaluated_df["Crosslink Scission (%)"] < 75.0)]
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
