"""
Horikx Plot Analysis for Rubber Devulcanization
Author: Rubber Devulcanization & Polymer Degradation Analytics
Reference: M. M. Horikx, J. Polym. Sci., 1956, 19, 445-454.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ---------------------------------------------------------
# 1. Page Configuration & Title
# ---------------------------------------------------------
st.set_page_config(
    page_title="Horikx Plot Analysis",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧪 Horikx Plot Analysis for Rubber Devulcanization")
st.markdown(
    """
    This tool evaluates devulcanization efficiency by comparing experimental sol fraction 
    and crosslink density reduction against the theoretical **Horikx (1956)** scission curves.
    """
)

# ---------------------------------------------------------
# 2. Sidebar Parameters & Styling Controls
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

unit_format = st.sidebar.radio(
    "Data Input Scale",
    options=["Fraction (0.0 to 1.0)", "Percentage (0% to 100%)"],
    index=0,
    help="Select whether your uploaded data table uses decimals (0.0 to 1.0) or percentages (0% to 100%)."
)

st.sidebar.subheader("🎨 Plot Styling")
crosslink_color = st.sidebar.color_picker("Selective Crosslink Curve Color", "#16A34A")
mainchain_color = st.sidebar.color_picker("Main-Chain Degradation Curve Color", "#DC2626")
point_color = st.sidebar.color_picker("Experimental Data Points Color", "#2563EB")
point_size = st.sidebar.slider("Scatter Point Size", min_value=20, max_value=180, value=70, step=5)
show_grid = st.sidebar.checkbox("Show Gridlines", value=True)
show_fill = st.sidebar.checkbox("Highlight Devulcanization Zone", value=True)
dpi_export = st.sidebar.selectbox("Export Resolution (DPI)", options=[150, 300, 600], index=1)

with st.sidebar.expander("📖 Theoretical Equations", expanded=False):
    st.markdown("**1. Main-Chain Degradation:**")
    st.latex(r"1 - \frac{\nu_f}{\nu_i} = 1 - \frac{\left(1 - \sqrt{S_f}\right)^2}{\left(1 - \sqrt{S_i}\right)^2}")
    st.markdown("**2. Selective Crosslink Cleavage:**")
    st.latex(r"1 - \frac{\nu_f}{\nu_i} = 1 - \frac{\gamma_f \left(1 - \sqrt{S_f}\right)^2}{\gamma_i \left(1 - \sqrt{S_i}\right)^2}")
    st.latex(r"\frac{\gamma_f}{\gamma_i} = \frac{S_i + \sqrt{S_i}}{S_f + \sqrt{S_f}}")

# ---------------------------------------------------------
# 3. Horikx Theoretical Calculation Engine
# ---------------------------------------------------------
def calculate_horikx_curves(s0_val, n_points=300):
    """
    Computes theoretical Horikx curve coordinates (Horikx, 1956).
    
    X-axis: Relative decrease in crosslink density = 1 - (nu_f / nu_i)
    Y-axis: Sol fraction = S_f
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
    
    # Crosslinking index gamma ratio: gamma_f / gamma_i = (s0 + sqrt(s0)) / (s + sqrt(s))
    gamma_ratio = (s0_val + np.sqrt(s0_val)) / (s_crosslink + np.sqrt(s_crosslink))
    ratio = gamma_ratio * (numerator / denom)
    x_crosslink = np.clip(1.0 - ratio, 0.0, 1.0)
    
    return x_crosslink, s_crosslink, x_mainchain, s_mainchain

# ---------------------------------------------------------
# 4. File Upload & Data Management
# ---------------------------------------------------------
st.subheader("📂 Experimental Data Upload")

# Built-in Default Dataset
sample_df = pd.DataFrame({
    "Sample Name": ["EPDM 180°C", "EPDM 200°C", "EPDM 220°C", "EPDM 240°C", "EPDM 260°C"],
    "Crosslink Density Decrease (1 - v/v0)": [0.35, 0.58, 0.74, 0.88, 0.95],
    "Sol Fraction (s)": [0.06, 0.12, 0.21, 0.40, 0.65],
    "Condition": ["180°C / 5 min", "200°C / 5 min", "220°C / 5 min", "240°C / 5 min", "260°C / 5 min"]
})

col_u1, col_u2 = st.columns([3, 1])

with col_u1:
    uploaded_file = st.file_uploader(
        "Upload Excel (.xlsx, .xls) or CSV dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a spreadsheet containing crosslink density decrease and sol fraction columns."
    )

with col_u2:
    st.write("")
    st.write("")
    csv_sample = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV Template",
        data=csv_sample,
        file_name="horikx_template.csv",
        mime="text/csv"
    )

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success(f"Loaded **{uploaded_file.name}** ({len(df)} rows)")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = sample_df.copy()
else:
    st.info("Showing default sample dataset below. Upload your own file above to analyze custom data.")
    df = sample_df.copy()

# ---------------------------------------------------------
# 5. Column Mapping & Processing
# ---------------------------------------------------------
if df is not None:
    cols = list(df.columns)
    
    def auto_find_col(keywords, default_idx=0):
        for i, c in enumerate(cols):
            clean = str(c).lower().replace(" ", "").replace("_", "").replace("-", "")
            if any(k in clean for k in keywords):
                return i
        return min(default_idx, len(cols) - 1)

    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox(
            "Crosslink Density Decrease Column (X-axis)",
            options=cols,
            index=auto_find_col(["crosslink", "decrease", "1v/v0", "1v", "xd", "loss", "density"], default_idx=1)
        )
    with c2:
        y_col = st.selectbox(
            "Sol Fraction Column (Y-axis)",
            options=cols,
            index=auto_find_col(["sol", "fraction", "soluble", "extractable", "s"], default_idx=2)
        )
    with c3:
        name_col = st.selectbox(
            "Sample Label Column (Optional)",
            options=["None"] + cols,
            index=1 if "Sample Name" in cols else 0
        )

    # Clean and parse numeric values
    clean_df = df.copy()
    clean_df[x_col] = pd.to_numeric(clean_df[x_col].astype(str).str.replace("%", "").str.strip(), errors='coerce')
    clean_df[y_col] = pd.to_numeric(clean_df[y_col].astype(str).str.replace("%", "").str.strip(), errors='coerce')
    clean_df = clean_df.dropna(subset=[x_col, y_col])

    scale = 100.0 if "Percentage" in unit_format else 1.0
    x_data = clean_df[x_col].values / scale
    y_data = clean_df[y_col].values / scale

    # ---------------------------------------------------------
    # 6. Matplotlib Horikx Plot Rendering
    # ---------------------------------------------------------
    st.subheader("📊 Interactive Horikx Diagram")

    x_cl, s_cl, x_mc, s_mc = calculate_horikx_curves(s0)

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=dpi_export)

    # Upper Dashed Line: Main-Chain Degradation
    ax.plot(
        x_mainchain:=x_mc, s_mc,
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

    # Experimental Scatter Data Points
    ax.scatter(
        x_data, y_data,
        color=point_color,
        s=point_size,
        edgecolors='black',
        linewidth=0.8,
        zorder=5,
        label=f'Experimental Data ({len(clean_df)} points)'
    )

    # Point Annotations
    if name_col != "None" and name_col in clean_df.columns:
        for _, row in clean_df.iterrows():
            lbl = str(row[name_col])
            px = row[x_col] / scale
            py = row[y_col] / scale
            ax.annotate(
                lbl,
                (px, py),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8.5,
                weight='500',
                alpha=0.85
            )

    # Axes limits, labels, and formatting
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Relative Decrease in Crosslink Density $\left(1 - \frac{\nu_f}{\nu_i}\right)$", fontsize=11, fontweight='semibold')
    ax.set_ylabel(r"Sol Fraction $\left(S_f\right)$", fontsize=11, fontweight='semibold')
    ax.set_title("Horikx Plot Analysis for Rubber Devulcanization", fontsize=13, fontweight='bold', pad=14)

    if show_grid:
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')

    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False, fontsize=9.5)
    plt.tight_layout()

    # Display Figure in Streamlit
    st.pyplot(fig)

    # Download Button for High-Resolution Plot
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi_export, bbox_inches="tight")
    buf.seek(0)
    
    st.download_button(
        label=f"💾 Download Plot as High-Res PNG ({dpi_export} DPI)",
        data=buf,
        file_name=f"horikx_plot_s0_{s0:.4f}.png",
        mime="image/png"
    )

    # ---------------------------------------------------------
    # 7. Data Table & Scission Classification Summary
    # ---------------------------------------------------------
    with st.expander("📋 Processed Data Table & Point Evaluation", expanded=True):
        # Calculate theoretical bounds for each point
        eval_records = []
        denom_mc = (1.0 - np.sqrt(s0)) ** 2
        for _, row in clean_df.iterrows():
            px = float(row[x_col]) / scale
            ps = float(row[y_col]) / scale
            
            # Theoretical Main-Chain x at this s:
            num_s = (1.0 - np.sqrt(max(s0, min(1.0, ps)))) ** 2
            x_mc_th = np.clip(1.0 - (num_s / denom_mc), 0.0, 1.0)
            
            # Theoretical Crosslink x at this s:
            gamma_r = (s0 + np.sqrt(s0)) / (ps + np.sqrt(ps))
            x_cl_th = np.clip(1.0 - (gamma_r * (num_s / denom_mc)), 0.0, 1.0)
            
            # Selectivity calculation
            if x_cl_th > x_mc_th:
                selectivity = np.clip(((px - x_mc_th) / (x_cl_th - x_mc_th)) * 100.0, 0.0, 100.0)
            else:
                selectivity = 50.0
                
            if px >= x_cl_th * 0.95:
                mech = "Selective Crosslink Cleavage"
            elif px <= x_mc_th * 1.05:
                mech = "Main-Chain Degradation"
            else:
                mech = "Mixed Scission Mechanism"
                
            eval_records.append({
                "Sample": row[name_col] if name_col != "None" else f"Point {len(eval_records)+1}",
                "1 - v_f/v_i": f"{px:.4f}" if unit_format.startswith("Fraction") else f"{px*100:.1f}%",
                "Sol Fraction (S_f)": f"{ps:.4f}" if unit_format.startswith("Fraction") else f"{ps*100:.1f}%",
                "Selectivity Index": f"{selectivity:.1f}%",
                "Predominant Mechanism": mech
            })
            
        st.dataframe(pd.DataFrame(eval_records), use_container_width=True)
