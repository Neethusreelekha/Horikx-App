import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Horikx Plot Analysis",
    page_icon="🧪",
    layout="wide"
)

st.title("Horikx Plot Analysis for Rubber Devulcanization")
st.markdown(
    "Upload experimental devulcanization data (Excel or CSV) to plot against "
    "theoretical curves for **selective crosslink scission** and **main-chain scission**."
)

# ---------------------------------------------------------
# Sidebar Controls & Parameters
# ---------------------------------------------------------
st.sidebar.header("Plot Parameters")

s0 = st.sidebar.number_input(
    "Initial Sol Fraction ($s_0$ or $s_i$)",
    min_value=0.0001,
    max_value=0.3000,
    value=${defaultS0.toFixed(3)},
    step=0.005,
    format="%.4f",
    help="Sol fraction of the original rubber vulcanizate prior to devulcanization (typically 0.01 - 0.05)."
)

unit_format = st.sidebar.radio(
    "Data Input Scale",
    options=["Fraction (0.0 to 1.0)", "Percentage (0% to 100%)"],
    index=0,
    help="Select whether your uploaded table uses decimals (0.0 - 1.0) or percentages (0 - 100)."
)

# Visual styling options
st.sidebar.subheader("Styling Options")
point_color = st.sidebar.color_picker("Data Points Color", "#1E40AF")
crosslink_color = st.sidebar.color_picker("Crosslink Scission Curve Color", "#16A34A")
mainchain_color = st.sidebar.color_picker("Main-chain Scission Curve Color", "#DC2626")
point_size = st.sidebar.slider("Scatter Point Size", min_value=20, max_value=150, value=65)
show_grid = st.sidebar.checkbox("Show Gridlines", value=True)
dpi_export = st.sidebar.selectbox("Export Figure DPI", options=[150, 300, 600], index=1)

# ---------------------------------------------------------
# Theoretical Horikx Curves Calculation
# ---------------------------------------------------------
def calculate_horikx_curves(s0_val, n_points=300):
    """
    Computes theoretical Horikx curve coordinates (Horikx, 1956).
    
    1. Main-Chain Scission:
       1 - (v_f / v_i) = 1 - [ (1 - np.sqrt(S_f))**2 / (1 - np.sqrt(S_i))**2 ]
       Inverted: S_f(x) = (1.0 - (1.0 - np.sqrt(s0_val)) * np.sqrt(1.0 - x)) ** 2
       
    2. Crosslink Scission:
       1 - (v_f / v_i) = 1 - [ (gamma_f * (1 - np.sqrt(S_f))**2) / (gamma_i * (1 - np.sqrt(S_i))**2) ]
       where gamma(S) = 1.0 / (S + np.sqrt(S))
       giving: gamma_f / gamma_i = (s0_val + np.sqrt(s0_val)) / (s_crosslink + np.sqrt(s_crosslink))
    """
    s0_val = float(max(0.0001, min(0.3, s0_val)))
    
    # 1. Main-chain scission curve (analytical s as function of x from 0 to 1)
    x_mainchain = np.linspace(0.0, 1.0, n_points)
    term = (1.0 - np.sqrt(s0_val)) * np.sqrt(np.maximum(0.0, 1.0 - x_mainchain))
    s_mainchain = (1.0 - term) ** 2
    s_mainchain = np.clip(s_mainchain, 0.0, 1.0)
    
    # 2. Crosslink scission curve (parameterized by s from s0 to 1)
    s_crosslink = np.linspace(s0_val, 1.0, n_points)
    denom = (1.0 - np.sqrt(s0_val)) ** 2
    numerator = (1.0 - np.sqrt(s_crosslink)) ** 2
    # gamma_f / gamma_i = (s0 + np.sqrt(s0)) / (s + np.sqrt(s))
    gamma_ratio = (s0_val + np.sqrt(s0_val)) / (s_crosslink + np.sqrt(s_crosslink))
    ratio = gamma_ratio * (numerator / denom)
    x_crosslink = np.clip(1.0 - ratio, 0.0, 1.0)
    
    return x_crosslink, s_crosslink, x_mainchain, s_mainchain

# ---------------------------------------------------------
# Theoretical Equations Display
# ---------------------------------------------------------
with st.sidebar.expander("📖 Theoretical Equations", expanded=False):
    st.markdown("**1. Main-Chain Degradation:**")
    st.latex(r"1 - \\frac{\\nu_f}{\\nu_i} = 1 - \\frac{\\left(1 - \\sqrt{S_f}\\right)^2}{\\left(1 - \\sqrt{S_i}\\right)^2}")
    st.markdown("**2. Crosslink Cleavage:**")
    st.latex(r"1 - \\frac{\\nu_f}{\\nu_i} = 1 - \\frac{\\gamma_f \\left(1 - \\sqrt{S_f}\\right)^2}{\\gamma_i \\left(1 - \\sqrt{S_i}\\right)^2}")
    st.latex(r"\\frac{\\gamma_f}{\\gamma_i} = \\frac{S_i + \\sqrt{S_i}}{S_f + \\sqrt{S_f}}")

# ---------------------------------------------------------
# File Upload & Data Processing
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Devulcanization Data (Excel or CSV)",
    type=["csv", "xlsx", "xls"],
    help="File should contain columns for sol fraction and crosslink density decrease (or relative decrease)."
)

# Sample template download
template_df = pd.DataFrame({
    "Sample Name": ["Sample A", "Sample B", "Sample C", "Sample D", "Sample E"],
    "Crosslink Density Decrease": [0.35, 0.55, 0.70, 0.85, 0.94],
    "Sol Fraction": [0.07, 0.14, 0.23, 0.42, 0.68],
    "Condition": ["180°C", "200°C", "220°C", "240°C", "260°C"]
})

col_t1, col_t2 = st.columns([1, 4])
with col_t1:
    csv_bytes = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download CSV Template",
        data=csv_bytes,
        file_name="horikx_sample_template.csv",
        mime="text/csv"
    )

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {uploaded_file.name} ({len(df)} rows)")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("No file uploaded yet. Showing sample experimental dataset below.")
    df = template_df.copy()

if df is not None:
    # Column mapping
    st.subheader("Data & Column Mapping")
    cols = list(df.columns)
    
    # Fuzzy match helper
    def find_col(keywords, default_idx=0):
        for i, c in enumerate(cols):
            c_low = str(c).lower()
            if any(k in c_low for k in keywords):
                return i
        return min(default_idx, len(cols) - 1)

    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox(
            "Crosslink Density Decrease Column (X-axis)",
            options=cols,
            index=find_col(["crosslink", "decrease", "1-v", "v/v0", "xd", "loss", "density"], default_idx=1)
        )
    with c2:
        y_col = st.selectbox(
            "Sol Fraction Column (Y-axis)",
            options=cols,
            index=find_col(["sol", "fraction", "soluble", "extractable", "s"], default_idx=2)
        )
    with c3:
        label_col = st.selectbox(
            "Sample Label / Name Column (Optional)",
            options=["None"] + cols,
            index=1 if "Sample Name" in cols else 0
        )

    # Convert values to numeric & scale if percentage
    clean_df = df.copy()
    clean_df[x_col] = pd.to_numeric(clean_df[x_col], errors='coerce')
    clean_df[y_col] = pd.to_numeric(clean_df[y_col], errors='coerce')
    clean_df = clean_df.dropna(subset=[x_col, y_col])

    scale_factor = 100.0 if "Percentage" in unit_format else 1.0
    x_data = clean_df[x_col].values / scale_factor
    y_data = clean_df[y_col].values / scale_factor

    # ---------------------------------------------------------
    # Generate Matplotlib Figure
    # ---------------------------------------------------------
    st.subheader("Horikx Plot")

    x_cl, s_cl, x_mc, s_mc = calculate_horikx_curves(s0)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=dpi_export)

    # Plot theoretical curves
    ax.plot(
        x_mc, s_mc,
        color=mainchain_color,
        linestyle='--',
        linewidth=2.2,
        label=f'Main-chain Degradation (Upper dashed line, $s_0 = {s0:.3f}$)'
    )
    ax.plot(
        x_cl, s_cl,
        color=crosslink_color,
        linestyle='-',
        linewidth=2.2,
        label=f'Selective Crosslink Cleavage (Lower line, $s_0 = {s0:.3f}$)'
    )

    # Optional shade between curves to highlight devulcanization region
    s_cl_interp = np.interp(x_mc, x_cl, s_cl, left=s0, right=1.0)
    ax.fill_between(
        x_mc, s_mc, s_cl_interp,
        color=crosslink_color,
        alpha=0.08,
        label='Selective Devulcanization Zone'
    )

    # Plot experimental scatter points
    ax.scatter(
        x_data, y_data,
        color=point_color,
        s=point_size,
        edgecolors='black',
        linewidth=0.8,
        zorder=5,
        label='Experimental Data'
    )

    # Add point annotations if label column is chosen
    if label_col != "None" and label_col in clean_df.columns:
        for idx, row in clean_df.iterrows():
            lbl = str(row[label_col])
            px = row[x_col] / scale_factor
            py = row[y_col] / scale_factor
            ax.annotate(
                lbl,
                (px, py),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.85
            )

    # Axes limits, labels, and styling
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Relative Decrease in Crosslink Density ($1 - \nu / \nu_0$)", fontsize=11, fontweight='semibold')
    ax.set_ylabel(r"Sol Fraction ($s$)", fontsize=11, fontweight='semibold')
    ax.set_title("Horikx Plot for Rubber Devulcanization", fontsize=13, fontweight='bold', pad=12)

    if show_grid:
        ax.grid(True, linestyle=':', alpha=0.6, color='gray')

    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False, fontsize=9.5)
    plt.tight_layout()

    # Render in Streamlit
    st.pyplot(fig)

    # Download Matplotlib plot as PNG/PDF
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi_export, bbox_inches="tight")
    buf.seek(0)
    
    st.download_button(
        label="💾 Download Plot as High-Res PNG",
        data=buf,
        file_name="horikx_plot_analysis.png",
        mime="image/png"
    )

    # ---------------------------------------------------------
    # Data Table View
    # ---------------------------------------------------------
    with st.expander("View Uploaded Data Table", expanded=False):
        st.dataframe(clean_df, use_container_width=True)
`;
