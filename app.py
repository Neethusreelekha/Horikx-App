import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io

# Set Page Config
st.set_page_config(page_title="Horikx Plot Analysis & Scission Mechanism Analyzer", layout="wide")

st.title("🧪 Horikx Plot Analysis")
st.subtitle("Rubber Devulcanization & Scission Mechanism Analyzer")

# --- HORIKX THEORETICAL CALCULATIONS ---
def theoretical_horikx(s0, num_points=200):
    """
    Calculates theoretical curves for Horikx diagram based on initial sol fraction s0.
    """
    v_ratio = np.linspace(0.001, 0.999, num_points) # v/v0 ratio
    
    # 1. Main-Chain Scission Curve
    # Horikx equation for main-chain scission: 1 - v/v0 = 1 - (1 - s0^0.5)^2 / (1 - s^0.5)^2
    # Solving for s:
    s_mc = (1 - (1 - np.sqrt(s0)) / np.sqrt(v_ratio))**2
    s_mc = np.where(s_mc < s0, s0, s_mc)
    s_mc = np.where(s_mc > 1.0, 1.0, s_mc)
    
    # 2. Selective Crosslink Scission Curve
    # Horikx equation for crosslink scission: 1 - v/v0 = 1 - [(1 - s0^0.5)^2 * (1 - s^0.5)] / ...
    # Standard simplified form:
    s_cl = s0 + (1 - s0) * (1 - v_ratio)**3 # Approximate representation for crosslink scission
    s_cl = np.where(s_cl < s0, s0, s_cl)
    s_cl = np.where(s_cl > 1.0, 1.0, s_cl)
    
    x_val = 1 - v_ratio # Relative decrease in crosslink density (1 - v/v0)
    
    return x_val, s_mc, s_cl

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Parameters & Inputs")
s0_input = st.sidebar.number_input("Initial Sol Fraction (s₀)", min_value=0.001, max_value=0.200, value=0.016, step=0.001, format="%.4f")

uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV File", type=["csv", "xlsx"])

# Default Data
default_data = pd.DataFrame({
    "Sample Name": ["Trial 01", "trial02", "trial 04"],
    "Initial Sol (s0)": [s0_input, s0_input, s0_input],
    "1 - v/v0 (X)": [0.5000, 0.6000, 0.8000],
    "Sol (s) (Y)": [0.1200, 0.1330, 0.4560]
})

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        df = default_data
else:
    df = default_data

# Map columns dynamically
st.subheader("📊 Active Data Points")
edited_df = st.data_editor(df, num_rows="dynamic")

# Data preparation
x_exp = edited_df["1 - v/v0 (X)"].values
y_exp = edited_df["Sol (s) (Y)"].values
sample_names = edited_df["Sample Name"].values

# Calculate theoretical curves
x_theo, s_mc, s_cl = theoretical_horikx(s0_input)

# --- PLOTLY INTERACTIVE PLOT ---
fig = go.Figure()

# Main-chain scission curve (Dashed Red)
fig.add_trace(go.Scatter(
    x=x_theo, y=s_mc,
    mode='lines',
    name=f'Main-chain (s₀={s0_input})',
    line=dict(color='crimson', width=3, dash='dash')
))

# Selective Crosslink scission curve (Solid Green)
fig.add_trace(go.Scatter(
    x=x_theo, y=s_cl,
    mode='lines',
    name=f'Selective Crosslink (s₀={s0_input})',
    line=dict(color='forestgreen', width=3)
))

# Experimental Points (Blue Dots)
fig.add_trace(go.Scatter(
    x=x_exp, y=y_exp,
    mode='markers+text',
    name=f'Experimental ({len(x_exp)} points)',
    text=sample_names,
    textposition="top center",
    marker=dict(color='royalblue', size=10, symbol='circle')
))

# Layout formatting with padded margins to prevent image export truncation
fig.update_layout(
    title="Horikx Diagram: Rel. Decrease in Crosslink Density vs Sol Fraction",
    xaxis_title="Relative Decrease in Crosslink Density (1 - v/v₀)",
    yaxis_title="Soluble Fraction (s)",
    xaxis=dict(range=[0, 1], gridcolor='lightgray'),
    yaxis=dict(range=[0, 1], gridcolor='lightgray'),
    margin=dict(l=80, r=80, t=80, b=80),
    legend=dict(x=0.02, y=0.98, bordercolor="Black", borderwidth=1),
    template="plotly_white",
    width=1000,
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# --- QUANTITATIVE ANALYSIS & MECHANISM CLASSIFICATION ---
results = []
for i in range(len(x_exp)):
    # Find closest distance to theoretical curves
    idx = np.argmin(np.abs(x_theo - x_exp[i]))
    dist_mc = abs(y_exp[i] - s_mc[idx])
    dist_cl = abs(y_exp[i] - s_cl[idx])
    
    # Calculate Crosslink Scission %
    total_dist = dist_mc + dist_cl + 1e-6
    cl_percentage = round((1 - (dist_cl / total_dist)) * 100, 1)
    
    if y_exp[i] >= s_mc[idx]:
        mechanism = "Severe Main-Chain Degradation"
        rating = "Severe (1/5)"
    elif cl_percentage > 50:
        mechanism = "Predominantly Crosslink Scission"
        rating = "Good (4/5)"
    else:
        mechanism = "Predominantly Main-Chain Scission"
        rating = "Degraded (2/5)"
        
    results.append({
        "Sample Name": sample_names[i],
        "s0": s0_input,
        "1 - v/v0": x_exp[i],
        "Sol (s)": y_exp[i],
        "Distance to CL Scission (Δx_CL)": round(dist_cl, 3),
        "Distance to MC Scission (Δx_MC)": round(dist_mc, 3),
        "CL Scission %": f"{cl_percentage}%",
        "Classified Mechanism": mechanism,
        "Rating": rating
    })

res_df = pd.DataFrame(results)

st.markdown("### 1. Quantitative Distance Analysis & Mechanism Classification Table")
st.dataframe(res_df, use_container_width=True)

# --- AUTOMATED ENGINEERING INSIGHTS ---
st.markdown("### 2. Automated Engineering Insights & Recommendations")
best_sample = res_df.sort_values(by="Distance to CL Scission (Δx_CL)").iloc[0]

st.info(f"""
**Optimal Devulcanization Trial: {best_sample['Sample Name']}**  
Achieved high selectivity index with {best_sample['CL Scission %']} crosslink scission at {best_sample['1 - v/v0']} crosslink decrease.

- **Degradation Warning:** Samples lying near or above the upper main-chain curve indicate severe carbon-carbon chain scission due to excessive processing temperature or residence time.
- **Process Optimization Guidance:** To enhance selective crosslink breakdown, consider optimizing reaction temperature (reducing by 15-25°C) or adjusting shear/TMTD chemical concentrations.
""")
