import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Carbon Footprint Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Clean Professional SaaS Styles (Noto Sans, Minimalist, No AI bloat)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Noto Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
    font-size: 14px;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Sidebar Radio */
div[data-testid="stRadio"] > div {
    gap: 4px;
}

div[data-testid="stRadio"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    font-weight: 500;
    color: #c9d1d9;
    transition: background 0.15s ease;
}

div[data-testid="stRadio"] label:hover {
    background: #21262d;
    color: #ffffff;
}

/* Page Header Card */
.page-header {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
}

.page-title {
    font-size: 20px;
    font-weight: 600;
    color: #f0f6fc;
    margin: 0 0 6px 0;
}

.page-desc {
    font-size: 13px;
    color: #8b949e;
    margin: 0;
    line-height: 1.5;
}

/* Content Container Cards */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
}

.card-title {
    font-size: 14px;
    font-weight: 600;
    color: #f0f6fc;
    margin: 0 0 12px 0;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Metric Display Cards */
.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 18px;
}

.metric-label {
    font-size: 12px;
    font-weight: 500;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: #f0f6fc;
}

.metric-meta {
    font-size: 12px;
    color: #3fb950;
    margin-top: 4px;
    font-weight: 500;
}

/* Impact Badges */
.badge-low {
    display: inline-block;
    background: rgba(46, 160, 67, 0.15);
    border: 1px solid #3fb950;
    color: #3fb950;
    padding: 4px 12px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
}

.badge-med {
    display: inline-block;
    background: rgba(210, 153, 34, 0.15);
    border: 1px solid #d29922;
    color: #e3b341;
    padding: 4px 12px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
}

.badge-high {
    display: inline-block;
    background: rgba(248, 81, 73, 0.15);
    border: 1px solid #f85149;
    color: #f85149;
    padding: 4px 12px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
}

/* Clean Button Styling */
div.stButton > button:first-child {
    background: #238636 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    border-radius: 6px !important;
    border: 1px solid rgba(240, 246, 252, 0.1) !important;
    box-shadow: 0 1px 0 rgba(27, 31, 36, 0.1) !important;
    width: 100% !important;
    transition: background-color 0.15s ease !important;
}

div.stButton > button:first-child:hover {
    background: #2ea043 !important;
    color: #ffffff !important;
}

/* Input Fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
    font-size: 13px !important;
}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
    border-color: #58a6ff !important;
}

/* Table styling */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d;
    border-radius: 6px;
    overflow: hidden;
}

/* Section divider */
.section-title {
    font-size: 16px;
    font-weight: 600;
    color: #f0f6fc;
    margin: 20px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Clean Matplotlib / Seaborn Theme
# -----------------------------------------------------------------------------
def apply_plot_theme(fig, ax):
    fig.patch.set_facecolor('#161b22')
    if isinstance(ax, (list, np.ndarray)):
        for a in np.array(ax).flat:
            _style_single_ax(a)
    else:
        _style_single_ax(ax)

def _style_single_ax(a):
    a.set_facecolor('#0d1117')
    a.tick_params(colors='#8b949e', labelsize=9)
    a.xaxis.label.set_color('#c9d1d9')
    a.yaxis.label.set_color('#c9d1d9')
    a.xaxis.label.set_fontweight('500')
    a.yaxis.label.set_fontweight('500')
    a.xaxis.label.set_fontsize(10)
    a.yaxis.label.set_fontsize(10)
    a.title.set_color('#f0f6fc')
    a.title.set_fontweight('600')
    a.title.set_fontsize(11)
    for spine in a.spines.values():
        spine.set_color('#30363d')
        spine.set_linewidth(1.0)
    a.grid(True, linestyle='--', alpha=0.3, color='#21262d')

# -----------------------------------------------------------------------------
# Data & Model Loader
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "personal_carbon_footprint_behavior.csv")
    if not os.path.exists(csv_path):
        csv_path = "personal_carbon_footprint_behavior.csv"
    return pd.read_csv(csv_path)

df = load_data()

@st.cache_resource
def load_model_files():
    def get_path(filename):
        p = os.path.join(BASE_DIR, filename)
        return p if os.path.exists(p) else filename

    model = joblib.load(get_path("best_model.pkl"))
    best_model_name = joblib.load(get_path("best_model_name.pkl"))
    scaler = joblib.load(get_path("scaler.pkl"))
    label_encoders = joblib.load(get_path("label_encoders.pkl"))
    feature_columns = joblib.load(get_path("feature_columns.pkl"))
    return model, best_model_name, scaler, label_encoders, feature_columns

model, best_model_name, scaler, label_encoders, feature_columns = load_model_files()

@st.cache_data
def load_model_insights():
    def get_path(filename):
        p = os.path.join(BASE_DIR, filename)
        return p if os.path.exists(p) else filename

    comparison_df = pd.read_csv(get_path("model_comparison.csv"))
    coef_df = pd.read_csv(get_path("linear_model_coefficients.csv"))
    return comparison_df, coef_df

comparison_df, coef_df = load_model_insights()

# -----------------------------------------------------------------------------
# Default Values & Features
# -----------------------------------------------------------------------------
def get_default_values(df):
    defaults = {}
    for col in df.columns:
        if col in ["carbon_footprint_kg", "carbon_impact_level", "user_id"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            mode_vals = df[col].mode(dropna=True)
            defaults[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else ""
    return defaults

defaults = get_default_values(df)

def create_features(df_input):
    df_feat = df_input.copy()

    # transport intensity
    if "distance_km" in df_feat.columns and "transport_mode" in df_feat.columns:
        transport_map = {
            "Car": 2.0,
            "Bus": 0.8,
            "EV": 0.3,
            "Bike": 0.0,
            "Walk": 0.0
        }
        df_feat["transport_intensity"] = (
            df_feat["distance_km"] * df_feat["transport_mode"].map(transport_map).fillna(0)
        )
        df_feat["long_distance"] = (df_feat["distance_km"] > 15).astype(int)
        df_feat["zero_emission_transport"] = df_feat["transport_mode"].isin(["Walk", "Bike"]).astype(int)

    # energy level
    if "electricity_kwh" in df_feat.columns:
        df_feat["energy_level"] = pd.cut(
            df_feat["electricity_kwh"],
            bins=[0, 4, 7, 1000],
            labels=["Low", "Medium", "High"],
            include_lowest=True
        )

    # renewable champion
    if "renewable_usage_pct" in df_feat.columns:
        df_feat["renewable_champion"] = (df_feat["renewable_usage_pct"] >= 75).astype(int)

    # dirty energy
    if "electricity_kwh" in df_feat.columns and "renewable_usage_pct" in df_feat.columns:
        df_feat["dirty_energy_kwh"] = (
            df_feat["electricity_kwh"] * (1 - df_feat["renewable_usage_pct"] / 100)
        )

    # waste per screen hour
    if "waste_generated_kg" in df_feat.columns and "screen_time_hours" in df_feat.columns:
        df_feat["waste_per_screen_hour"] = (
            df_feat["waste_generated_kg"] / (df_feat["screen_time_hours"] + 1)
        )

    # food score
    if "food_type" in df_feat.columns:
        food_map = {
            "Vegan": 0.5,
            "Vegetarian": 0.8,
            "Pescatarian": 1.0,
            "Omnivore": 1.5,
            "Non-Vegetarian": 1.5,
            "Veg": 0.8,
            "Non-Veg": 1.5,
            "Mixed": 1.1
        }
        df_feat["diet_emission_score"] = df_feat["food_type"].map(food_map).fillna(1.0)

    return df_feat

def prepare_input(user_input_dict):
    input_df = pd.DataFrame([user_input_dict])
    input_df = create_features(input_df)

    categorical_cols = input_df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            input_df[col] = input_df[col].astype(str)
            known_classes = list(le.classes_)
            input_df[f"{col}_encoded"] = input_df[col].apply(
                lambda x: le.transform([x])[0] if x in known_classes else -1
            )

    input_df.drop(columns=categorical_cols, inplace=True, errors="ignore")
    input_df = input_df.select_dtypes(include=[np.number]).copy()

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]
    input_df = input_df.fillna(0)
    return input_df

def predict_carbon(user_input_dict):
    model_input = prepare_input(user_input_dict)
    if best_model_name == "Linear Regression":
        input_data = scaler.transform(model_input)
    else:
        input_data = model_input

    prediction = model.predict(input_data)[0]
    prediction = max(prediction, 0)
    return prediction, model_input

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 4px 16px 4px;">
        <div style="font-weight: 700; font-size: 15px; color: #f0f6fc;">Carbon Footprint Predictor</div>
        <div style="font-size: 12px; color: #8b949e; margin-top: 2px;">Machine Learning Application</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Data Analysis",
            "Predict Footprint",
            "Category Sensitivity",
            "Model Performance"
        ]
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #30363d; margin: 18px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 4px;">
        <div style="font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase; margin-bottom: 8px;">Dataset Baseline</div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
            <span style="color: #8b949e;">Target (< 1.5°C):</span>
            <span style="font-weight: 600; color: #3fb950;">&lt; 5.0 kg</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
            <span style="color: #8b949e;">Dataset Median:</span>
            <span style="font-weight: 600; color: #58a6ff;">7.84 kg</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span style="color: #8b949e;">Active Model:</span>
            <span style="font-weight: 600; color: #c9d1d9;">Linear Reg.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top Page Banner (Professional, Non-AI)
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="page-header">
    <div class="page-title">Personal Carbon Footprint Predictor</div>
    <div class="page-desc">
        Estimation of daily individual carbon emissions (kg CO₂/day) based on lifestyle, transportation, energy, and consumption metrics using <b>{best_model_name}</b> (R²: 0.9998).
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. DASHBOARD
# -----------------------------------------------------------------------------
if page == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Dataset Records</div>
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-meta" style="color: #8b949e;">Individual observations</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_val = df["carbon_footprint_kg"].mean()
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Average Emission</div>
            <div class="metric-value">{avg_val:.2f} <span style="font-size: 14px; font-weight: 400; color: #8b949e;">kg/day</span></div>
            <div class="metric-meta" style="color: #8b949e;">Median: {df['carbon_footprint_kg'].median():.2f} kg</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">99.98%</div>
            <div class="metric-meta">Test R² Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        low_ratio = (df["carbon_impact_level"] == "Low").mean() * 100
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Low Impact Cohort</div>
            <div class="metric-value">{low_ratio:.1f}%</div>
            <div class="metric-meta">&lt; 8.0 kg CO₂/day</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.1, 0.9])

    with c_left:
        st.markdown("""
        <div class="card">
            <div class="card-title">Project Overview</div>
            <p style="color: #8b949e; line-height: 1.6; margin: 0 0 12px 0;">
                This system models personal daily carbon footprints using behavioral attributes across transport mode, distance travelled, electricity usage, clean energy adoption, and waste generation.
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px;">
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="font-size: 11px; color: #8b949e; font-weight: 600;">HIGHEST CORRELATION</div>
                    <div style="color: #f0f6fc; font-weight: 600; font-size: 13px; margin-top: 2px;">Dirty Energy (kWh)</div>
                    <div style="color: #3fb950; font-size: 11px;">Strongest positive driver</div>
                </div>
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="font-size: 11px; color: #8b949e; font-weight: 600;">TRANSIT IMPACT</div>
                    <div style="color: #f0f6fc; font-weight: 600; font-size: 13px; margin-top: 2px;">Transport Intensity</div>
                    <div style="color: #58a6ff; font-size: 11px;">Distance × Mode weight</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div class="card">
            <div class="card-title">Impact Distribution</div>
        """, unsafe_allow_html=True)

        counts = df["carbon_impact_level"].value_counts(normalize=True) * 100
        for level, color, desc in [
            ("Low", "#3fb950", "Under 8.0 kg CO₂/day"),
            ("Medium", "#d29922", "8.0 to 15.0 kg CO₂/day"),
            ("High", "#f85149", "Over 15.0 kg CO₂/day")
        ]:
            pct = counts.get(level, 0.0)
            st.markdown(f"""
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
                    <span style="color: #c9d1d9;"><b>{level}</b> ({desc})</span>
                    <span style="color: {color}; font-weight: 600;">{pct:.1f}%</span>
                </div>
                <div style="background: #21262d; height: 6px; border-radius: 3px; overflow: hidden;">
                    <div style="background: {color}; width: {pct}%; height: 100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset Sample</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

# -----------------------------------------------------------------------------
# 2. DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
elif page == "Data Analysis":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Distributions", "Energy & Diet", "Feature Correlation"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(df["carbon_footprint_kg"], bins=25, kde=True, color="#238636", ax=ax)
            ax.axvline(df["carbon_footprint_kg"].mean(), color='#58a6ff', linestyle='--', label=f'Mean ({df["carbon_footprint_kg"].mean():.2f})')
            ax.axvline(df["carbon_footprint_kg"].median(), color='#d29922', linestyle=':', label=f'Median ({df["carbon_footprint_kg"].median():.2f})')
            ax.set_title("Carbon Footprint Distribution (kg CO₂/day)")
            ax.set_xlabel("Footprint (kg CO₂/day)")
            ax.set_ylabel("Count")
            ax.legend(facecolor='#161b22', edgecolor='none', labelcolor='#c9d1d9')
            apply_plot_theme(fig, ax)
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            avg_mode = df.groupby("transport_mode")["carbon_footprint_kg"].mean().sort_values(ascending=False)
            sns.barplot(x=avg_mode.index, y=avg_mode.values, palette="crest", ax=ax)
            ax.set_title("Average Carbon Footprint by Transport Mode")
            ax.set_xlabel("Transport Mode")
            ax.set_ylabel("Average Footprint (kg CO₂/day)")
            for i, v in enumerate(avg_mode.values):
                ax.text(i, v + 0.1, f"{v:.2f}", ha='center', color='#f0f6fc', fontsize=9)
            apply_plot_theme(fig, ax)
            st.pyplot(fig)
            plt.close()

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.scatterplot(
                data=df,
                x="renewable_usage_pct",
                y="carbon_footprint_kg",
                hue="transport_mode",
                palette="viridis",
                alpha=0.75,
                s=35,
                ax=ax
            )
            ax.set_title("Renewable Energy Usage vs. Carbon Footprint")
            ax.set_xlabel("Renewable Usage (%)")
            ax.set_ylabel("Carbon Footprint (kg CO₂/day)")
            ax.legend(facecolor='#161b22', edgecolor='none', labelcolor='#c9d1d9', fontsize=8)
            apply_plot_theme(fig, ax)
            st.pyplot(fig)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            available_diets = sorted(df["food_type"].dropna().unique().tolist())
            sns.boxplot(
                data=df,
                x="food_type",
                y="carbon_footprint_kg",
                order=available_diets if len(available_diets) > 0 else None,
                palette="mako",
                ax=ax
            )
            ax.set_title("Emissions by Food / Diet Type")
            ax.set_xlabel("Food Type")
            ax.set_ylabel("Carbon Footprint (kg CO₂/day)")
            apply_plot_theme(fig, ax)
            st.pyplot(fig)
            plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(9, 5))
        numeric_df = df.select_dtypes(include=np.number)
        sns.heatmap(
            numeric_df.corr(),
            cmap="mako",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor='#161b22',
            cbar_kws={'label': 'Correlation'},
            ax=ax
        )
        ax.set_title("Correlation Heatmap of Behavioral Features")
        apply_plot_theme(fig, ax)
        st.pyplot(fig)
        plt.close()

# -----------------------------------------------------------------------------
# 3. PREDICT FOOTPRINT
# -----------------------------------------------------------------------------
elif page == "Predict Footprint":
    st.markdown('<div class="section-title">Daily Carbon Footprint Prediction</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8b949e; font-size: 13px; margin-bottom: 16px;">Configure your daily activity inputs below to generate an estimated carbon footprint.</p>', unsafe_allow_html=True)

    input_data = {}

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="card">
            <div class="card-title">Transportation & Commute</div>
        """, unsafe_allow_html=True)

        modes = sorted(df["transport_mode"].dropna().astype(str).unique().tolist())
        input_data["transport_mode"] = st.selectbox("Transport Mode", modes, index=modes.index(str(defaults.get("transport_mode", modes[0]))))

        dist_min = float(df["distance_km"].min())
        input_data["distance_km"] = st.number_input(
            "Distance Travelled (km)",
            min_value=dist_min,
            value=float(defaults.get("distance_km", 9.0)),
            step=0.5
        )

        days = sorted(df["day_type"].dropna().astype(str).unique().tolist())
        input_data["day_type"] = st.selectbox("Day Type", days, index=days.index(str(defaults.get("day_type", days[0]))))

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">Diet & Food Consumption</div>
        """, unsafe_allow_html=True)

        foods = sorted(df["food_type"].dropna().astype(str).unique().tolist())
        input_data["food_type"] = st.selectbox("Food Type", foods, index=foods.index(str(defaults.get("food_type", foods[0]))))

        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
            <div class="card-title">Electricity & Power Usage</div>
        """, unsafe_allow_html=True)

        elec_min = float(df["electricity_kwh"].min())
        input_data["electricity_kwh"] = st.number_input(
            "Daily Electricity Consumed (kWh)",
            min_value=elec_min,
            value=float(defaults.get("electricity_kwh", 5.0)),
            step=0.5
        )

        input_data["renewable_usage_pct"] = st.slider(
            "Renewable Usage (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.get("renewable_usage_pct", 30.0)),
            step=1.0
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">Waste & Screen Time</div>
        """, unsafe_allow_html=True)

        waste_min = float(df["waste_generated_kg"].min())
        input_data["waste_generated_kg"] = st.number_input(
            "Solid Waste Generated (kg)",
            min_value=waste_min,
            value=float(defaults.get("waste_generated_kg", 1.2)),
            step=0.1
        )

        screen_min = float(df["screen_time_hours"].min())
        input_data["screen_time_hours"] = st.number_input(
            "Daily Screen Time (Hours)",
            min_value=screen_min,
            value=float(defaults.get("screen_time_hours", 6.0)),
            step=0.5
        )

        input_data["eco_actions"] = st.number_input(
            "Eco-Actions Count",
            min_value=0.0,
            max_value=20.0,
            value=float(defaults.get("eco_actions", 1.0)),
            step=1.0
        )

        st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("Calculate Carbon Footprint")

    if predict_btn:
        prediction, model_input = predict_carbon(input_data)

        if prediction < 8.0:
            badge_html = '<span class="badge-low">Low Impact</span>'
            status_text = "Your daily footprint is within the sustainable target range."
            border_color = "#3fb950"
        elif prediction < 15.0:
            badge_html = '<span class="badge-med">Moderate Impact</span>'
            status_text = "Your daily footprint is average. Consider transit and energy adjustments to reduce emissions."
            border_color = "#d29922"
        else:
            badge_html = '<span class="badge-high">High Impact</span>'
            status_text = "Your daily footprint is significantly above average. Major emission drivers are outlined below."
            border_color = "#f85149"

        st.markdown(f"""
        <div style="background: #161b22; border: 1px solid {border_color}; border-radius: 8px; padding: 22px; margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: 600;">Prediction Result</div>
                    <div style="font-size: 32px; font-weight: 700; color: #f0f6fc; margin: 4px 0;">
                        {prediction:.2f} <span style="font-size: 16px; color: #8b949e; font-weight: 400;">kg CO₂ / day</span>
                    </div>
                    <div style="font-size: 13px; color: #c9d1d9;">{status_text}</div>
                </div>
                <div>{badge_html}</div>
            </div>
            
            <hr style="border: none; border-top: 1px solid #30363d; margin: 16px 0;">
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;">
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="font-size: 11px; color: #8b949e;">ANNUALIZED FOOTPRINT</div>
                    <div style="font-size: 15px; font-weight: 600; color: #f0f6fc;">{(prediction * 365 / 1000):.2f} tonnes/year</div>
                </div>
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="font-size: 11px; color: #8b949e;">PARIS TARGET BENCHMARK</div>
                    <div style="font-size: 15px; font-weight: 600; color: #3fb950;">&lt; 5.0 kg CO₂/day</div>
                </div>
                <div style="background: #0d1117; padding: 12px; border-radius: 6px; border: 1px solid #30363d;">
                    <div style="font-size: 11px; color: #8b949e;">DATASET COHORT MEDIAN</div>
                    <div style="font-size: 15px; font-weight: 600; color: #58a6ff;">7.84 kg CO₂/day</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Key Action Items to Reduce Emissions</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)

        with r1:
            if input_data["transport_mode"] in ["Car"]:
                st.info("Transport: Commuting by bus, bike, or EV can save roughly 2.0 to 4.0 kg CO₂ per day.")
            else:
                st.success("Transport: Your transit mode is already low-emission.")

        with r2:
            if input_data["renewable_usage_pct"] < 50.0:
                st.info("Electricity: Increasing clean energy share to over 50% helps reduce grid emissions significantly.")
            else:
                st.success("Electricity: You have a high renewable energy share.")

        with r3:
            if input_data["food_type"] in ["Non-Veg"]:
                st.info("Diet: Introducing plant-based meal replacements 2-3 days a week lowers dietary emissions.")
            else:
                st.success("Diet: Your food choices align with lower carbon intensity.")

        with st.expander("View Model Input Vector"):
            st.dataframe(model_input, use_container_width=True)

# -----------------------------------------------------------------------------
# 4. CATEGORY SENSITIVITY
# -----------------------------------------------------------------------------
elif page == "Category Sensitivity":
    st.markdown('<div class="section-title">Single Category Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8b949e; font-size: 13px; margin-bottom: 16px;">Vary a single category while holding all other parameters fixed at cohort medians.</p>', unsafe_allow_html=True)

    category_choice = st.selectbox(
        "Select Category to Vary",
        ["Transport", "Electricity & Grid", "Diet", "Lifestyle & Screen Time", "Waste Generation"]
    )

    user_input = {}
    for col in df.columns:
        if col not in ["carbon_footprint_kg", "carbon_impact_level", "user_id"]:
            user_input[col] = defaults[col]

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if category_choice == "Transport":
        transport_options = sorted(df["transport_mode"].dropna().astype(str).unique().tolist())
        user_input["transport_mode"] = st.selectbox("Transport Mode", transport_options)
        user_input["distance_km"] = st.slider("Daily Distance (km)", min_value=0.5, max_value=40.0, value=float(defaults["distance_km"]), step=0.5)

    elif category_choice == "Electricity & Grid":
        user_input["electricity_kwh"] = st.slider("Electricity Used (kWh)", min_value=0.5, max_value=30.0, value=float(defaults["electricity_kwh"]), step=0.5)
        user_input["renewable_usage_pct"] = st.slider("Renewable Share (%)", min_value=0.0, max_value=100.0, value=float(defaults["renewable_usage_pct"]), step=1.0)

    elif category_choice == "Diet":
        food_options = sorted(df["food_type"].dropna().astype(str).unique().tolist())
        user_input["food_type"] = st.selectbox("Food Type", food_options)

    elif category_choice == "Lifestyle & Screen Time":
        user_input["screen_time_hours"] = st.slider("Screen Time (Hours)", min_value=0.0, max_value=18.0, value=float(defaults["screen_time_hours"]), step=0.5)
        user_input["eco_actions"] = st.slider("Proactive Eco Actions", min_value=0.0, max_value=10.0, value=float(defaults["eco_actions"]), step=1.0)

    elif category_choice == "Waste Generation":
        user_input["waste_generated_kg"] = st.slider("Waste Generated (kg)", min_value=0.1, max_value=10.0, value=float(defaults["waste_generated_kg"]), step=0.1)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run Simulation"):
        prediction, model_input = predict_carbon(user_input)
        baseline_prediction, _ = predict_carbon(defaults)
        delta = prediction - baseline_prediction

        delta_sign = "+" if delta > 0 else ""
        delta_color = "#f85149" if delta > 0.05 else ("#3fb950" if delta < -0.05 else "#8b949e")

        st.markdown(f"""
        <div class="card" style="margin-top: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div>
                    <div class="metric-label">Simulated Footprint</div>
                    <div style="font-size: 26px; font-weight: 700; color: #f0f6fc;">
                        {prediction:.2f} <span style="font-size: 13px; font-weight: 400; color: #8b949e;">kg CO₂/day</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="metric-label">Difference vs. Baseline</div>
                    <div style="font-size: 20px; font-weight: 700; color: {delta_color};">
                        {delta_sign}{delta:.2f} kg CO₂
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "Model Performance":
    st.markdown('<div class="section-title">Model Evaluation & Insights</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Best Model</div>
            <div class="metric-value" style="font-size: 20px;">{best_model_name}</div>
            <div class="metric-meta" style="color: #8b949e;">Selected by test R² score</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Test R² Score</div>
            <div class="metric-value">0.9998</div>
            <div class="metric-meta">Explained variance</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Mean Absolute Error</div>
            <div class="metric-value">0.03 <span style="font-size: 14px; font-weight: 400; color: #8b949e;">kg</span></div>
            <div class="metric-meta" style="color: #8b949e;">RMSE: 0.04 kg CO₂/day</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    st.markdown("<b>Model Comparison Table</b>", unsafe_allow_html=True)
    st.dataframe(comparison_df, use_container_width=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    col_chart, col_guide = st.columns([1.2, 0.8])

    with col_chart:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        top_coef = coef_df.head(10)
        colors = ['#f85149' if x > 0 else '#3fb950' for x in top_coef["Coefficient"]]
        sns.barplot(data=top_coef, x="Coefficient", y="Feature", palette=colors, ax=ax)
        ax.set_title("Top 10 Feature Coefficients (Linear Regression)")
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Feature")
        apply_plot_theme(fig, ax)
        st.pyplot(fig)
        plt.close()

    with col_guide:
        st.markdown("""
        <div class="card" style="height: 100%;">
            <div class="card-title">Coefficient Interpretation</div>
            <ul style="color: #8b949e; font-size: 13px; line-height: 1.7; padding-left: 18px; margin: 0;">
                <li><b style="color: #f85149;">Positive (+):</b> Increases estimated carbon footprint (e.g. <i>dirty_energy_kwh</i>, <i>diet_score</i>).</li>
                <li><b style="color: #3fb950;">Negative (-):</b> Decreases estimated carbon footprint (e.g. <i>sustainability_score</i>, <i>eco_actions</i>).</li>
                <li><b>Magnitude:</b> Indicates the strength of the variable's impact on daily emissions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
