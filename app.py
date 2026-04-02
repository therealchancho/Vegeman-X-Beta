import os
import re
import tempfile
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from model_engine import run_growth_pipeline

st.set_page_config(page_title="FPL Vegetation Predictor", layout="wide")

st.markdown(
    """
    <style>
        button[title="Step down"] {display: none;}
        button[title="Step up"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌴 VegeMan X Beta")

def parse_year_from_name(name: str) -> int:
    match = re.search(r"(20\d{2})", name)
    if not match:
        raise ValueError(f"Could not parse year from filename: {name}")
    return int(match.group(1))

@st.cache_data
def convert_df(df: pd.DataFrame) -> bytes:
    cols_to_drop = [c for c in ["geometry", "risk_color"] if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore").to_csv(index=False).encode("utf-8")

# ==========================================
# UI CONTROLS (SIDEBAR)
# ==========================================
st.sidebar.header("📁 Dynamic Data Input")
num_files = st.sidebar.number_input("How many CSV files are you uploading?", min_value=2, max_value=20, value=3)

uploaded_files = []
for i in range(num_files):
    f = st.sidebar.file_uploader(f"Upload CSV {i+1}", type=["csv"], key=f"file_{i}")
    if f is not None:
        uploaded_files.append(f)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Scenario Parameters")

prediction_rainfall = st.sidebar.number_input("Forecast Rainfall for Prediction Year", value=48.7, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("🧮 Clearance Handling")

uploaded_names = [f.name for f in uploaded_files]
compute_clearance_for = st.sidebar.multiselect(
    "Which uploaded files should compute Clearance from encroachment bins (u0...u4)?",
    options=uploaded_names,
    default=[],
)

# ==========================================
# EXECUTION BUTTON
# ==========================================
if st.sidebar.button("🚀 Run Prediction Engine"):
    if len(uploaded_files) < 2:
        st.sidebar.error("⚠️ Please upload at least 2 CSV files.")
    else:
        try:
            years = sorted(set(parse_year_from_name(f.name) for f in uploaded_files))
            if len(years) < 2:
                st.sidebar.error("⚠️ Need at least 2 distinct years in uploaded filenames.")
            else:
                st.sidebar.subheader("🌧️ Historical Rainfall Inputs")
                st.session_state["years_detected"] = years
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Filename/year parsing error: {e}")

# ==========================================
# DYNAMIC RAINFALL INPUTS
# ==========================================
if "years_detected" in st.session_state:
    years = st.session_state["years_detected"]
    st.sidebar.markdown("---")
    st.sidebar.header("🌧️ Rainfall Parameters (inches)")

    rainfall_values = {}
    for i in range(len(years) - 1):
        y1 = years[i]
        y2 = years[i + 1]
        rainfall_values[(y1, y2)] = st.sidebar.number_input(
            f"Historical Rain: {y1} ➔ {y2}",
            value=50.0,
            step=0.1,
            key=f"rain_{y1}_{y2}",
        )

    if st.sidebar.button("✅ Confirm Inputs and Run Model"):
        try:
            temp_filepaths = []

            for uf in uploaded_files:
                suffix = Path(uf.name).suffix or ".csv"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.getbuffer())
                    temp_filepaths.append(tmp.name)

            outputs = run_growth_pipeline(
                filepaths=temp_filepaths,
                rainfall_by_interval=rainfall_values,
                prediction_rainfall=prediction_rainfall,
                output_dir="outputs",
                make_plots=False,   # Let Streamlit handle display
            )

            df_final = outputs["results_df"].copy()
            pred_year = outputs["pred_year"]
            pred_clearance_col = outputs["pred_clearance_col"]
            pred_growth_col = outputs["pred_growth_col"]
            years = outputs["years"]
            base_year = years[0]

            # Save useful metadata for display
            st.session_state["df_final"] = df_final
            st.session_state["pred_year"] = pred_year
            st.session_state["pred_clearance_col"] = pred_clearance_col
            st.session_state["pred_growth_col"] = pred_growth_col
            st.session_state["base_lon_col"] = f"lon_{base_year}"
            st.session_state["base_lat_col"] = f"lat_{base_year}"

            st.success("✅ Model Complete! Dashboards are live.")

        except Exception as e:
            st.error(f"Model run failed: {e}")

        finally:
            for fp in temp_filepaths:
                try:
                    os.remove(fp)
                except Exception:
                    pass

# ==========================================
# MAIN DISPLAY & FILTERS
# ==========================================
if "df_final" in st.session_state:
    df_display = st.session_state["df_final"].copy()
    pred_year = st.session_state["pred_year"]
    pred_clearance_col = st.session_state["pred_clearance_col"]
    pred_growth_col = st.session_state["pred_growth_col"]
    lon_col = st.session_state["base_lon_col"]
    lat_col = st.session_state["base_lat_col"]

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Display Filters")

    if "substation_plot" in df_display.columns:
        substations = ["All Substations"] + sorted(df_display["substation_plot"].dropna().unique().tolist())
        user_substation = st.sidebar.selectbox("Filter by Substation", substations)
        if user_substation != "All Substations":
            df_display = df_display[df_display["substation_plot"] == user_substation]

    st.success("✅ Model Complete! Operations Dashboards are live.")
    st.write(f"📊 **Total Trees Tracked:** {len(st.session_state['df_final'])}")
    st.write(f"🔍 **Currently Displaying (Filtered):** {len(df_display)}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🗺️ Risk Map", "🌱 Growth Map", "📊 Summary"])

    with tab1:
        st.header("Predicted Clearance Risk Map")

        st.markdown("**Risk Level Legend (Predicted Distance to Wire):**")
        st.markdown("🟥 Critical   |   🟧 High Priority   |   🟨 Watch List   |   🟩 Healthy   |   ⬜ Safe")

        if df_display.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

            gdf_plot = gpd.GeoDataFrame(
                df_display,
                geometry=gpd.points_from_xy(df_display[lon_col], df_display[lat_col]),
                crs="EPSG:4326"
            ).to_crs(epsg=3857)

            ax.scatter(gdf_plot.geometry.x, gdf_plot.geometry.y, c=gdf_plot["risk_color"], s=25, alpha=0.9)
            ax.set_aspect("equal")
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=15)
            ax.set_axis_off()
            st.pyplot(fig)

    with tab2:
        st.header("Predicted Growth Map")

        if df_display.empty:
            st.warning("No data available for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

            gdf_plot = gpd.GeoDataFrame(
                df_display,
                geometry=gpd.points_from_xy(df_display[lon_col], df_display[lat_col]),
                crs="EPSG:4326"
            ).to_crs(epsg=3857)

            sc = ax.scatter(
                gdf_plot.geometry.x,
                gdf_plot.geometry.y,
                c=df_display[pred_growth_col],
                s=25,
                alpha=0.9,
                cmap="viridis"
            )
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=15)
            ax.set_axis_off()
            plt.colorbar(sc, ax=ax, label="Predicted Growth (ft)")
            st.pyplot(fig)

    with tab3:
        st.header("Risk Distribution Summary")

        if not df_display.empty:
            risk_summary = df_display.groupby("risk_bucket").size().reset_index(name="Point Count")
            st.dataframe(risk_summary, use_container_width=True)

            csv_data = convert_df(df_display)
            st.download_button(
                label="📥 Download Predicted Vegetation (CSV)",
                data=csv_data,
                file_name="predicted_vegetation_export.csv",
                mime="text/csv",
            )
        else:
            st.write("No data to summarize.")
