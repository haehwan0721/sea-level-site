import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# Sea Level Fingerprint Web App
# Single-file Streamlit app
# ============================================================

st.set_page_config(
    page_title="Sea Level Fingerprint Simulator",
    page_icon="🌊",
    layout="wide"
)

# -----------------------------
# Constants
# -----------------------------
R_EARTH = 6_371_000.0          # m
RHO_WATER = 1000.0             # kg/m^3
GT_TO_KG = 1e12                # 1 gigaton = 1e12 kg
OCEAN_AREA = 3.61e14           # m^2 (approx global ocean area)

# -----------------------------
# Grid
# -----------------------------
NLAT = 181
NLON = 360
LATS = np.linspace(90, -90, NLAT)
LONS = np.linspace(0, 359, NLON)
LON2D, LAT2D = np.meshgrid(LONS, LATS)
WEIGHTS = np.cos(np.deg2rad(LAT2D))
WEIGHTS = np.clip(WEIGHTS, 0, None)


# -----------------------------
# Coarse land mask for map display
# land=True, ocean=False
# -----------------------------
def ellipse_mask(lat, lon, lat0, lon0, a_lat, a_lon):
    dlon = ((lon - lon0 + 180) % 360) - 180
    return ((lat - lat0) / a_lat) ** 2 + (dlon / a_lon) ** 2 <= 1


def build_land_mask(lat2d, lon2d):
    land = np.zeros_like(lat2d, dtype=bool)

    # Very coarse continent ellipses
    land |= ellipse_mask(lat2d, lon2d, 50, 260, 22, 35)   # North America
    land |= ellipse_mask(lat2d, lon2d, 15, 280, 12, 18)   # Central America
    land |= ellipse_mask(lat2d, lon2d, -15, 300, 28, 18)  # South America
    land |= ellipse_mask(lat2d, lon2d, 55, 60, 22, 70)    # Eurasia
    land |= ellipse_mask(lat2d, lon2d, 10, 20, 28, 22)    # Africa
    land |= ellipse_mask(lat2d, lon2d, -25, 135, 12, 16)  # Australia
    land |= ellipse_mask(lat2d, lon2d, 73, 320, 10, 16)   # Greenland
    land |= (lat2d < -75)                                 # Antarctica

    return land


LAND_MASK = build_land_mask(LAT2D, LON2D)
OCEAN_MASK = ~LAND_MASK


# -----------------------------
# Presets
# -----------------------------
PRESETS = {
    "Greenland": {"lat": 72.0, "lon": 320.0, "radius_deg": 12.0},
    "West Antarctica": {"lat": -78.0, "lon": 250.0, "radius_deg": 16.0},
    "East Antarctica": {"lat": -78.0, "lon": 110.0, "radius_deg": 20.0},
    "Alaska": {"lat": 61.0, "lon": 210.0, "radius_deg": 8.0},
    "Patagonia": {"lat": -49.0, "lon": 290.0, "radius_deg": 8.0},
    "Himalaya": {"lat": 30.0, "lon": 85.0, "radius_deg": 8.0},
    "Custom": {"lat": 72.0, "lon": 320.0, "radius_deg": 12.0},
}

CITIES = {
    "Seoul": (37.5665, 126.9780),
    "Tokyo": (35.6762, 139.6503),
    "Shanghai": (31.2304, 121.4737),
    "Singapore": (1.3521, 103.8198),
    "Jakarta": (-6.2088, 106.8456),
    "Sydney": (-33.8688, 151.2093),
    "Honolulu": (21.3069, -157.8583),
    "Los Angeles": (34.0522, -118.2437),
    "New York": (40.7128, -74.0060),
    "London": (51.5074, -0.1278),
    "Amsterdam": (52.3676, 4.9041),
    "Mumbai": (19.0760, 72.8777),
    "Cape Town": (-33.9249, 18.4241),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Nuuk": (64.1835, -51.7216),
}


# -----------------------------
# Utility functions
# -----------------------------
def wrap_lon_360(lon_deg):
    return lon_deg % 360.0


def angular_distance_deg(lat1, lon1, lat2, lon2):
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)

    cos_gamma = (
        np.sin(lat1r) * np.sin(lat2r)
        + np.cos(lat1r) * np.cos(lat2r) * np.cos(lon1r - lon2r)
    )
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_gamma))


def mass_gt_to_eustatic_slr_m(mass_gt):
    mass_kg = mass_gt * GT_TO_KG
    water_volume_m3 = mass_kg / RHO_WATER
    return water_volume_m3 / OCEAN_AREA


def compute_fingerprint(
    source_lat,
    source_lon,
    mass_gt,
    source_radius_deg,
    grav_strength=1.25,
    uplift_strength=0.55,
    farfield_boost=0.10
):
    """
    Physically-inspired approximate fingerprint model for a website.
    Output: relative sea level change [m] over ocean.
    """
    source_lon = wrap_lon_360(source_lon)
    gamma = angular_distance_deg(LAT2D, LON2D, source_lat, source_lon)

    eustatic = mass_gt_to_eustatic_slr_m(mass_gt)

    # Near-field gravitational loss: sea level drops near melt source
    sigma_g = max(source_radius_deg * 1.8, 6.0)
    grav_term = -grav_strength * eustatic * np.exp(-(gamma / sigma_g) ** 2)

    # Bedrock uplift: additional near-field local fall
    sigma_u = max(source_radius_deg * 1.1, 4.0)
    uplift_term = -uplift_strength * eustatic * np.exp(-(gamma / sigma_u) ** 2)

    # Far-field enhancement: distant regions rise a bit more
    far_term = farfield_boost * eustatic * (1.0 - np.cos(np.deg2rad(gamma)))

    raw = grav_term + uplift_term + far_term

    # Mass conservation over ocean:
    # ocean-weighted mean must equal eustatic SLR
    ocean_weights = WEIGHTS * OCEAN_MASK
    raw_mean_ocean = np.nansum(raw * ocean_weights) / np.nansum(ocean_weights)
    corrected = raw + (eustatic - raw_mean_ocean)

    # Only ocean is meaningful
    rsl = np.where(OCEAN_MASK, corrected, np.nan)

    return rsl, eustatic


def nearest_grid_value(lat, lon, field):
    lon = wrap_lon_360(lon)
    i = np.argmin(np.abs(LATS - lat))
    j = np.argmin(np.abs(LONS - lon))
    return float(field[i, j])


def add_marker_trace(fig, city_rows):
    fig.add_trace(
        go.Scattergeo(
            lon=[r["lon"] for r in city_rows],
            lat=[r["lat"] for r in city_rows],
            text=[f'{r["city"]}: {r["rsl_cm"]:+.2f} cm' for r in city_rows],
            mode="markers+text",
            textposition="top center",
            marker=dict(size=6),
            name="Cities"
        )
    )


def build_map_figure(rsl_m, title):
    z = rsl_m * 100.0  # cm
    vmax = np.nanmax(np.abs(z))
    if not np.isfinite(vmax) or vmax < 0.01:
        vmax = 0.01

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=LONS,
            y=LATS,
            colorscale="RdBu_r",
            zmin=-vmax,
            zmax=vmax,
            colorbar=dict(title="cm"),
            hovertemplate="Lon %{x}°<br>Lat %{y}°<br>ΔRSL %{z:.2f} cm<extra></extra>"
        )
    )

    # Overlay land in light gray
    land_numeric = np.where(LAND_MASK, 1.0, np.nan)
    fig.add_trace(
        go.Heatmap(
            z=land_numeric,
            x=LONS,
            y=LATS,
            colorscale=[[0.0, "#d9d9d9"], [1.0, "#d9d9d9"]],
            showscale=False,
            hoverinfo="skip",
            opacity=0.95
        )
    )

    fig.update_layout(
        title=title,
        height=650,
        xaxis=dict(title="Longitude", range=[0, 359]),
        yaxis=dict(title="Latitude", range=[-90, 90]),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


# -----------------------------
# UI
# -----------------------------
st.title("🌊 Sea Level Fingerprint Simulator")
st.markdown(
    """
특정 빙하/빙상 지역이 녹았을 때, 전 세계 각 지역의 **상대 해수면 변화**를 계산해 보여주는 사이트입니다.  
입력한 질량 손실량에 따라 지도와 주요 도시의 해수면 변화를 바로 확인할 수 있습니다.
"""
)

with st.sidebar:
    st.header("입력 (Input)")

    preset = st.selectbox("빙하/빙상 위치", list(PRESETS.keys()), index=0)
    default = PRESETS[preset]

    if preset == "Custom":
        source_lat = st.number_input("위도 (lat)", min_value=-90.0, max_value=90.0, value=float(default["lat"]), step=1.0)
        source_lon = st.number_input("경도 (lon, -180~180 또는 0~360)", min_value=-180.0, max_value=360.0, value=-40.0, step=1.0)
        radius_deg = st.slider("영향 반경 (도)", min_value=3, max_value=40, value=int(default["radius_deg"]))
    else:
        source_lat = float(default["lat"])
        source_lon = float(default["lon"])
        radius_deg = st.slider("빙상/빙하 분포 반경 (도)", min_value=3, max_value=40, value=int(default["radius_deg"]))

    mass_gt = st.number_input("녹은 얼음 질량 (Gt)", min_value=1.0, max_value=10_000_000.0, value=1000.0, step=100.0)

    with st.expander("고급 설정"):
        grav_strength = st.slider("중력 약화 효과", 0.5, 2.5, 1.25, 0.05)
        uplift_strength = st.slider("지반 융기 효과", 0.0, 1.5, 0.55, 0.05)
        farfield_boost = st.slider("원거리 증폭 효과", 0.0, 0.5, 0.10, 0.01)

    selected_cities = st.multiselect(
        "출력할 도시",
        list(CITIES.keys()),
        default=["Nuuk", "Seoul", "Tokyo", "Honolulu", "New York", "London", "Sydney"]
    )

    run_button = st.button("계산 실행", use_container_width=True)

# -----------------------------
# Run calculation
# -----------------------------
if run_button or "last_result" not in st.session_state:
    rsl_m, eustatic_m = compute_fingerprint(
        source_lat=source_lat,
        source_lon=source_lon,
        mass_gt=mass_gt,
        source_radius_deg=radius_deg,
        grav_strength=grav_strength,
        uplift_strength=uplift_strength,
        farfield_boost=farfield_boost
    )

    city_rows = []
    for city in selected_cities:
        lat, lon = CITIES[city]
        value_m = nearest_grid_value(lat, lon, rsl_m)
        city_rows.append({
            "city": city,
            "lat": lat,
            "lon": wrap_lon_360(lon),
            "rsl_m": value_m,
            "rsl_cm": value_m * 100.0
        })

    max_rise_cm = float(np.nanmax(rsl_m) * 100.0)
    max_fall_cm = float(np.nanmin(rsl_m) * 100.0)

    st.session_state["last_result"] = {
        "rsl_m": rsl_m,
        "eustatic_m": eustatic_m,
        "city_rows": city_rows,
        "params": {
            "preset": preset,
            "source_lat": source_lat,
            "source_lon": source_lon,
            "radius_deg": radius_deg,
            "mass_gt": mass_gt,
            "grav_strength": grav_strength,
            "uplift_strength": uplift_strength,
            "farfield_boost": farfield_boost
        },
        "max_rise_cm": max_rise_cm,
        "max_fall_cm": max_fall_cm
    }

result = st.session_state["last_result"]
rsl_m = result["rsl_m"]
eustatic_m = result["eustatic_m"]
city_rows = result["city_rows"]

# -----------------------------
# Summary metrics
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("균일 해수면 상승량(전지구 평균)", f"{eustatic_m * 100:.3f} cm")
col2.metric("최대 상승 지역", f"{result['max_rise_cm']:+.3f} cm")
col3.metric("최대 하강 지역", f"{result['max_fall_cm']:+.3f} cm")

st.markdown(
    f"""
**입력 요약**  
- 위치: **{result['params']['preset']}**
- 중심 좌표: **({result['params']['source_lat']:.1f}°, {wrap_lon_360(result['params']['source_lon']):.1f}°)**
- 녹은 질량: **{result['params']['mass_gt']:,.1f} Gt**
"""
)

# -----------------------------
# Map
# -----------------------------
map_title = f"Relative Sea Level Change Fingerprint — {result['params']['preset']} ({result['params']['mass_gt']:,.0f} Gt)"
fig = build_map_figure(rsl_m, map_title)
add_marker_trace(fig, city_rows)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# City table
# -----------------------------
st.subheader("주요 도시별 해수면 변화")
city_df = pd.DataFrame(city_rows)[["city", "lat", "lon", "rsl_cm"]].copy()
city_df.columns = ["City", "Latitude", "Longitude", "Relative Sea Level Change (cm)"]
city_df["Relative Sea Level Change (cm)"] = city_df["Relative Sea Level Change (cm)"].map(lambda x: round(x, 3))
st.dataframe(city_df, use_container_width=True, hide_index=True)

# -----------------------------
# Download
# -----------------------------
download_df = pd.DataFrame({
    "lat": np.repeat(LATS, NLON),
    "lon": np.tile(LONS, NLAT),
    "rsl_m": rsl_m.flatten(),
    "rsl_cm": (rsl_m * 100.0).flatten(),
    "is_ocean": OCEAN_MASK.flatten()
})
csv = download_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="결과 CSV 다운로드",
    data=csv,
    file_name="sea_level_fingerprint_result.csv",
    mime="text/csv"
)

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "이 웹앱은 사이트용으로 바로 실행되도록 만든 물리 기반 근사 시뮬레이터입니다."
)
