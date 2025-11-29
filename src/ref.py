import math
from typing import List

import folium
import mercantile
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import pdist, squareform
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Map Logic Comparison: All in One")


# ---------------------------------------------------------
# 1. データ生成 & 指標計算
# ---------------------------------------------------------
@st.cache_data
def generate_dummy_data(n: int = 1000):
    base_lat = 35.690921
    base_lon = 139.700258
    data = pd.DataFrame(
        {
            "id": range(n),
            "lat": np.random.normal(base_lat, 0.008, n),
            "lon": np.random.normal(base_lon, 0.008, n),
            "score": np.round(np.random.uniform(2.5, 5.0, n), 2),
            "name": [f"Dining_{i}" for i in range(n)],
        }
    )
    return data


def calculate_spatial_metrics(df: pd.DataFrame, zoom: int, min_overlap_px: int) -> dict:
    if len(df) < 2:
        return {"avg_nnd": 0.0, "collision_count": 0}

    earth_circumference = 2 * math.pi * 6378137
    meters_per_pixel = earth_circumference / (256 * 2**zoom)

    coords_px = []
    for _, row in df.iterrows():
        mx, my = mercantile.xy(row["lon"], row["lat"])
        px = mx / meters_per_pixel
        py = my / meters_per_pixel
        coords_px.append([px, py])

    dist_matrix = squareform(pdist(coords_px))
    np.fill_diagonal(dist_matrix, np.inf)

    collision_count = np.sum(np.triu(dist_matrix < min_overlap_px, k=1))
    nearest_dists = dist_matrix.min(axis=1)
    avg_nnd = np.mean(nearest_dists)

    return {"avg_nnd": avg_nnd, "collision_count": int(collision_count)}


# ---------------------------------------------------------
# 2. ロジック定義
# ---------------------------------------------------------
def select_points_baseline(df_in_view: pd.DataFrame, limit: int):
    return df_in_view.sort_values("score", ascending=False).head(limit)


def select_points_hard_greedy(
    df_in_view: pd.DataFrame, limit: int, zoom: int, min_pixel_dist: int
):
    if df_in_view.empty:
        return df_in_view
    earth_circumference = 2 * math.pi * 6378137
    meters_per_pixel = earth_circumference / (256 * 2**zoom)
    min_dist_meters = min_pixel_dist * meters_per_pixel

    candidates = df_in_view.sort_values("score", ascending=False)
    selected = []
    selected_meters = []

    for _, row in candidates.iterrows():
        if len(selected) >= limit:
            break
        mx, my = mercantile.xy(row["lon"], row["lat"])
        if not selected:
            selected.append(row)
            selected_meters.append((mx, my))
            continue
        is_far_enough = True
        for smx, smy in selected_meters:
            dist_m = math.sqrt((mx - smx) ** 2 + (my - smy) ** 2)
            if dist_m < min_dist_meters:
                is_far_enough = False
                break
        if is_far_enough:
            selected.append(row)
            selected_meters.append((mx, my))
    return pd.DataFrame(selected)


def select_points_soft_penalty(
    df_in_view: pd.DataFrame,
    limit: int,
    zoom: int,
    min_pixel_dist: int,
    penalty_weight: float,
):
    if df_in_view.empty:
        return df_in_view
    earth_circumference = 2 * math.pi * 6378137
    meters_per_pixel = earth_circumference / (256 * 2**zoom)
    limit_dist_meters = min_pixel_dist * meters_per_pixel

    candidates = df_in_view.copy()
    coords = candidates.apply(lambda x: mercantile.xy(x["lon"], x["lat"]), axis=1)
    candidates["mx"] = [c[0] for c in coords]
    candidates["my"] = [c[1] for c in coords]

    selected_indices = []
    selected_coords = []

    for _ in range(limit):
        if len(candidates) == 0:
            break
        remaining = candidates.drop(selected_indices)
        if remaining.empty:
            break

        if not selected_indices:
            best_idx = remaining["score"].idxmax()
            selected_indices.append(best_idx)
            selected_coords.append(
                (remaining.loc[best_idx, "mx"], remaining.loc[best_idx, "my"])
            )
            continue

        rem_coords = remaining[["mx", "my"]].values
        rem_scores = remaining["score"].values
        sel_coords_arr = np.array(selected_coords)
        diff = rem_coords[:, np.newaxis, :] - sel_coords_arr[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))

        normalized_dists = dists / limit_dist_meters
        penalties = np.maximum(0, 1.0 - normalized_dists)
        total_penalties = np.sum(penalties, axis=1)
        gains = rem_scores - (penalty_weight * total_penalties)

        best_global_idx = remaining.index[np.argmax(gains)]
        selected_indices.append(best_global_idx)
        selected_coords.append(
            (remaining.loc[best_global_idx, "mx"], remaining.loc[best_global_idx, "my"])
        )

    return df_in_view.loc[selected_indices]


def create_map(
    center: List[float], zoom: int, df_pins: pd.DataFrame, color: str, height: int = 400
):
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
    for _, row in df_pins.iterrows():
        popup_html = f"""<div style="width:120px"><b>{row["name"]}</b><br>Score: {row["score"]}</div>"""
        folium.Marker(
            [row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row['name']} ({row['score']})",
            icon=folium.Icon(color=color, icon="cutlery", prefix="fa"),
        ).add_to(m)
    return m


# ---------------------------------------------------------
# UI Implementation
# ---------------------------------------------------------
if "map_state" not in st.session_state:
    st.session_state["map_state"] = {
        "center": [35.690921, 139.700258],
        "zoom": 15,
        "bounds": None,
    }

st.title("📊 Map Logic Comparison Board")

with st.sidebar:
    st.header("Settings")
    limit_pins = st.slider("表示上限数", 5, 100, 30)
    min_pixels = st.slider(
        "基準距離 R (px)", 10, 100, 50, help="この距離未満をCollisionとみなします"
    )
    lambda_val = st.slider(
        "Soft許容度 λ", 0.0, 10.0, 2.0, 0.1, help="Soft Constraintの重み"
    )
    st.info("ℹ️ 一番上の「Baselineマップ」を動かすと、下の2つのマップも追従します。")

# --- マップの状態管理 (Masterは一番上のBaseline) ---
last_interaction = st.session_state.get("map_baseline_widget", None)
if last_interaction and last_interaction.get("bounds"):
    current_center = [
        last_interaction["center"]["lat"],
        last_interaction["center"]["lng"],
    ]
    current_zoom = last_interaction["zoom"]
    bounds = last_interaction["bounds"]
else:
    current_center = st.session_state["map_state"]["center"]
    current_zoom = st.session_state["map_state"]["zoom"]
    bounds = None

# データ生成とビューポートフィルタリング
df_all = generate_dummy_data()
if bounds:
    sw, ne = bounds["_southWest"], bounds["_northEast"]
    df_view = df_all[
        (df_all["lat"] >= sw["lat"])
        & (df_all["lat"] <= ne["lat"])
        & (df_all["lon"] >= sw["lng"])
        & (df_all["lon"] <= ne["lng"])
    ]
else:
    df_view = df_all.copy()

# --- 全ロジックの一括計算 ---
# 1. Baseline
df_base = select_points_baseline(df_view, limit_pins)
met_base = calculate_spatial_metrics(df_base, current_zoom, min_pixels)

# 2. Hard
df_hard = select_points_hard_greedy(df_view, limit_pins, current_zoom, min_pixels)
met_hard = calculate_spatial_metrics(df_hard, current_zoom, min_pixels)

# 3. Soft
df_soft = select_points_soft_penalty(
    df_view, limit_pins, current_zoom, min_pixels, lambda_val
)
met_soft = calculate_spatial_metrics(df_soft, current_zoom, min_pixels)

# --- 比較テーブルの表示 ---
st.subheader("📈 Metrics Summary")
summary_data = {
    "Logic": [
        "🔴 Baseline (Score Only)",
        "🔵 Hard Constraint (Greedy)",
        "🟢 Soft Constraint (Penalty)",
    ],
    "Displayed Pins": [len(df_base), len(df_hard), len(df_soft)],
    "Avg Score (Quality)": [
        f"{df_base['score'].mean():.2f}" if not df_base.empty else "0.00",
        f"{df_hard['score'].mean():.2f}" if not df_hard.empty else "0.00",
        f"{df_soft['score'].mean():.2f}" if not df_soft.empty else "0.00",
    ],
    "Collision Count (Clutter)": [
        met_base["collision_count"],
        met_hard["collision_count"],
        met_soft["collision_count"],
    ],
    "AvgNND (Dispersion)": [
        f"{met_base['avg_nnd']:.1f}px",
        f"{met_hard['avg_nnd']:.1f}px",
        f"{met_soft['avg_nnd']:.1f}px",
    ],
}
st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

st.divider()

# --- マップの縦並び表示 ---

# Row 1: Baseline (Controller)
st.markdown("### 🔴 1. Baseline (Score Only)")
st.caption(
    "最もスコアが高い店を表示。距離は無視するため重なりが多い。このマップを操作してください。"
)
m1 = create_map(current_center, current_zoom, df_base, "red", 400)
# これがコントローラーになる
st_folium(m1, width="100%", height=400, key="map_baseline_widget")

st.markdown("---")

# Row 2: Hard Constraint
st.markdown("### 🔵 2. Hard Constraint (Greedy)")
st.caption(
    f"距離 {min_pixels}px 以内の重なりを**絶対に許さない**。視認性は完璧だが、密集地のスコアを取りこぼす。"
)
m2 = create_map(current_center, current_zoom, df_hard, "blue", 400)
st_folium(m2, width="100%", height=400, key="map_hard_widget")

st.markdown("---")

# Row 3: Soft Constraint
st.markdown(f"### 🟢 3. Soft Constraint (Penalty λ={lambda_val})")
st.caption("スコアが高ければ多少の重なりを許容する。BaselineとHardの中間的な挙動。")
m3 = create_map(current_center, current_zoom, df_soft, "green", 400)
st_folium(m3, width="100%", height=400, key="map_soft_widget")
