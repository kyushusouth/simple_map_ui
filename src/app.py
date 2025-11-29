import math

import folium
import mercantile
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import pdist
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Map Re-ranking Demo")


@st.cache_data
def generate_dummy_data(n: int = 1000):
    """ランダムな場所データを生成

    Args:
        n (int): 場所の数

    Returns:
        pd.DataFrame: 生成データ
    """
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


def calc_collision_count(df: pd.DataFrame, zoom: int, min_overlap_px: int) -> dict:
    """評価指標の計算"""
    if len(df) < 2:
        return 0

    earth_circumference = 2 * math.pi * 6378137
    meters_per_pixel = earth_circumference / (256 * 2**zoom)

    coords_px = []
    for _, row in df.iterrows():
        mx, my = mercantile.xy(row["lon"], row["lat"])
        px = mx / meters_per_pixel
        py = my / meters_per_pixel
        coords_px.append([px, py])

    collision_count = np.sum(pdist(coords_px) < min_overlap_px)
    return collision_count


def select_points_by_score(df_in_view: pd.DataFrame, limit: int):
    """ピンを立てる地点をスコアから決定する

    Args:
        df_in_view (pd.DataFrame): ビューポート内にある地点のデータ
        limit (int): 表示するピンの上限数

    Returns:
        pd.DataFrame: 選択された地点のデータ
    """
    return df_in_view.sort_values("score", ascending=False).head(limit)


def select_points_score_dist_hard(
    df_in_view: pd.DataFrame, limit: int, zoom: int, min_pixel_dist: int = 50
):
    """ピンを立てる地点をスコアと距離から決定する。ピン同士の最小距離が`min_pixel_dist`より大きくなるよう選択する。

    Args:
        df_in_view (pd.DataFrame): ビューポート内にある地点のデータ
        limit (int): 表示するピンの上限数
        zoom (int): ズームレベル
        min_pixel_dist (int): 表示されるピンとピンの最小距離

    Returns:
        pd.DataFrame: 選択された地点のデータ
    """
    if df_in_view.empty:
        return df_in_view

    # 地球の円周(m)
    # 地球の半径の値は(https://www.gsi.go.jp/common/000203066.pdf)を参考
    earth_circumference = 2 * math.pi * 6378137

    # 1pxあたり何mかを計算
    meters_per_pixel = earth_circumference / (256 * 2**zoom)

    # 判定距離をpx単位からWebメルカトル図法におけるm単位に変換
    min_dist_meters = min_pixel_dist * meters_per_pixel

    candidates = df_in_view.sort_values("score", ascending=False)
    selected = []
    selected_meters = []

    for _, row in candidates.iterrows():
        if len(selected) >= limit:
            break

        # 緯度経度 -> Webメルカトル図法のm座標
        mx, my = mercantile.xy(row["lon"], row["lat"])

        if not selected:
            selected.append(row)
            selected_meters.append((mx, my))
            continue

        is_far_enough = True
        for smx, smy in selected_meters:
            dist_m = math.sqrt((mx - smx) ** 2 + (my - smy) ** 2)
            if dist_m <= min_dist_meters:
                is_far_enough = False
                break

        # 選択済みの場所と十分離れた場所であるなら追加
        if is_far_enough:
            selected.append(row)
            selected_meters.append((mx, my))

    return pd.DataFrame(selected)


def select_points_score_dist_soft(
    df_in_view: pd.DataFrame,
    limit: int,
    zoom: int,
    min_pixel_dist: int,
    penalty_weight: float,
):
    """ピンを立てる地点をスコアと距離から決定する。スコアが高ければ距離が`min_pixel_dist`以下でも採用する。

    Args:
        df_in_view (pd.DataFrame): ビューポート内にある地点のデータ
        limit (int): 表示するピンの上限数
        zoom (int): ズームレベル
        min_pixel_dist (int): 表示されるピンとピンの最小距離
        penalty_weight (float): 距離制約の重み係数

    Returns:
        pd.DataFrame: 選択された地点のデータ
    """
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
        best_local_idx = np.argmax(gains)
        best_global_idx = remaining.index[best_local_idx]

        selected_indices.append(best_global_idx)
        selected_coords.append(
            (remaining.loc[best_global_idx, "mx"], remaining.loc[best_global_idx, "my"])
        )

    return df_in_view.loc[selected_indices]


def create_map(center: list[float], zoom: int, df_pins: pd.DataFrame, color: str):
    """地図インスタンスを作成してピンを配置する

    Args:
        center (list[float]): 中心座標
        zoom (int): ズームレベル
        df_pins (pd.DataFrame): 表示するピンのデータ
        color (str): ピンの色

    Returns:
        folium.Map: 地図インスタンス
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    for _, row in df_pins.iterrows():
        popup_html = f"""
        <div style="width:120px">
            <b>{row["name"]}</b><br>
            Score: {row["score"]}<br>
        </div>
        """
        folium.Marker(
            [row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row['name']} ({row['score']})",
            icon=folium.Icon(color=color, icon="cutlery", prefix="fa"),
        ).add_to(m)
    return m


if "map_state" not in st.session_state:
    st.session_state["map_state"] = {
        "center": [35.690921, 139.700258],
        "zoom": 15,
        "bounds": None,
    }


with st.sidebar:
    st.header("Global Settings")
    limit_pins = st.slider("表示上限数 (Top N)", 5, 100, 30)
    min_pixels = st.slider("最小間隔 (px)", 10, 150, 50)
    lambda_val = st.slider("距離ソフト制約の重み係数", 0.0, 10.0, 2.0, 0.1)
    st.info("ベースラインの地図を動かすと、その他の地図も追従します。")

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


df_all = generate_dummy_data()

if bounds:
    sw = bounds["_southWest"]
    ne = bounds["_northEast"]
    df_view = df_all[
        (df_all["lat"] >= sw["lat"])
        & (df_all["lat"] <= ne["lat"])
        & (df_all["lon"] >= sw["lng"])
        & (df_all["lon"] <= ne["lng"])
    ]
else:
    df_view = df_all.copy()


df_points_score_only = select_points_by_score(df_view, limit_pins)
df_points_score_dist_hard = select_points_score_dist_hard(
    df_view, limit_pins, current_zoom, min_pixels
)
df_points_score_dist_soft = select_points_score_dist_soft(
    df_view, limit_pins, current_zoom, min_pixels, lambda_val
)

collision_count_score_only = calc_collision_count(
    df_points_score_only, current_zoom, min_pixels
)
collision_count_score_dist_hard = calc_collision_count(
    df_points_score_dist_hard, current_zoom, min_pixels
)
collision_count_score_dist_soft = calc_collision_count(
    df_points_score_dist_soft, current_zoom, min_pixels
)

st.subheader("Metrics Summary")
summary_data = {
    "Logic": [
        "Score Only",
        "Score and Distance (Hard Constraint)",
        "Score and Distance (Soft Constraint)",
    ],
    "Displayed Pins": [
        len(df_points_score_only),
        len(df_points_score_dist_soft),
        len(df_points_score_dist_hard),
    ],
    "Avg Score": [
        f"{df_points_score_only['score'].mean():.2f}"
        if not df_points_score_only.empty
        else "0.00",
        f"{df_points_score_dist_hard['score'].mean():.2f}"
        if not df_points_score_dist_hard.empty
        else "0.00",
        f"{df_points_score_dist_hard['score'].mean():.2f}"
        if not df_points_score_dist_hard.empty
        else "0.00",
    ],
    "Collision Count": [
        collision_count_score_only,
        collision_count_score_dist_hard,
        collision_count_score_dist_soft,
    ],
}
st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

st.divider()

st.markdown("### Score Only")
map_score_only = create_map(current_center, current_zoom, df_points_score_only, "red")
st_folium(map_score_only, width="100%", height=400)

st.markdown("### Score and Distance (Hard Constraint)")
map_score_dist_hard = create_map(
    current_center, current_zoom, df_points_score_dist_hard, "blue"
)
st_folium(map_score_dist_hard, width="100%", height=400)

st.markdown("### Score and Distance (Soft Constraint)")
map_score_disst_soft = create_map(
    current_center, current_zoom, df_points_score_dist_soft, "green"
)
st_folium(map_score_disst_soft, width="100%", height=400)
