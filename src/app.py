import math

import folium
import mercantile
import numpy as np
import pandas as pd
import streamlit as st
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


def select_points_by_score_dist(
    df_in_view: pd.DataFrame, limit: int, zoom: int, min_pixel_dist: int = 50
):
    """ピンを立てる地点をスコアと距離から決定する

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
            if dist_m < min_dist_meters:
                is_far_enough = False
                break

        # 選択済みの場所と十分離れた場所であるなら追加
        if is_far_enough:
            selected.append(row)
            selected_meters.append((mx, my))

    return pd.DataFrame(selected)


def select_points_by_score(df_in_view: pd.DataFrame, limit: int):
    """ピンを立てる地点をスコアから決定する

    Args:
        df_in_view (pd.DataFrame): ビューポート内にある地点のデータ
        limit (int): 表示するピンの上限数

    Returns:
        pd.DataFrame: 選択された地点のデータ
    """
    return df_in_view.sort_values("score", ascending=False).head(limit)


def create_map(
    center: list[float, float], zoom: int, df_pins: pd.DataFrame, color: str
):
    """地図インスタンスを作成してピンを配置する

    Args:
        center (list[float, float]): 中心座標
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
    min_pixels = st.slider("最小間隔 (px)", 10, 150, 50, help="Smartロジックのみ適用")
    st.info("👈 左側の地図（Baseline）を動かすと、右側も追従します。")

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

df_score_only = select_points_by_score(df_view, limit_pins)
df_score_dist = select_points_by_score_dist(
    df_view, limit_pins, current_zoom, min_pixel_dist=min_pixels
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Score Only")
    if not df_score_only.empty:
        score_1 = df_score_only["score"].mean()
        st.metric("Avg Score", f"{score_1:.2f}")
    else:
        st.metric("Avg Score", "0.00")

    m1 = create_map(current_center, current_zoom, df_score_only, "red")
    st_folium(m1, width="100%", height=500, key="map_baseline_widget")
    with st.expander("Show List (Baseline)"):
        st.dataframe(df_score_only[["name", "score"]], hide_index=True)

with col2:
    st.subheader("Score + Distance")
    if not df_score_dist.empty:
        score_2 = df_score_dist["score"].mean()
        st.metric("Avg Score", f"{score_2:.2f}")
    else:
        st.metric("Avg Score", "0.00")

    m2 = create_map(current_center, current_zoom, df_score_dist, "blue")
    st_folium(m2, width="100%", height=500, key="map_smart_widget")
    with st.expander("Show List (Smart)"):
        st.dataframe(df_score_dist[["name", "score"]], hide_index=True)
