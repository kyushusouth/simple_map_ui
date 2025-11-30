import math

import mercantile
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm  # 進捗バー表示用 (pip install tqdm)

# ==========================================
# 1. 共通ロジック & 計算関数 (既存流用)
# ==========================================


def filter_by_viewport(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width_px: int = 375,
    height_px: int = 812,
) -> pd.DataFrame:
    """
    指定された中心座標・ズーム・画面サイズに基づいて、
    ビューポート（画面）内に含まれるデータのみを抽出する。

    Args:
        df: 全データ (lat, lonカラムが必要)
        center_lat, center_lon: 地図の中心座標
        zoom: ズームレベル
        width_px: 画面の幅 (デフォルトはiPhone相当)
        height_px: 画面の高さ
    """
    if df.empty:
        return df

    # 1. 中心座標をWebメルカトル図法のメートル座標に変換
    center_mx, center_my = mercantile.xy(center_lon, center_lat)

    # 2. 現在のズームレベルにおける 1px あたりのメートル数を計算
    # 地球の円周は約 40,075,016m
    earth_circumference = 2 * math.pi * 6378137
    meters_per_pixel = earth_circumference / (256 * 2**zoom)

    # 3. 画面の半分のサイズ（メートル）を計算
    half_width_m = (width_px / 2) * meters_per_pixel
    half_height_m = (height_px / 2) * meters_per_pixel

    # 4. バウンディングボックス（表示範囲）の定義 (メートル単位)
    min_mx = center_mx - half_width_m
    max_mx = center_mx + half_width_m
    min_my = center_my - half_height_m
    max_my = center_my + half_height_m

    # 5. データフレームの全点をメートル変換して判定
    # 高速化のため、mercantile.xy を apply するのではなく、簡易判定用に近似するか、
    # あるいはここで一括変換してしまうのが正確。評価用なら一括変換がおすすめ。

    # 判定用の一時カラムを作成
    # (注: データ数が多い場合はここがボトルネックになるので、事前にlat/lonで粗く絞るのが定石ですが、数千件ならこれで十分です)
    merc_coords = df.apply(lambda row: mercantile.xy(row["lon"], row["lat"]), axis=1)
    df_temp = df.copy()
    df_temp["mx"] = [c[0] for c in merc_coords]
    df_temp["my"] = [c[1] for c in merc_coords]

    # 範囲内フィルタリング
    df_filtered = df_temp[
        (df_temp["mx"] >= min_mx)
        & (df_temp["mx"] <= max_mx)
        & (df_temp["my"] >= min_my)
        & (df_temp["my"] <= max_my)
    ].drop(columns=["mx", "my"])  # 一時カラムは削除して元の形に戻す

    return df_filtered


def latlon_to_meters(lat, lon):
    return mercantile.xy(lon, lat)


def get_meters_per_pixel(zoom):
    earth_circumference = 2 * math.pi * 6378137
    return earth_circumference / (256 * 2**zoom)


def calculate_metrics(df, zoom, min_overlap_px):
    """評価指標を計算 (Collision Rateを追加)"""
    if len(df) < 2:
        return {
            "avg_score": df["score"].mean() if not df.empty else 0,
            "collision_rate": 0.0,
            "avg_nnd": 0.0,
        }

    # 座標変換
    m_per_px = get_meters_per_pixel(zoom)
    coords_px = []
    for _, row in df.iterrows():
        mx, my = latlon_to_meters(row["lat"], row["lon"])
        coords_px.append([mx / m_per_px, my / m_per_px])

    # 距離行列
    dist_matrix = squareform(pdist(coords_px))
    np.fill_diagonal(dist_matrix, np.inf)

    # Collision Count
    collision_count = np.sum(np.triu(dist_matrix < min_overlap_px, k=1))

    # Collision Rate (全ペア数に対する割合)
    n = len(df)
    total_pairs = n * (n - 1) / 2
    collision_rate = collision_count / total_pairs if total_pairs > 0 else 0

    # AvgNND
    nearest_dists = dist_matrix.min(axis=1)
    avg_nnd = np.mean(nearest_dists)

    return {
        "avg_score": df["score"].mean(),
        "collision_rate": collision_rate,
        "avg_nnd": avg_nnd,
        "count": n,
    }


# ==========================================
# 2. データ生成 (シナリオ対応版)
# ==========================================


def generate_scenario_data(n_samples, center_lat, center_lon, spread_sigma, seed):
    """
    spread_sigma: データの散らばり具合 (0.002=密集, 0.02=過疎 など)
    """
    np.random.seed(seed)  # 再現性のためシード固定
    data = pd.DataFrame(
        {
            "id": range(n_samples),
            "lat": np.random.normal(center_lat, spread_sigma, n_samples),
            "lon": np.random.normal(center_lon, spread_sigma, n_samples),
            "score": np.round(np.random.uniform(2.5, 5.0, n_samples), 2),
        }
    )
    return data


# ==========================================
# 3. 評価実行クラス
# ==========================================


class MapEvaluator:
    def __init__(self, limit=30, min_px=50, soft_lambda=2.0):
        self.limit = limit
        self.min_px = min_px
        self.soft_lambda = soft_lambda

    def run_baseline(self, df):
        return df.sort_values("score", ascending=False).head(self.limit)

    def run_hard(self, df, zoom):
        # ... (前回の select_points_hard_greedy の中身を実装) ...
        # ※ここでは長くなるので省略、中身はUIコードと同じものを貼り付け
        pass

    def run_soft(self, df, zoom):
        # ... (前回の select_points_soft_penalty の中身を実装) ...
        # ※ここでは長くなるので省略、中身はUIコードと同じものを貼り付け
        pass

    # 簡易実装用のダミー関数（実際は前回のロジックを使ってください）
    # テストで動くように簡易的なものを置いておきます
    def _dummy_logic(self, df):
        return df.head(self.limit)


# ==========================================
# 4. メイン評価ループ
# ==========================================


def run_experiment():
    # パラメータ設定
    SCENARIOS = [
        {"name": "Urban (Dense)", "lat": 35.69, "lon": 139.70, "sigma": 0.003},  # 新宿
        {
            "name": "Suburban (Sparse)",
            "lat": 35.65,
            "lon": 139.30,
            "sigma": 0.02,
        },  # 八王子郊外
    ]
    ZOOM_LEVELS = [13, 15, 17]
    SEEDS = range(10)  # 各設定で10回試行

    # 評価したいロジック
    # (ここでは関数ポインタやクラスメソッドを呼ぶ想定)
    # ※実際はご自身のロジック関数に置き換えてください
    LOGICS = ["Baseline", "Hard", "Soft"]

    results = []

    # 総当たり評価
    total_iters = len(SCENARIOS) * len(ZOOM_LEVELS) * len(SEEDS)

    print(f"Starting Evaluation: {total_iters} trials...")

    with tqdm(total=total_iters) as pbar:
        for scenario in SCENARIOS:
            for zoom in ZOOM_LEVELS:
                for seed in SEEDS:
                    # 1. データ生成
                    df_all = generate_scenario_data(
                        n_samples=500,  # 候補数
                        center_lat=scenario["lat"],
                        center_lon=scenario["lon"],
                        spread_sigma=scenario["sigma"],
                        seed=seed,
                    )

                    df_view = filter_by_viewport(
                        df_all,
                        center_lat=scenario["lat"],
                        center_lon=scenario["lon"],
                        zoom=zoom,
                        width_px=375,
                        height_px=812,
                    )

                    # 2. 各ロジック実行 & 計測
                    # 注意: ここではモックです。実際の select_points_... 関数を呼んでください
                    # -------------------------------------------------------

                    # [Logic A] Baseline
                    df_base = df_view.sort_values("score", ascending=False).head(30)
                    met_base = calculate_metrics(df_base, zoom, min_overlap_px=50)
                    met_base.update(
                        {
                            "Logic": "Baseline",
                            "Scenario": scenario["name"],
                            "Zoom": zoom,
                            "Seed": seed,
                        }
                    )
                    results.append(met_base)

                    # [Logic B] Hard (本来は関数呼び出し)
                    # df_hard = select_points_hard_greedy(df_view, 30, zoom, 50)
                    # met_hard = calculate_metrics(df_hard, zoom, 50)
                    # met_hard.update({"Logic": "Hard", "Scenario": scenario['name'], "Zoom": zoom, "Seed": seed})
                    # results.append(met_hard)

                    # [Logic C] Soft (本来は関数呼び出し)
                    # df_soft = select_points_soft_penalty(df_view, 30, zoom, 50, penalty_weight=2.0)
                    # met_soft = calculate_metrics(df_soft, zoom, 50)
                    # met_soft.update({"Logic": "Soft", "Scenario": scenario['name'], "Zoom": zoom, "Seed": seed})
                    # results.append(met_soft)

                    # -------------------------------------------------------
                    pbar.update(1)

    # 3. 集計
    df_results = pd.DataFrame(results)

    # ピボットテーブルで平均を集計
    summary = (
        df_results.groupby(["Scenario", "Zoom", "Logic"])[
            ["avg_score", "collision_rate", "avg_nnd"]
        ]
        .mean()
        .reset_index()
    )

    return summary


if __name__ == "__main__":
    # 実行
    # ※実際のロジック関数をimportしてから実行してください
    summary_df = run_experiment()
    print("\n=== Evaluation Summary ===")
    print(summary_df)

    # CSV保存
    summary_df.to_csv("evaluation_report.csv", index=False)
