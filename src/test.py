import math

import mercantile
import numpy as np
import pandas as pd


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


def select_points_soft_penalty(
    df_in_view: pd.DataFrame,
    limit: int,
    zoom: int,
    min_pixel_dist: int,
    penalty_weight: float,
):
    """
    Soft Constraint: スコアが高ければ重なりを許容する (Penalty Method)
    Maximize: Score - (lambda * OverlapDegree)
    """
    if df_in_view.empty:
        return df_in_view

    # メートル換算
    earth_circumference = 2 * math.pi * 6378137
    meters_per_pixel = earth_circumference / (256 * 2**zoom)
    limit_dist_meters = min_pixel_dist * meters_per_pixel

    # 候補リスト準備 (座標計算済みにする)
    candidates = df_in_view.copy()
    # applyで一括変換
    coords = candidates.apply(lambda x: mercantile.xy(x["lon"], x["lat"]), axis=1)
    candidates["mx"] = [c[0] for c in coords]
    candidates["my"] = [c[1] for c in coords]

    selected_indices = []
    selected_coords = []  # (mx, my) のリスト

    # Greedyループ
    for _ in range(limit):
        if len(candidates) == 0:
            break

        # まだ選ばれていない候補
        remaining = candidates.drop(selected_indices)
        if remaining.empty:
            break

        # 1個目は無条件でスコア最大
        if not selected_indices:
            best_idx = remaining["score"].idxmax()
            selected_indices.append(best_idx)
            selected_coords.append(
                (remaining.loc[best_idx, "mx"], remaining.loc[best_idx, "my"])
            )
            continue

        # 2個目以降: 実効スコア (Marginal Gain) の計算
        rem_coords = remaining[["mx", "my"]].values
        rem_scores = remaining["score"].values
        sel_coords_arr = np.array(selected_coords)

        # 距離行列計算 (Remaining x Selected)
        # 各候補(行)から、既存ピン(列)への距離
        # shape: (N_rem, N_sel)
        breakpoint()
        # (N_rem, 1, N_dim) - (1, N_sel, N_dim) -> (N_rem, N_sel, N_dim): 残っているポイントと選択済みポイントの全組み合わせで差
        diff = rem_coords[:, np.newaxis, :] - sel_coords_arr[np.newaxis, :, :]
        # (N_rem, N_sel)
        dists = np.sqrt(np.sum(diff**2, axis=2))

        # 重なり度合い (Overlap Degree)
        # 距離が閾値以下なら、近さに応じて 1.0(直撃) ~ 0.0(境界) のペナルティ
        # Penalty = max(0, 1 - dist / R)
        normalized_dists = dists / limit_dist_meters
        penalties = np.maximum(0, 1.0 - normalized_dists)

        # 各候補の総ペナルティ (和をとることで、複数個と重なるとより痛くする)
        # (N_rem,)
        total_penalties = np.sum(penalties, axis=1)

        # 実効スコア = スコア - (重み * ペナルティ)
        gains = rem_scores - (penalty_weight * total_penalties)

        # 最大ゲインを持つものを採用
        best_local_idx = np.argmax(gains)
        best_global_idx = remaining.index[best_local_idx]

        selected_indices.append(best_global_idx)
        selected_coords.append(
            (remaining.loc[best_global_idx, "mx"], remaining.loc[best_global_idx, "my"])
        )

    return df_in_view.loc[selected_indices]


df_all = generate_dummy_data()
df_result = select_points_soft_penalty(df_all, 20, 15, 50, 0.5)
breakpoint()
