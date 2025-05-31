#!/usr/bin/env python3
# visualize_station_result.py

import pandas as pd
import folium
import argparse
from pathlib import Path

def load_station_results(csv_path: str) -> pd.DataFrame:
    """
    讀取 station_results.csv，並確保必要欄位存在。
    預期欄位: station_id, balanced, final_bikes, total, latitude, longitude
    """
    df = pd.read_csv(csv_path)
    required_cols = {"station_id", "balanced", "final_bikes", "total", "latitude", "longitude"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"輸入的 CSV 欄位缺少：{missing}")
    # 確保 total 欄位存在且轉型
    if "total" not in df.columns:
        raise ValueError("輸入的 CSV 欄位缺少：total")
    df = df.astype({
        "station_id": "str",
        "balanced": "int",
        "final_bikes": "int",
        "total": "int",
        "latitude": "float",
        "longitude": "float"
    })
    return df

def draw_station_results_map(df: pd.DataFrame, output_file: str):
    """
    根據 station_results.csv 的內容，繪製 Folium 地圖並儲存為 HTML。
    - 平衡站點（balanced == 1）標記為淡綠色 (#90EE90)
    - 非平衡站點（balanced == 0）標記為淡紅色 (#FF9999)
    """
    # 若欲篩選地理範圍，可如下定義
    MIN_LNG = 121.58498      # 西邊界
    MAX_LNG = 123            # 東邊界（稍微定鬆，避免過濾掉北市東側）
    MIN_LAT = 25.04615       # 南邊界
    MAX_LAT = 25.08550       # 北邊界

    # 根據經緯度範圍過濾（可視專案需求自行調整或拿掉）
    geo_df = df[
        (df.longitude > MIN_LNG) & (df.longitude < MAX_LNG) &
        (df.latitude  > MIN_LAT ) & (df.latitude  < MAX_LAT )
    ]

    if geo_df.empty:
        raise ValueError("過濾後沒有任何站點符合指定的地理範圍！")

    # 以所有站點經緯度平均值作為地圖中心
    center_lat = geo_df.latitude.mean()
    center_lng = geo_df.longitude.mean()

    # 建立 Folium 地圖
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=14,
        tiles="CartoDB positron"
    )

    # 逐一把站點加到地圖上
    for _, row in geo_df.iterrows():
        # 使用 final_bikes 及 total 計算比例
        ratio = row.final_bikes / row.total if row.total else 0

        if ratio == 0:
            color = "#FF8C00"
        elif ratio == 1:
            color = "red"
        elif ratio < 0.3:
            color = "#FFD580"  # 淡橘（light orange）
        elif ratio < 0.7:
            color = "#90EE90"  # 淡綠（light green）
        else:
            color = "#FF9999"  # 淡紅（light red）

        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=(
                f"站點 ID: {row.station_id}<br>"
                f"最終車輛數: {row.final_bikes} / 總柱: {row.total}<br>"
                f"平衡狀態: {'是' if row.balanced == 1 else '否'}"
            )
        ).add_to(m)

    # 儲存成 HTML
    m.save(output_file)
    print(f"✔ 已將地圖輸出至：{output_file}")

def main():
    parser = argparse.ArgumentParser(description="將 Gurobi 優化後的 station_results.csv 繪製成 Folium 地圖")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="輸入的 station_results.csv 檔案路徑"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="station_results_map.html",
        help="輸出地圖 HTML 檔案路徑（預設：station_results_map.html）"
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到輸入檔案：{input_path}")

    # 讀取 station_results.csv
    df = load_station_results(str(input_path))

    # 繪製地圖並儲存
    draw_station_results_map(df, str(output_path))

if __name__ == "__main__":
    main()