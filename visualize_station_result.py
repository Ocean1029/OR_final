import pandas as pd
import folium
import argparse
from pathlib import Path

def draw_station_result(csv_path: str, output_html: str = None):
    df = pd.read_csv(csv_path)
    if output_html is None:
        output_html = Path(csv_path).with_suffix('.html').name

    def color_row(row):
        # 綠色表示 balanced=1，紅色 otherwise
        return "#32CD32" if row['balanced'] == 1 else "#FF4500"

    m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()],
                   zoom_start=14, tiles="CartoDB positron")

    for _, r in df.iterrows():
        popup = (f"站點 {r.station_id}<br>"
                 f"平衡: {'✔' if r.balanced else '✘'}<br>"
                 f"最終車量: {r.final_bikes}")
        folium.CircleMarker(
            location=[r.latitude, r.longitude],
            radius=7,
            color=color_row(r),
            fill=True,
            fill_color=color_row(r),
            fill_opacity=0.85,
            popup=popup
        ).add_to(m)

    m.save(output_html)
    print(f"✅ 地圖已輸出至 {output_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize station_results.csv on a folium map")
    parser.add_argument("csv", help="Path to station_results.csv")
    parser.add_argument("--out", help="Output html filename (optional)")
    args = parser.parse_args()
    draw_station_result(args.csv, args.out)