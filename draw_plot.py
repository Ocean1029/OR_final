import requests, pandas as pd
import folium
import argparse
from pathlib import Path

def fetch_realtime_data():
    URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
    raw = requests.get(URL, timeout=10).json()
    return pd.DataFrame(raw)

def load_csv_data(csv_path):
    return pd.read_csv(csv_path)

def draw_all_intervals(folder="print_sth", output_folder="map_outputs"):
    folder = Path(folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    for file in sorted(folder.glob("interval_*.csv")):
        time_str = file.stem.split("_")[1]
        out_file = output_folder / f"map_{time_str}.html"
        print(f"處理 {file.name} ...")
        df = pd.read_csv(file)
        draw_map(df, str(out_file))
        print(f"✔ 已輸出 {out_file.name}")

    print("✔ 所有時段地圖已輸出至", output_folder)


def draw_map(df, output_file="youbike_map.html"):
    # 轉型別、保留必要欄位
    cols = ["sno", "sarea", "sna", "latitude", "longitude", "total", "available_rent_bikes", "available_return_bikes", "srcUpdateTime"]
    df = df[cols].astype({"latitude":"float", "longitude":"float", "total":"int",
                        "available_rent_bikes":"float", "available_return_bikes":"float"})
    
    # 依你給的數字（可再調整）
    MIN_LNG = 121.591256      # 西邊界
    MAX_LNG = 123            # 東邊界（臨時設個 123°E，比台北再東一些）
    MIN_LAT = 25.04615       # 南邊界
    MAX_LAT = 25.062016       # 北邊界

    # 過濾掉不在指定範圍的資料
    df = df[(df.longitude > MIN_LNG) & (df.longitude < MAX_LNG) &
            (df.latitude > MIN_LAT) & (df.latitude < MAX_LAT)]

    # 以台北車站為中心
    m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()],
                   zoom_start=14,
                   tiles="CartoDB positron")   # 乾淨底圖

    for _, r in df.iterrows():
        ratio = r.available_rent_bikes / r.total if r.total else 0

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
            location=[r.latitude, r.longitude],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=(f"{r.sna}（{r.sarea}）<br>"
                   f"可借 {r.available_rent_bikes} / 總柱 {r.total}<br>"
                   f"可還 {r.available_return_bikes}<br>"
                   f"更新 {r.srcUpdateTime}")
        ).add_to(m)

    m.save(output_file)

def main():
    parser = argparse.ArgumentParser(description='繪製 YouBike 地圖')
    parser.add_argument('--mode', choices=['realtime', 'csv'], default='csv',
                      help='選擇資料模式：realtime 為即時資料，csv 為從 CSV 檔案讀取')
    parser.add_argument('--output', default='youbike_map.html',
                      help='輸出檔案路徑（預設：youbike_map.html）')
    
    args = parser.parse_args()
    
    if args.mode == 'realtime':
        df = fetch_realtime_data()
        draw_map(df, args.output)
    else:
        draw_all_intervals()
    

if __name__ == "__main__":
    main()


