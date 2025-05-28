import requests, pandas as pd
import folium
import argparse

def fetch_realtime_data():
    URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
    raw = requests.get(URL, timeout=10).json()
    return pd.DataFrame(raw)

def load_csv_data(csv_path):
    return pd.read_csv(csv_path)

def draw_map(df):
    # 轉型別、保留必要欄位
    cols = ["sno", "sarea", "sna", "latitude", "longitude", "total", "available_rent_bikes", "available_return_bikes", "srcUpdateTime"]
    df = df[cols].astype({"latitude":"float", "longitude":"float", "total":"int",
                        "available_rent_bikes":"float", "available_return_bikes":"float"})
    
    # 依你給的數字（可再調整）
    MIN_LNG = 121.58498      # 西邊界
    MAX_LNG = 123            # 東邊界（臨時設個 123°E，比台北再東一些）
    MIN_LAT = 25.04615       # 南邊界
    MAX_LAT = 25.08550       # 北邊界

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

    m.save("youbike_map.html")

def main():
    parser = argparse.ArgumentParser(description='繪製 YouBike 地圖')
    parser.add_argument('--mode', choices=['realtime', 'csv'], default='csv',
                      help='選擇資料模式：realtime 為即時資料，csv 為從 CSV 檔案讀取')
    parser.add_argument('--csv_path', default="interval_outputs/interval_0900.csv", help='CSV 檔案路徑（當 mode 為 csv 時需要）')
    
    args = parser.parse_args()
    
    if args.mode == 'realtime':
        df = fetch_realtime_data()
    else:
        if not args.csv_path:
            print("錯誤：使用 CSV 模式時必須提供 --csv_path 參數")
            return
        df = load_csv_data(args.csv_path)
    
    draw_map(df)

if __name__ == "__main__":
    main()


