import requests, pandas as pd
import folium

URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
raw = requests.get(URL, timeout=10).json()          # 10 秒逾時避免 hang
df  = pd.DataFrame(raw)

# 轉型別、保留必要欄位
cols = ["sno", "sarea", "sna", "latitude", "longitude", "total", "available_rent_bikes", "available_return_bikes", "act", "srcUpdateTime"]
df   = df[cols].astype({"latitude":"float", "longitude":"float", "total":"int",
                        "available_rent_bikes":"int", "available_return_bikes":"int"})
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


