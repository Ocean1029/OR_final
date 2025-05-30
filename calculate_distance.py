import pandas as pd
import numpy as np
import os

def calculate_distance_matrix(input_path: str, output_path: str) -> pd.DataFrame:
    """
    計算站點之間的距離矩陣並存成 CSV 檔案
    
    Args:
        input_path (str): 輸入的站點資料 CSV 檔案路徑
        output_path (str): 輸出的距離矩陣 CSV 檔案路徑
    
    Returns:
        pd.DataFrame: 距離矩陣
    """
    # --- 1. 讀取站點經緯度 ---
    stations = pd.read_csv(input_path)

    # 只留下必須欄位
    coords = stations[["sno", "latitude", "longitude"]].copy()
    coords["latitude_rad"] = np.deg2rad(coords["latitude"])
    coords["longitude_rad"] = np.deg2rad(coords["longitude"])

    lat = coords["latitude_rad"].values[:, None]    # shape (n,1)
    lon = coords["longitude_rad"].values[:, None]   # shape (n,1)

    # --- 2. 向量化 Haversine ----
    R = 6371.0  # km
    dlat = lat - lat.T           # (n,n)
    dlon = lon - lon.T
    a = np.sin(dlat/2)**2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon/2)**2
    # 數值誤差處理 (a 可能微小負值)
    a = np.clip(a, 0, 1)
    distance_matrix = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # --- 3. 轉換為 DataFrame ---
    dist_df = pd.DataFrame(distance_matrix,
                          index=coords["sno"],
                          columns=coords["sno"])

    # --- 4. 存檔 ---
    # 創建輸出目錄如果不存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dist_df.to_csv(output_path, float_format="%.4f")

    return dist_df

def main():
    """主程式入口點"""
    # 設定輸入輸出路徑
    input_path = "./interval_outputs/interval_0000.csv"
    output_path = "./distance_output/distance_matrix.csv"
    
    # 計算距離矩陣
    dist_df = calculate_distance_matrix(input_path, output_path)
    
    # 顯示結果摘要
    print(f"Distance matrix shape: {dist_df.shape}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()

