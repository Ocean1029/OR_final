#!/usr/bin/env python3
# aggregate_interval.py
#
# 需求：
#   pip install pandas pytz
# 用法範例：
#   python aggregate_interval.py \
#       --data-dir data-log/data \
#       --output avg_by_interval.csv
#
#   產出欄位：
#     interval_start  (Asia/Taipei, 30 分鐘粒度)
#     sno             站點代碼
#     sarea           行政區
#     sna             站名
#     lat, lon        座標
#     avg_total       平均樁數
#     avg_empty       平均空位
#
# 資料欄位來源見 fetch_snapshot.py 中的定義 :contentReference[oaicite:0]{index=0}
# ---------------------------------------------------------------

from pathlib import Path
import argparse
import pandas as pd

TZ = "Asia/Taipei"          # 目標時區 (GMT+8)

def load_snapshots(data_dir: Path) -> pd.DataFrame:
    """把資料夾裏所有 snapshot_*.csv 串接起來"""
    files = sorted(data_dir.glob("snapshot_*.csv"))
    if not files:
        raise FileNotFoundError(f"找不到任何 snapshot_*.csv 於 {data_dir}")
    dfs = [pd.read_csv(f) for f in files]
    dfs = [df for df in dfs if not df.empty]  # 🚨 過濾掉空的 DataFrame
    if not dfs:
        raise ValueError("所有 snapshot 檔案都為空，無法合併")
    df = pd.concat(dfs, ignore_index=True)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(TZ)
    df["interval_start"] = df["timestamp"].dt.floor("30min")

    # 確保欄位存在且型別正確
    df["act"] = pd.to_numeric(df["act"], errors="coerce")
    df = df[df["act"] == 1] 

    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df["available_return_bikes"] = pd.to_numeric(df["available_return_bikes"], errors="coerce")
    df["available_rent_bikes"] = pd.to_numeric(df["available_rent_bikes"], errors="coerce")

    return df


def aggregate(df: pd.DataFrame) -> dict:
    """依照每個 30 分鐘時段跨日平均，並分別回傳每段的 DataFrame"""
    df["time_only"] = df["interval_start"].dt.time  # 只取時間部分（例如 08:30:00）

    grouped = (
        df.groupby(["time_only", "sno"])
          .agg(
              sarea=("sarea", "first"),
              sna=("sna", "first"),
              latitude=("latitude", "first"),
              longitude=("longitude", "first"),
              total=("total", "mean"),
              available_rent_bikes=("available_rent_bikes", "mean"),
              available_rent_bikes_std=("available_rent_bikes", "std"),  # 新增標準差
              available_return_bikes=("available_return_bikes", "mean"),
              srcUpdateTime=("srcUpdateTime", "first"),  # 任選一筆保留顯示
          )
          .reset_index()
    )

    # 每個時段的結果切出來
    by_interval = {
        t.strftime("%H%M"): g[
            ["sno", "sarea", "sna", "latitude", "longitude", "total", 
             "available_rent_bikes", "available_rent_bikes_std", "available_return_bikes", "srcUpdateTime"]
        ].reset_index(drop=True)
        for t, g in grouped.groupby("time_only")
    }

    return by_interval


def main():
    parser = argparse.ArgumentParser(description="Ubike 時段平均分析器")
    parser.add_argument("--data-dir", default="data/", help="包含 snapshot_*.csv 的資料夾")
    parser.add_argument("--output-dir", default="interval_outputs", help="輸出資料夾")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_snapshots(data_dir)
    df_pre = preprocess(df_raw)
    interval_dict = aggregate(df_pre)

    for time_str, df_interval in interval_dict.items():
        filename = output_dir / f"interval_{time_str}.csv"
        df_interval.to_csv(filename, index=False, encoding="utf-8")
        print(f"✔ 輸出 {filename.name}（{len(df_interval)} 筆）")

    print(f"\n🎉 共輸出 {len(interval_dict)} 個時段檔案到 {output_dir}/")

if __name__ == "__main__":
    main()
