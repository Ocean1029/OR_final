import requests, json, csv, pathlib, datetime

API = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

def fetch_snapshot():
    ts = datetime.datetime.now().astimezone().isoformat(timespec="minutes")
    r = requests.get(API, timeout=10)
    r.raise_for_status()
    records = r.json()
    filename = DATA_DIR / f"snapshot_{ts.replace(':','-')}.csv"
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "sno", "sarea", "sna", "latitude", "longitude", "total", "available_rent_bikes", "available_return_bikes", "act", "srcUpdateTime"])
        writer.writeheader()
        for row in records:
            writer.writerow({
                "timestamp": ts,
                "sno": row["sno"],
                "sarea": row["sarea"],
                "sna": row["sna"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "total": row["total"],
                "available_rent_bikes": row["available_rent_bikes"],
                "available_return_bikes": row["available_return_bikes"],
                "act": row["act"],
                "srcUpdateTime": row["srcUpdateTime"]
            })

if __name__ == "__main__":
    fetch_snapshot()
