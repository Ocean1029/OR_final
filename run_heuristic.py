import pandas as pd
import math
from typing import List, Dict, Tuple
import numpy as np

def simple_reset_heuristic(
    stations: pd.DataFrame,
    outskirts: pd.DataFrame,
    distances: pd.DataFrame,
    time_slot: str,
    K: int,
    Q: int = 28,
    L: float = 0.5,
    S: int = 60,
    T: int = 30
) -> dict[int, list[str]]:
    """
    一個更簡單的啟發式：
      - 根據當前 time_slot（"09:00"、"17:00"、"22:00"）來決定
        是做一次「外圍 → 車站」(inbound) 還是「車站 → 外圍」(outbound) 的來回。
      - 每次來回只跑一個"最優先"目標（選最缺車或最滿車的站點），
        然後回到外圍（或回到車站），檢查時間是否仍未超過 T，如果還有剩餘時間就再做一次同樣的來回。
      - Outbound：卡車直接從車站送車到 outskirt，不先收車
      - Inbound：卡車直接從 outskirt 拿車補給站點
    
    參數：
      stations:    DataFrame，必須包含欄位 id (str)、C (int)、B (int)
      outskirts:   DataFrame，必須包含欄位 id (str)、C (int)、B (int)
      distances:   DataFrame，index & columns 都是 station/outskirt 的 id，值為距離(公里)
      time_slot:   "09:00"、"17:00" 或 "22:00"
      K:           卡車數量
      Q:           卡車載重上限 (輛)
      L:           裝/卸 一輛車所需分鐘
      S:           行駛速度 (km/h)
      T:           單趟總時限 (分鐘)
    
    回傳：
      routes: dict[int, list[str]]，每輛卡車的節點走訪順序（只記錄來回的節點 id）
    """
    # 1. 決定 mode
    if time_slot not in ["09:00", "17:00", "22:00"]:
        raise ValueError("time_slot 只能是 '09:00', '17:00', '22:00'")
    mode = "outbound" if time_slot == "17:00" else "inbound"
    #    └------ 下午 5 點 才 outbound，其餘(in morning & night) 都 inbound。

    # 2. 方便查詢：把 id 設成 index
    stations = stations.set_index("id", drop=False)
    outskirts = outskirts.set_index("id", drop=False)

    # 3. 預先計算行駛時間矩陣：t_time[(i,j)] = (公里 / S) * 60 → 分鐘
    all_nodes = list(stations.index) + list(outskirts.index)
    t_time: dict[tuple[str, str], float] = {}
    for i in all_nodes:
        for j in all_nodes:
            if (i in distances.index) and (j in distances.columns):
                t_time[(i, j)] = (distances.at[i, j] / S) * 60

    # 4. 動態追蹤：station_bikes, outskirts_bikes
    station_bikes = {sid: stations.at[sid, "B"] for sid in stations.index}
    outskirts_bikes = {oid: outskirts.at[oid, "B"] for oid in outskirts.index}

    # 5. 初始化路徑與時間
    routes = {k: [] for k in range(K)}
    time_used = {k: 0.0 for k in range(K)}

    # 6. 每輛卡車分別模擬
    for k in range(K):
        # 從 depot 開始
        curr_loc = '0'  # depot
        routes[k].append(curr_loc)
        load = 0  # 卡車目前載重

        if mode == "outbound":
            # Outbound：直接從車站送車到 outskirt
            while time_used[k] < T:
                # 找 ratio 最高的站點
                ratio_list = []
                for sid in stations.index:
                    if sid == '0':  # 跳過 depot
                        continue
                    cap = stations.at[sid, "C"]
                    br = station_bikes[sid]
                    ratio = (br / cap) if cap > 0 else 0.0
                    if ratio > 0.7:  # 只考慮超載的站點
                        ratio_list.append((sid, ratio))
                
                if not ratio_list:
                    break
                    
                ratio_list.sort(key=lambda x: x[1], reverse=True)
                target_station = ratio_list[0][0]
                
                # 移動到目標站點
                travel = t_time.get((curr_loc, target_station), float("inf"))
                if time_used[k] + travel >= T:
                    break
                    
                time_used[k] += travel
                curr_loc = target_station
                routes[k].append(curr_loc)
                
                # 裝車
                cap = stations.at[curr_loc, "C"]
                br = station_bikes[curr_loc]
                threshold = math.floor(0.7 * cap)
                extra = max(0, br - threshold)
                can_load = min(extra, Q)
                
                if can_load <= 0:
                    break
                    
                time_used[k] += can_load * L
                load += can_load
                station_bikes[curr_loc] -= can_load
                
                # 找最近的 outskirt 卸車
                outskirts_ratios = []
                for oid in outskirts.index:
                    oc = outskirts.at[oid, "C"]
                    ob = outskirts_bikes[oid]
                    oratio = (ob / oc) if oc > 0 else 0.0
                    outskirts_ratios.append((oid, oratio))
                outskirts_ratios.sort(key=lambda x: x[1])  # ratio 最小 → 空間最大
                return_outskirt = outskirts_ratios[0][0]
                
                travel_back = t_time.get((curr_loc, return_outskirt), float("inf"))
                if time_used[k] + travel_back >= T:
                    break
                    
                time_used[k] += travel_back
                curr_loc = return_outskirt
                routes[k].append(curr_loc)
                
                # 卸車
                if load > 0:
                    time_used[k] += load * L
                    outskirts_bikes[curr_loc] += load
                    load = 0
                
                # 回到 depot
                travel_to_depot = t_time.get((curr_loc, '0'), float("inf"))
                if time_used[k] + travel_to_depot >= T:
                    break
                    
                time_used[k] += travel_to_depot
                curr_loc = '0'
                routes[k].append(curr_loc)
                
        else:  # inbound
            # Inbound：直接從 outskirt 拿車補給站點
            while time_used[k] < T:
                # 找有多車的 outskirt
                valid_outskirts = [(oid, outskirts_bikes[oid]) for oid in outskirts.index if outskirts_bikes[oid] > 0]
                if not valid_outskirts:
                    break
                    
                # 找最近的 outskirt
                nearest_outskirt = None
                min_dist = float('inf')
                for oid, _ in valid_outskirts:
                    dist = t_time.get((curr_loc, oid), float("inf"))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_outskirt = oid
                
                if nearest_outskirt is None or min_dist >= T - time_used[k]:
                    break
                    
                # 移動到 outskirt
                time_used[k] += min_dist
                curr_loc = nearest_outskirt
                routes[k].append(curr_loc)
                
                # 裝車
                can_load = min(outskirts_bikes[curr_loc], Q)
                if can_load <= 0:
                    break
                    
                time_used[k] += can_load * L
                load += can_load
                outskirts_bikes[curr_loc] -= can_load
                
                # 找最缺車的站點
                station_ratios = []
                for sid in stations.index:
                    if sid == '0':  # 跳過 depot
                        continue
                    cap = stations.at[sid, "C"]
                    br = station_bikes[sid]
                    ratio = (br / cap) if cap > 0 else 0.0
                    if ratio < 0.3:  # 只考慮缺車的站點
                        station_ratios.append((sid, ratio))
                
                if not station_ratios:
                    break
                    
                station_ratios.sort(key=lambda x: x[1])  # ratio 最小排前面
                target_station = station_ratios[0][0]
                
                # 移動到目標站點
                travel = t_time.get((curr_loc, target_station), float("inf"))
                if time_used[k] + travel >= T:
                    break
                    
                time_used[k] += travel
                curr_loc = target_station
                routes[k].append(curr_loc)
                
                # 卸車
                cap = stations.at[curr_loc, "C"]
                br = station_bikes[curr_loc]
                space = cap - br
                to_unload = min(load, space)
                
                if to_unload > 0:
                    time_used[k] += to_unload * L
                    station_bikes[curr_loc] += to_unload
                    load -= to_unload
                
                # 回到 depot
                travel_to_depot = t_time.get((curr_loc, '0'), float("inf"))
                if time_used[k] + travel_to_depot >= T:
                    break
                    
                time_used[k] += travel_to_depot
                curr_loc = '0'
                routes[k].append(curr_loc)

    return routes
    

def main():
    # 加上時間戳記的資料夾
    import os
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "generated_instances"
    results_dir = "optimization_results_for_heuristic/" + timestamp
    os.makedirs(results_dir, exist_ok=True)

    # 讀取距離矩陣
    distances = pd.read_csv("distance_output/distance_matrix.csv")
    
    # 定義 station 和 outskirt 的 ID
    station_ids = ['500111003', '500111013', '500111019', '500111025', '500111026',
                  '500111027', '500111056', '500111061', '500111062', '500111068', 
                  '500111069', '500111079', '500111097']
    
    # 讀取所有時間段的實例
    time_slots = ["09:00", "17:00", "22:00"]
    truck_configs = ["2trucks_30min", "3trucks_30min", "4trucks_30min"]
    
    for truck_config in truck_configs:
        for time_slot in time_slots:
            instance_dir = f"{base_dir}/{truck_config}/{time_slot}"
            if not os.path.exists(instance_dir):
                continue
                
            # 讀取該時間段的所有實例
            for instance_file in os.listdir(instance_dir):
                if not instance_file.endswith('.csv'):
                    continue
                    
                # 讀取實例數據
                instance_data = pd.read_csv(f"{instance_dir}/{instance_file}")
                
                # 分離 station 和 outskirt 數據
                stations = instance_data[instance_data['sno'].isin(station_ids)].copy()
                outskirts = instance_data[~instance_data['sno'].isin(station_ids)].copy()
                
                # 重命名 sno 為 id
                stations = stations.rename(columns={'sno': 'id'})
                outskirts = outskirts.rename(columns={'sno': 'id'})
                
                # 設定參數
                K = int(truck_config[0])  # 從配置名稱中提取卡車數量
                T = 30  # 時間限制（分鐘）
                Q = 28  # 卡車容量
                L = 0.5  # 裝卸時間（分鐘）
                S = 60  # 行駛速度（km/h）
                
                # 執行啟發式算法
                routes = simple_reset_heuristic(
                    stations=stations,
                    outskirts=outskirts,
                    distances=distances,
                    time_slot=time_slot,
                    K=K,
                    Q=Q,
                    L=L,
                    S=S,
                    T=T
                )
                
                # 計算總距離和時間
                total_distance = 0
                total_time = 0
                for k, route in routes.items():
                    for i in range(len(route)-1):
                        total_distance += distances.at[route[i], route[i+1]]
                        total_time += (distances.at[route[i], route[i+1]] / S) * 60  # 轉換為分鐘
                        if i < len(route)-2:  # 不計算最後一個站點的裝卸時間
                            total_time += L
                
                # 輸出結果
                output_file = f"{results_dir}/{truck_config}_{time_slot}_{instance_file.replace('.csv', '_result.txt')}"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"實例：{instance_file}\n")
                    f.write(f"時間段：{time_slot}\n")
                    f.write(f"卡車數量：{K}\n")
                    f.write(f"總移動距離：{total_distance:.2f} 公里\n")
                    f.write(f"總時間：{total_time:.2f} 分鐘\n\n")
                    
                    f.write("卡車路徑：\n")
                    for k, route in routes.items():
                        f.write(f"卡車 {k+1}: {' -> '.join(route)}\n")
                
                print(f"已處理 {instance_file} 並輸出結果到 {output_file}")

if __name__ == "__main__":
    main()
