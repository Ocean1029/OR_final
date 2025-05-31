import pandas as pd
import math
from typing import List, Dict, Tuple
import numpy as np
import os

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
) -> Tuple[dict[int, list[str]], Dict[str, int]]:
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
      routes_plain: dict[int, list[str]]，每輛卡車的節點走訪順序（只記錄來回的節點 id）
      routes_fmt: dict[int, list[str]]，每輛卡車的節點走訪順序（每個節點含取/放車資訊）
      start_loads: dict[int, int]，每輛卡車離開 depot 時的載重
      final_bikes: dict[str, int]，每個站點/外圍的剩餘車輛數
    """
    # 1. 決定 mode
    if time_slot not in ["09:00", "17:00", "22:00"]:
        raise ValueError("time_slot 只能是 '09:00', '17:00', '22:00'")
    mode = "inbound" if time_slot == "17:00" else "outbound"
    #    └------ 下午 5 點 才 outbound，其餘(in morning & night) 都 inbound。

    # 2. 方便查詢：把 id 設成 index

    stations['B'] = stations['available_rent_bikes'].round().astype(int)
    stations['C'] = stations['total'].astype(int)
    outskirts['B'] = outskirts['available_rent_bikes'].round().astype(int)
    outskirts['C'] = outskirts['total'].astype(int)

    stations = stations[['id', 'C', 'B', 'latitude', 'longitude']]
    outskirts = outskirts[['id', 'C', 'B', 'latitude', 'longitude']]

    # 篩選 stations 的經緯度
    # MIN_LNG = 121.591256      # 西邊界   原本是：121.58498
    MIN_LNG = 121.58498      # 西大邊界 
    MAX_LNG = 123            # 東邊界（臨時設個 123°E，比台北再東一些）
    MIN_LAT = 25.04615       # 南邊界
    # MAX_LAT = 25.062016       # 北邊界 隨便改的 原本是：25.08550
    MAX_LAT = 25.08550       # 北大邊界
    
    # 篩選站點
    stations = stations[(stations.longitude > MIN_LNG) & (stations.longitude < MAX_LNG) &
                       (stations.latitude > MIN_LAT) & (stations.latitude < MAX_LAT)]

    # 篩選外圍
    outskirts = outskirts[(outskirts.longitude > MIN_LNG) & (outskirts.longitude < MAX_LNG) &
                        (outskirts.latitude > MIN_LAT) & (outskirts.latitude < MAX_LAT)]
    print(f"篩選後的車站數量: {len(stations)}, 外圍數量: {len(outskirts)}")
    # 加入 depot (0) df
    depot = pd.DataFrame({
        'id': ['0'],
        'C': [100],  # 假設 depot 容量很大
        'B': [50],
        'latitude': [25.05326],
        'longitude': [121.60656]
    })
    stations = pd.concat([depot, stations], ignore_index=True)

    # 重新確保 id 為字串，並設為 index
    stations['id'] = stations['id'].astype(str)
    stations = stations.set_index('id', drop=False)

    outskirts['id'] = outskirts['id'].astype(str)
    outskirts = outskirts.set_index('id', drop=False)

    # === 距離 → 時間矩陣 (分鐘) ===
    # 確保距離矩陣的 index/columns 皆為字串
    distances.index = distances.index.astype(str)
    distances.columns = distances.columns.astype(str)

    all_nodes = list(stations.index) + list(outskirts.index)
    t_time: dict[tuple[str, str], float] = {}
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                t_time[(i, j)] = 0.0
            elif (i in distances.index) and (j in distances.columns):
                t_time[(i, j)] = (distances.at[i, j] / S) * 60
    # print("t_time sample:", list(t_time.items()))

    # 4. 動態追蹤：station_bikes, outskirts_bikes
    station_bikes = {sid: stations.at[sid, "B"] for sid in stations.index}
    outskirts_bikes = {oid: outskirts.at[oid, "B"] for oid in outskirts.index}

    # 5. 初始化路徑與時間
    routes_plain = {k: [] for k in range(K)}
    routes_fmt = {k: [] for k in range(K)}
    time_used = {k: 0.0 for k in range(K)}
    start_loads = {}

    # 6. 每輛卡車分別模擬
    for k in range(K):
        # 從 depot 開始
        curr_loc = '0'  # depot
        routes_plain[k].append(curr_loc)
        routes_fmt[k].append(f"{curr_loc}()")
        load = 0  # 卡車目前載重
        start_load_k = None

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
                    print(f"卡車 {k+1} 時間超過限制，停止行動，當前時間 {time_used[k]} 分鐘")
                    break
                time_used[k] += travel
                curr_loc = target_station
                routes_plain[k].append(curr_loc)
                # 裝車
                cap = stations.at[curr_loc, "C"]
                br = station_bikes[curr_loc]
                threshold = math.floor(0.7 * cap)
                extra = max(0, br - threshold)
                can_load = min(extra, Q)
                pick = can_load if can_load > 0 else 0
                drop = 0
                if can_load <= 0:
                    # 無可裝車，仍然記錄動作
                    routes_fmt[k].append(f"{curr_loc}()")
                    break
                time_used[k] += can_load * L
                load += can_load
                station_bikes[curr_loc] -= can_load
                # 記錄格式化動作
                action_str = []
                if pick > 0:
                    action_str.append(f"-{pick}")
                if drop > 0:
                    action_str.append(f"+{drop}")
                routes_fmt[k].append(f"{curr_loc}({''.join(action_str)})")
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
                    routes_plain[k].append(return_outskirt)
                    routes_fmt[k].append(f"{return_outskirt}()")
                    break
                time_used[k] += travel_back
                curr_loc = return_outskirt
                routes_plain[k].append(curr_loc)
                # 卸車
                pick2 = 0
                drop2 = load if load > 0 else 0
                if load > 0:
                    time_used[k] += load * L
                    outskirts_bikes[curr_loc] += load
                    load = 0
                # 記錄格式化動作
                action_str2 = []
                if pick2 > 0:
                    action_str2.append(f"-{pick2}")
                if drop2 > 0:
                    action_str2.append(f"+{drop2}")
                routes_fmt[k].append(f"{curr_loc}({''.join(action_str2)})")
                # 回到 depot
                travel_to_depot = t_time.get((curr_loc, '0'), float("inf"))
                if time_used[k] + travel_to_depot >= T:
                    routes_plain[k].append('0')
                    routes_fmt[k].append(f"0()")
                    break
                time_used[k] += travel_to_depot
                curr_loc = '0'
                routes_plain[k].append(curr_loc)
                routes_fmt[k].append(f"{curr_loc}()")
                # 記錄 start_load
                if start_load_k is None and len(routes_plain[k]) > 1:
                    # 第一次離開 depot 後的載重
                    # 在 outbound, 取車後才有載重, 但我們定義 start_load 為第一次離開 depot 之後的載重
                    # 這裡的 load 其實已經歸零, 所以需要記錄
                    pass  # for outbound, start_load is after pick, but since we always start from 0, we can set after the first pick
            # 記錄 start_load
            if len(routes_plain[k]) > 1:
                # start_load 是第一次離開 depot 後的載重, 這裡設為 Q 或第一次 pick 數
                # 其實就是第一次 pick 的數量
                # 找到第一個有 -X 的步驟
                for s in routes_fmt[k]:
                    if "(" in s and "-" in s:
                        try:
                            load_val = int(s.split("(-")[1].split(")")[0])
                            start_load_k = load_val
                        except Exception:
                            start_load_k = Q
                        break
                if start_load_k is None:
                    start_load_k = 0
            else:
                start_load_k = 0
            start_loads[k] = start_load_k
        else:
            # inbound
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
                routes_plain[k].append(curr_loc)
                # 裝車
                can_load = min(outskirts_bikes[curr_loc], Q)
                pick = can_load if can_load > 0 else 0
                drop = 0
                if can_load <= 0:
                    routes_fmt[k].append(f"{curr_loc}()")
                    break
                time_used[k] += can_load * L
                load += can_load
                outskirts_bikes[curr_loc] -= can_load
                # 記錄格式化動作
                action_str = []
                if pick > 0:
                    action_str.append(f"-{pick}")
                if drop > 0:
                    action_str.append(f"+{drop}")
                routes_fmt[k].append(f"{curr_loc}({''.join(action_str)})")
                # 記錄 start_load
                if start_load_k is None:
                    start_load_k = load
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
                    print(f"卡車 {k+1} 時間超過限制，停止行動，當前時間 {time_used[k]} 分鐘")
                    routes_plain[k].append(target_station)
                    routes_fmt[k].append(f"{target_station}()")
                    break
                time_used[k] += travel
                curr_loc = target_station
                routes_plain[k].append(curr_loc)
                # 卸車
                cap = stations.at[curr_loc, "C"]
                br = station_bikes[curr_loc]
                space = cap - br
                to_unload = min(load, space)
                pick2 = 0
                drop2 = to_unload if to_unload > 0 else 0
                if to_unload > 0:
                    time_used[k] += to_unload * L
                    station_bikes[curr_loc] += to_unload
                    load -= to_unload
                # 記錄格式化動作
                action_str2 = []
                if pick2 > 0:
                    action_str2.append(f"-{pick2}")
                if drop2 > 0:
                    action_str2.append(f"+{drop2}")
                routes_fmt[k].append(f"{curr_loc}({''.join(action_str2)})")
                # 回到 depot
                travel_to_depot = t_time.get((curr_loc, '0'), float("inf"))
                if time_used[k] + travel_to_depot >= T:
                    routes_plain[k].append('0')
                    routes_fmt[k].append(f"0()")
                    break
                time_used[k] += travel_to_depot
                curr_loc = '0'
                routes_plain[k].append(curr_loc)
                routes_fmt[k].append(f"{curr_loc}()")
            if start_load_k is None:
                start_load_k = 0
            start_loads[k] = start_load_k

    # merge station and outskirt bike counts so the caller sees everything
    final_bikes = {**station_bikes, **outskirts_bikes}
    print(routes_plain)
    return routes_plain, routes_fmt, start_loads, final_bikes
    

def main():
    # Add timestamped results directory
    import os
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "generated_instances"
    results_dir = "optimization_results_for_heuristic/" + timestamp
    os.makedirs(results_dir, exist_ok=True)

    # Read distance matrix (ensure index_col=0 so IDs match)
    distances = pd.read_csv("distance_output/distance_matrix.csv", index_col=0)

    # Define station IDs for identifying stations vs outskirts
    station_ids = [
        500111003, 500111013, 500111019, 500111025, 500111026,
        500111027, 500111056, 500111061, 500111062, 500111068,
        500111069, 500111079, 500111097, 500111076
    ]

    # Parameters for truck capacity, load/unload time, and speed
    Q = 28  # truck capacity (bikes)
    L = 0.5  # load/unload time per bike (minutes)
    S = 60  # travel speed (km/h)

    # Iterate over all scenarios in generated_instances
    for scenario_dir in sorted(os.listdir(base_dir)):
        scenario_path = os.path.join(base_dir, scenario_dir)
        if not os.path.isdir(scenario_path):
            continue

        # Parse truck count K and time limit T from scenario_dir (e.g., "2trucks_30min")
        parts = scenario_dir.split('_')
        K = int(parts[0].replace('trucks', ''))
        T = int(parts[1].replace('min', ''))

        # Map directory names to time_slot values expected by heuristic
        time_slot_mapping = {
            'morning_9am': '09:00',
            'evening_5pm': '17:00',
            'night_10pm': '22:00'
        }

        # Iterate over each time slot directory (e.g., "morning_9am", "evening_5pm", "night_10pm")
        for time_slot_dir in sorted(os.listdir(scenario_path)):
            if time_slot_dir not in time_slot_mapping:
                continue
            time_slot = time_slot_mapping[time_slot_dir]
            time_path = os.path.join(scenario_path, time_slot_dir)
            if not os.path.isdir(time_path):
                continue

            # Process each instance file
            for instance_file in sorted(os.listdir(time_path)):
                if not instance_file.endswith('.csv'):
                    continue

                # 1. Extract instance name
                instance_name = instance_file.replace('.csv', '')

                instance_path = os.path.join(time_path, instance_file)
                # Read instance data
                instance_data = pd.read_csv(instance_path)

                # Separate station rows and outskirt rows
                print(f"Processing {instance_file} for scenario {scenario_dir} at time slot {time_slot}...")
                stations_df = instance_data[instance_data['sno'].isin(station_ids)].copy()
                outskirts_df = instance_data[~instance_data['sno'].isin(station_ids)].copy()

                # Rename 'sno' to 'id' for both DataFrames
                stations_df = stations_df.rename(columns={'sno': 'id'})
                outskirts_df = outskirts_df.rename(columns={'sno': 'id'})
                print(stations_df)
                # ---- baseline: initial balanced station count (no dispatch) ----
                all_station_df = pd.concat([stations_df, outskirts_df], ignore_index=True)
                initial_balanced = sum(
                    1
                    for _, row in all_station_df.iterrows()
                    if 0.3 * row['total'] <= round(row['available_rent_bikes']) <= 0.7 * row['total']
                )
                # Call the heuristic function
                routes_plain, routes_fmt, start_loads, final_station_bikes = simple_reset_heuristic(
                    stations=stations_df,
                    outskirts=outskirts_df,
                    distances=distances,
                    time_slot=time_slot,
                    K=K,
                    Q=Q,
                    L=L,
                    S=S,
                    T=T
                )

                # Calculate total distance and total time for this instance (using plain route)
                total_distance = 0.0
                total_time = 0.0
                for k, route in routes_plain.items():
                    # Sum pairwise distances and travel times + loading times
                    for i in range(len(route) - 1):
                        src = route[i]
                        dst = route[i + 1]
                        dist_km = distances.at[src, dst]
                        total_distance += dist_km
                        # Travel time in minutes
                        travel_time = (dist_km / S) * 60
                        total_time += travel_time
                        # Loading/unloading occurs at every intermediate stop (not final)
                        if i < len(route) - 2:
                            total_time += L

                # Prepare output directory for this scenario/time_slot/instance
                out_dir = os.path.join(results_dir, scenario_dir, time_slot_dir, instance_name)
                os.makedirs(out_dir, exist_ok=True)

                # ---------- Generate station_results.csv ----------
                all_station_df = pd.concat([stations_df, outskirts_df], ignore_index=True)
                station_results = []
                for sid in all_station_df['id']:
                    cap = int(all_station_df.loc[all_station_df['id'] == sid, 'total'].values[0])
                    # Use original 'total' column from instance_data to get capacity
                    initial_bikes = int(all_station_df.loc[all_station_df['id'] == sid, 'available_rent_bikes'].round().astype(int).values[0])
                    final_bikes = final_station_bikes.get(sid, initial_bikes)
                    balanced = 1 if (0.3 * cap) <= final_bikes <= (0.7 * cap) else 0
                    lat = all_station_df.loc[all_station_df['id'] == sid, 'latitude'].values[0]
                    lng = all_station_df.loc[all_station_df['id'] == sid, 'longitude'].values[0]
                    station_results.append({
                        "station_id": sid,
                        "balanced": balanced,
                        "final_bikes": final_bikes,
                        "latitude": lat,
                        "longitude": lng
                    })
                pd.DataFrame(station_results).to_csv(os.path.join(out_dir, "station_results.csv"), index=False)
                final_balanced = sum(r['balanced'] for r in station_results)


                # Output file name: instance_<n>_result.txt
                output_file = os.path.join(
                    out_dir,
                    instance_file.replace('.csv', '_result.txt')
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Instance: {instance_file}\n")
                    f.write(f"Time slot: {time_slot}\n")
                    f.write(f"Truck count: {K}\n")
                    f.write(f"Total distance: {total_distance:.2f} km\n")
                    f.write(f"Total time: {total_time:.2f} minutes\n")
                    f.write(f"Balanced before (baseline): {initial_balanced}\n")
                    f.write(f"Objective (balanced stations): {final_balanced}\n\n")
                    f.write("Truck routes:\n")
                    for k in sorted(routes_fmt.keys()):
                        start_load = start_loads.get(k, 0)
                        f.write(f"Truck {k} (start_load={start_load}): " +
                                " -> ".join(routes_fmt[k]) + "\n")

                print(f"Processed {instance_file} for {scenario_dir}/{time_slot}, output to {output_file}")


if __name__ == "__main__":
    main()
