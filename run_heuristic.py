import pandas as pd
import math

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
      - 每次來回只跑一個“最優先”目標（選最缺車或最滿車的站點），
        然後回到外圍（或回到車站），檢查時間是否仍未超過 T，如果還有剩餘時間就再做一次同樣的來回。
    
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
        # 每次來回，都從“起始點”開始：
        #   inbound → 起始在 一個有車的 outskirts；outbound → 起始在 一個有車的 stations
        # 所以先找一個「最多車」的起始點
        if mode == "inbound":
            valid_outskirts = [(oid, outskirts_bikes[oid]) for oid in outskirts.index if outskirts_bikes[oid] > 0]
            if not valid_outskirts:
                # 如果沒有任何 outskirts 有車，就直接不跑
                continue
            start_node = max(valid_outskirts, key=lambda x: x[1])[0]
        else:  # outbound
            valid_stations = [(sid, station_bikes[sid]) for sid in stations.index]
            # 只要有站點，即使車量=0，我們也先挑最大的 ratio，但 ratio=0 意味著不滿足條件，就得換下一站。
            # 先按 ratio 排序：ratio = bike / capacity
            ratio_list = []
            for sid, b in valid_stations:
                cap = stations.at[sid, "C"]
                ratio = (b / cap) if cap > 0 else 0.0
                ratio_list.append((sid, ratio))
            # ratio 最大往前。若 ratio<=0.7 都沒有車要載，這次卡車就不跑了
            ratio_list.sort(key=lambda x: x[1], reverse=True)
            if ratio_list[0][1] <= 0.7:
                # 沒有任何站點 ratio > 0.7，就不跑
                continue
            start_node = ratio_list[0][0]

        # 第一趟“來回”
        curr_loc = start_node
        routes[k].append(curr_loc)
        load = 0  # 卡車目前載重

        # 先做起始點的「載／卸動作」
        if mode == "inbound":
            # 從 curr_loc (outskirt) 裝最多車到卡車
            can_load = min(outskirts_bikes[curr_loc], Q)
            load += can_load
            outskirts_bikes[curr_loc] -= can_load
            time_used[k] += can_load * L
        else:
            # 從 curr_loc (station) 載走多餘車到卡車，只要 ratio > 0.7 的多餘就載
            cap = stations.at[curr_loc, "C"]
            b = station_bikes[curr_loc]
            threshold = math.floor(0.7 * cap)
            extra = max(0, b - threshold)
            can_load = min(extra, Q)
            load += can_load
            station_bikes[curr_loc] -= can_load
            time_used[k] += can_load * L

        # 接著找「這趟來回的目標」：若 inbound → 到最缺車的 station；若 outbound → 到最滿車的 station 再載到 outskirts
        if mode == "inbound":
            # 找 ratio 最低 (bikes/capacity) 的 station
            station_ratios = []
            for sid in stations.index:
                cap = stations.at[sid, "C"]
                br = station_bikes[sid]
                ratio = (br / cap) if cap > 0 else 0.0
                station_ratios.append((sid, ratio))
            station_ratios.sort(key=lambda x: x[1])  # ratio 小的排前面
            target_station = station_ratios[0][0]
            # 計算行駛時間
            travel = t_time.get((curr_loc, target_station), float("inf"))
            if time_used[k] + travel >= T:
                # 時間不夠，結束在 outskirts (不跑這趟到站)
                continue
            time_used[k] += travel
            curr_loc = target_station
            routes[k].append(curr_loc)

            # 「卸車」：把 load 輛卸到 target_station（或只卸到它的容量上限）
            cap = stations.at[curr_loc, "C"]
            br = station_bikes[curr_loc]
            space = cap - br
            to_unload = min(load, space)
            time_used[k] += to_unload * L
            station_bikes[curr_loc] += to_unload
            load -= to_unload

            # 這趟 inbound 完成，回到一個「有空間收車」的 outskirts
            # 找最少 ratio (bikes/capacity) 的 outskirts 當作回程終點
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
                # 時間不夠，只停在當前站，結束
                continue
            time_used[k] += travel_back
            curr_loc = return_outskirt
            routes[k].append(curr_loc)

            # 把剩下的 load（如果還有）卸到 outskirts
            if load > 0:
                time_used[k] += load * L
                outskirts_bikes[curr_loc] += load
                load = 0

        else:  # mode == "outbound"
            # 找 ratio 最高 (bikes/capacity) 的 station (已經在 start_node，但可能要再確認一次)
            br_list = []
            for sid in stations.index:
                cap = stations.at[sid, "C"]
                br = station_bikes[sid]
                ratio = (br / cap) if cap > 0 else 0.0
                br_list.append((sid, ratio))
            br_list.sort(key=lambda x: x[1], reverse=True)
            target_station = br_list[0][0]
            # 如果起始站不是這個 target，先開過去
            if curr_loc != target_station:
                travel = t_time.get((curr_loc, target_station), float("inf"))
                if time_used[k] + travel >= T:
                    continue
                time_used[k] += travel
                curr_loc = target_station
                routes[k].append(curr_loc)

            # 載車：已在 start_node 載了部分，現在卸載到 outskirts
            # 先找「可以放得下車數最多」的 outskirts：ratio 最小
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
                continue
            time_used[k] += travel_back
            curr_loc = return_outskirt
            routes[k].append(curr_loc)

            # 把 load 載到 outskirts
            if load > 0:
                time_used[k] += load * L
                outskirts_bikes[curr_loc] += load
                load = 0

        # 7. 如果跑完一次來回，還有剩餘時間，就再跑一次「同樣模式」的來回
        while time_used[k] < T - 1e-6:  # 給一點浮點誤差空間
            # 判斷剩餘時間是否足夠再做一次完整來回(粗略檢查：至少要有「去程 + 回程 + 一次裝卸」的時間)
            # 假設「典型裝卸一台車時間 = L * (Q/2)」（以大約載一半為粗略估計）
            avg_unload = (Q / 2) * L
            # 一趟去程＋回程：先假設最遠距離 max(distances.values)
            est_max_dist = distances.max().max()
            est_trip_time = (est_max_dist / S) * 60 * 2 + avg_unload
            if time_used[k] + est_trip_time >= T:
                break  # 沒時間再跑一趟

            # 跑下一趟：從當前 curr_loc (必定在 outskirts) 再做一次
            # inbound → from outskirts 拿車到 station → 回 outskirts
            # outbound → from station (若 curr_loc 在 outskirts 先去 station) → 回 outskirts

            if mode == "inbound":
                # 確認 curr_loc 在 outskirts，如果不在就先換到 ratio 最大的 outskirts
                if curr_loc not in outskirts.index:
                    valid_outskirts = [(oid, outskirts_bikes[oid]) for oid in outskirts.index if outskirts_bikes[oid] > 0]
                    if not valid_outskirts:
                        break
                    new_start = max(valid_outskirts, key=lambda x: x[1])[0]
                    travel0 = t_time.get((curr_loc, new_start), float("inf"))
                    if time_used[k] + travel0 >= T:
                        break
                    time_used[k] += travel0
                    curr_loc = new_start
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
                    cap = stations.at[sid, "C"]
                    br = station_bikes[sid]
                    ratio = (br / cap) if cap > 0 else 0.0
                    station_ratios.append((sid, ratio))
                station_ratios.sort(key=lambda x: x[1])
                target_station = station_ratios[0][0]

                travel1 = t_time.get((curr_loc, target_station), float("inf"))
                if time_used[k] + travel1 >= T:
                    break
                time_used[k] += travel1
                curr_loc = target_station
                routes[k].append(curr_loc)

                # 卸車到 station
                cap = stations.at[curr_loc, "C"]
                br = station_bikes[curr_loc]
                space = cap - br
                to_unload = min(load, space)
                time_used[k] += to_unload * L
                station_bikes[curr_loc] += to_unload
                load -= to_unload

                # 回到最空的 outskirts
                outskirts_ratios = []
                for oid in outskirts.index:
                    oc = outskirts.at[oid, "C"]
                    ob = outskirts_bikes[oid]
                    oratio = (ob / oc) if oc > 0 else 0.0
                    outskirts_ratios.append((oid, oratio))
                outskirts_ratios.sort(key=lambda x: x[1])
                return_outskirt = outskirts_ratios[0][0]
                travel2 = t_time.get((curr_loc, return_outskirt), float("inf"))
                if time_used[k] + travel2 >= T:
                    break
                time_used[k] += travel2
                curr_loc = return_outskirt
                routes[k].append(curr_loc)

                # 把車卸回 outskirts
                if load > 0:
                    time_used[k] += load * L
                    outskirts_bikes[curr_loc] += load
                    load = 0

            else:  # outbound
                # 先確保 curr_loc 在 station，如果不在就先到 ratio 最大的 station
                if curr_loc not in stations.index:
                    # 找 ratio 最大的站點
                    ratio_list = []
                    for sid in stations.index:
                        cap = stations.at[sid, "C"]
                        br = station_bikes[sid]
                        ratio = (br / cap) if cap > 0 else 0.0
                        ratio_list.append((sid, ratio))
                    ratio_list.sort(key=lambda x: x[1], reverse=True)
                    if ratio_list[0][1] <= 0.7:
                        break
                    new_start = ratio_list[0][0]
                    travel0 = t_time.get((curr_loc, new_start), float("inf"))
                    if time_used[k] + travel0 >= T:
                        break
                    time_used[k] += travel0
                    curr_loc = new_start
                    routes[k].append(curr_loc)

                # 從 curr_loc (station) 載車
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

                # 選擇最能放車的 outskirts
                outskirts_ratios = []
                for oid in outskirts.index:
                    oc = outskirts.at[oid, "C"]
                    ob = outskirts_bikes[oid]
                    oratio = (ob / oc) if oc > 0 else 0.0
                    outskirts_ratios.append((oid, oratio))
                outskirts_ratios.sort(key=lambda x: x[1])
                return_outskirt = outskirts_ratios[0][0]

                travel1 = t_time.get((curr_loc, return_outskirt), float("inf"))
                if time_used[k] + travel1 >= T:
                    break
                time_used[k] += travel1
                curr_loc = return_outskirt
                routes[k].append(curr_loc)

                # 把車卸到 outskirts
                if load > 0:
                    time_used[k] += load * L
                    outskirts_bikes[curr_loc] += load
                    load = 0

        # 這輛卡車結束，記錄完路徑，跳到下一輛

    return routes