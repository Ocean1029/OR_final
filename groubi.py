# youbike_rebalancing_gurobi.py
# Author: 煥軒  (NTU IM)
# Description: Multi-truck bike rebalancing MILP formulated in Gurobi

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os

# ========= 1. 資料讀取與參數設定 ========= #

def load_data(station_file: str, distance_file: str):
    """讀入站點容量與初始車輛數，以及站點間距離矩陣。"""
    # 讀取站點資料
    stations = pd.read_csv(station_file)
    
    # 計算初始車輛數，並四捨五入到最接近的整數
    stations['B'] = stations['available_rent_bikes'].round().astype(int)
    stations['C'] = stations['total'].astype(int)
    
    # 重命名 sno 為 id
    stations = stations.rename(columns={'sno': 'id'})
    
    # 保留所有需要的欄位
    stations = stations[['id', 'C', 'B', 'latitude', 'longitude']]
    
    # 篩選 stations 的經緯度
    MIN_LNG = 121.58498      # 西邊界   原本是：121.58498
    MAX_LNG = 123            # 東邊界（臨時設個 123°E，比台北再東一些）
    MIN_LAT = 25.04615       # 南邊界
    MAX_LAT = 25.057       # 北邊界 隨便改的 原本是：25.08550
    
    # 篩選站點
    stations = stations[(stations.longitude > MIN_LNG) & (stations.longitude < MAX_LNG) &
                       (stations.latitude > MIN_LAT) & (stations.latitude < MAX_LAT)]

    # 加入 depot (0) df
    depot = pd.DataFrame({
        'id': ['0'],
        'C': [100],  # 假設 depot 容量很大
        'B': [50],
        'latitude': [25.05326],
        'longitude': [121.60656]
    })
    stations = pd.concat([depot, stations], ignore_index=True)
    # 確保站點 ID 為字串類型
    
    # 讀取距離矩陣
    dist_df = pd.read_csv(distance_file, index_col=0)
    
    # 確保距離矩陣的索引和欄位名稱與站點 ID 一致
    dist_df.index = dist_df.index.astype(str)
    dist_df.columns = dist_df.columns.astype(str)
    
    # 確保站點 ID 的格式一致
    stations['id'] = stations['id'].astype(str)

    # 使用站點 ID 來選擇距離矩陣
    station_ids = stations['id'].tolist()
    dist_df = dist_df.loc[station_ids, station_ids]
    
    # 確保距離矩陣包含所有需要的站點
    stations_set = set(stations['id'])
    dist_stations = set(dist_df.index)
    
    # print(f"站點 ID 的前5個元素: {sorted(list(stations_set))[:5]}")
    # print(f"距離矩陣站點的前5個元素: {sorted(list(dist_stations))[:5]}")
    # print(f"站點數量: {len(stations_set)}")
    # print(f"距離矩陣站點數量: {len(dist_stations)}")
    
    # 檢查是否有任何匹配的站點
    matching_stations = stations_set.intersection(dist_stations)
    # print(f"匹配的站點數量: {len(matching_stations)}")
    
    if len(matching_stations) == 0:
        raise ValueError("錯誤：距離矩陣中沒有任何與站點資料匹配的站點 ID。請檢查距離矩陣檔案是否正確。")
    
    missing_stations = stations_set - dist_stations
    
    if missing_stations:
        print("enter missing")
        avg_dist = dist_df.values.mean() # TODO 這裡 dist 會超大
        # 先補 column
        for station in missing_stations:
            dist_df[station] = avg_dist
        # 再補 row
        for station in missing_stations:
            dist_df.loc[station] = avg_dist
        # 重新排序 row/column
        dist_df = dist_df.loc[station_ids, station_ids]

    return stations, dist_df

def build_scenario(scenario_id: int):
    """根據 Table~\ref{scenario} 回傳 (K, T)。"""
    scenario_map = {
        # sid: (trucks, time_window)
        1: (2, 30), 2: (2, 30), 3: (2, 30),
        4: (2, 60), 5: (2, 60), 6: (2, 60),
        7: (4, 30), 8: (4, 30), 9: (4, 30),
        10: (4, 60), 11: (4, 60), 12: (4, 60),
        13: (6, 30), 14: (6, 30), 15: (6, 30),
        16: (6, 60), 17: (6, 60), 18: (6, 60)
    }
    return scenario_map[scenario_id]

# ========= 2. 模型建構 ========= #

def build_model(stations: pd.DataFrame,
                distances: pd.DataFrame,
                K: int,
                T: int,
                Q: int = 28,
                L: float = 0.5,   # 實際裝卸時間/車 (分鐘)
                S: int = 40     # 車子的時速
               ) -> gp.Model:

    # 檢查 stations id 是 str
    if not all(isinstance(i, str) for i in stations["id"]):
        raise ValueError("站點 ID 必須是字串類型，請檢查輸入資料。")
    # 檢查 distances index 和 columns 是 str
    if not all(isinstance(i, str) for i in distances.index) or \
       not all(isinstance(i, str) for i in distances.columns):
        raise ValueError("距離矩陣的索引和欄位名稱必須是字串類型，請檢查輸入資料。")
    
    
    N_with_depot = stations["id"].tolist()  # 包含 depot (0)
    N = N_with_depot[1:]  # 不包含 depot (0)
    print(f"\n建立模型：")
    print(f"- 站點數：{len(N)}")
    print(f"- 卡車數：{K}")
    print(f"- 時間限制：{T} 分鐘")
    
    d = {}
    for i in N_with_depot:
        for j in N_with_depot:
            d[(i, j)] = distances.at[i, j] if i != j else 0
    
    # 轉換為時間（分鐘）
    t = {k: v / S * 60 for k, v in d.items()}  # 30 km/h → 轉成分鐘
    C = {row.id: row.C for _, row in stations.iterrows()}
    B = {row.id: row.B for _, row in stations.iterrows()}

    bigM = max(C.values())                   # 站點容量上界 (for balanced range)

    m = gp.Model("YouBike_Rebalancing")

    # === Decision variables === #
    # 確保所有站點 ID 都是字串類型
    N_with_depot = [str(i) for i in N_with_depot]
    N = [str(i) for i in N]
    
    # 路徑變數
    x = m.addVars(N_with_depot, N_with_depot, range(K), vtype=GRB.BINARY, name="x")  # 路徑
    a = m.addVars(N,  range(K), vtype=GRB.INTEGER, name="a")     # 取車
    b = m.addVars(N,  range(K), vtype=GRB.INTEGER, name="b")     # 放車
    y = m.addVars(N,              vtype=GRB.BINARY,  name="y")   # 平衡指標
    W = m.addVars(N_with_depot, range(K), vtype=GRB.INTEGER, name="W")     # 載重
    u = m.addVars(N,  range(K), vtype=GRB.CONTINUOUS, name="u")  # MTZ 排序變數

    # === Objective: maximize balanced stations === #
    m.setObjective(gp.quicksum(y[i] for i in N), GRB.MAXIMIZE)

    # ---------- Station constraints --------- #
    for i in N:
        # (a) Balanced range 30%-70%
        m.addConstr(0.3 * C[i] - bigM * (1 - y[i])
                    <= B[i] + gp.quicksum(b[i,k] - a[i,k] for k in range(K)))
        m.addConstr(B[i] + gp.quicksum(b[i,k] - a[i,k] for k in range(K))
                    <= 0.7 * C[i] + bigM * (1 - y[i]))

        # (b) Station capacity bounds
        m.addConstr(0 <= B[i] + gp.quicksum(b[i,k] - a[i,k] for k in range(K)))
        m.addConstr(B[i] + gp.quicksum(b[i,k] - a[i,k] for k in range(K)) <= C[i])

        # (c) Visitation-operation consistency
        for k in range(K):
            m.addConstr(a[i,k] <= Q * gp.quicksum(x[h,i,k] for h in N_with_depot if h != i))
            m.addConstr(b[i,k] <= Q * gp.quicksum(x[h,i,k] for h in N_with_depot if h != i))

    # ---------- Truck route constraints --------- #
    for k in range(K):
        # Start / end at depot
        m.addConstr(gp.quicksum(x['0',j,k] for j in N) == 1)
        m.addConstr(gp.quicksum(x[i,'0',k] for i in N) == 1)

        for i in N:
            # At most visit once
            m.addConstr(gp.quicksum(x[h,i,k] for h in N_with_depot if h != i) <= 1)
            m.addConstr(gp.quicksum(x[i,j,k] for j in N_with_depot if j != i) <= 1)
            # Flow conservation
            m.addConstr(gp.quicksum(x[h,i,k] for h in N_with_depot if h != i)
                        == gp.quicksum(x[i,j,k] for j in N_with_depot if j != i))

        # Total time window (travel + load/unload)
        travel = gp.quicksum(t[i,j] * x[i,j,k] for i in N_with_depot for j in N_with_depot if i!=j)
        handling = L * gp.quicksum(a[i,k] + b[i,k] for i in N)  # 只計算站點的裝卸時間
        m.addConstr(travel + handling <= T)

    # ---------- MTZ to prevent subtours ---------- #
    # u[i,k] in [1, |N|] when station visited by truck k
    for k in range(K):
        for i in N:
            m.addConstr(u[i,k] <= len(N))          # upper bound
            m.addConstr(u[i,k] >= 1)               # lower bound
        for i in N:
            for j in N:
                if i != j:
                    m.addConstr(u[i,k] - u[j,k] + 1
                                <= (len(N)) * (1 - x[i,j,k]))

    # ---------- Load flow & capacity ---------- #
    for k in range(K):
        for i in N_with_depot:
            m.addConstr(0 <= W[i,k])
            m.addConstr(W[i,k] <= Q)

        for i in N_with_depot:
            for j in N_with_depot:
                if i != j:
                    if j == '0':  # 到達 depot
                        m.addConstr(W[j,k] >= W[i,k] - Q*(1 - x[i,j,k]))
                        m.addConstr(W[j,k] <= W[i,k] + Q*(1 - x[i,j,k]))
                    elif i == '0':  # 從 depot 出發
                        m.addConstr(W[j,k] >= W[i,k] + a[j,k] - b[j,k] - Q*(1 - x[i,j,k]))
                        m.addConstr(W[j,k] <= W[i,k] + a[j,k] - b[j,k] + Q*(1 - x[i,j,k]))
                    else:  # 站點間移動
                        m.addConstr(W[j,k] >= W[i,k] + a[j,k] - b[j,k] - Q*(1 - x[i,j,k]))
                        m.addConstr(W[j,k] <= W[i,k] + a[j,k] - b[j,k] + Q*(1 - x[i,j,k]))


    # ---------- Flow conservation (truck-level in / out) ---------- #
    for k in range(K):
        m.addConstr(gp.quicksum(a[i,k] for i in N) == gp.quicksum(b[i,k] for i in N))

    # ---------- Variable domains (已於 addVars 指定) ---------- #
    m.update()
    return m

# ========= 3. 主程式 ========= #

def get_scenario_params(scenario_dir: str) -> tuple[int, int]:
    """從目錄名稱解析卡車數量和時間窗口"""
    parts = scenario_dir.split('_')
    trucks = int(parts[0].replace('trucks', ''))
    time_window = int(parts[1].replace('min', ''))
    return trucks, time_window

def process_instance(stations_path: str, distances_path: str, K: int, T: int, output_dir: str):
    """處理單一實例"""
    # 載入資料
    stations, dist_df = load_data(stations_path, distances_path)
    
    # 只保留模型需要的欄位
    model_stations = stations[['id', 'C', 'B']]
    
    # 站點 ID (含/不含 depot)
    station_ids = model_stations["id"].tolist()
    station_ids_no_depot = [sid for sid in station_ids if sid != '0']

    # ---- baseline: initial balanced station count (no dispatch) ----
    initial_balanced = sum(
        1
        for sid in station_ids_no_depot
        if 0.3 * model_stations.loc[model_stations.id == sid, "C"].values[0]
           <= model_stations.loc[model_stations.id == sid, "B"].values[0]
           <= 0.7 * model_stations.loc[model_stations.id == sid, "C"].values[0]
    )
    
    # 建模並求解
    model = build_model(model_stations, dist_df, K=K, T=T)
    model.setParam("TimeLimit", 60)   # 5 分鐘上限
    model.optimize()

    # --------- Record solver status & KPI --------- #
    status_code = model.Status
    status_str = "OPTIMAL" if status_code == GRB.OPTIMAL else (
        "TIME_LIMIT" if status_code == GRB.TIME_LIMIT else str(status_code))
    success = model.SolCount > 0           # 有可行解即視為成功
    obj_val = model.ObjVal if success else None

    # ---------- Output only when a feasible solution exists ---------- #
    if success:
        # ========== 3‑A. Station‑level CSV ========= #
        results = []
        for i in station_ids_no_depot:
            balanced = int(model.getVarByName(f"y[{i}]").X + 0.5)
            final_bikes = model_stations.loc[model_stations.id == i, "B"].values[0] \
                        + sum(model.getVarByName(f"b[{i},{k}]").X -
                              model.getVarByName(f"a[{i},{k}]").X
                              for k in range(K))
            results.append({
                "station_id": i,
                "balanced": balanced,
                "final_bikes": int(final_bikes),
                "latitude": stations.loc[stations.id == i, "latitude"].values[0],
                "longitude": stations.loc[stations.id == i, "longitude"].values[0]
            })
        pd.DataFrame(results).to_csv(f"{output_dir}/station_results.csv", index=False)

        # ========== 3‑B. Truck routes with actions ========= #
        routes_lines = []
        for k in range(K):
            route = [0]
            current = 0
            while True:
                next_nodes = [j for j in [0] + station_ids_no_depot
                              if j != current and
                              model.getVarByName(f"x[{current},{j},{k}]").X > 0.5]
                if not next_nodes:
                    break
                current = next_nodes[0]
                route.append(current)
                if current == 0:
                    break

            # Build detailed step string with pick/drop info
            step_strs = []
            for node in route:
                if str(node) == '0':
                    step_strs.append('0')
                    continue
                pick = int(model.getVarByName(f"a[{node},{k}]").X)
                drop = int(model.getVarByName(f"b[{node},{k}]").X)
                action = []
                if pick > 0:
                    action.append(f"-{pick}")
                if drop > 0:
                    action.append(f"+{drop}")
                step_strs.append(f"{node}({''.join(action)})")

            total_pick = sum(int(model.getVarByName(f"a[{i},{k}]").X) for i in station_ids_no_depot)
            total_drop = sum(int(model.getVarByName(f"b[{i},{k}]").X) for i in station_ids_no_depot)
            start_load = int(model.getVarByName(f"W[0,{k}]").X)
            routes_lines.append(
                f"Truck {k} (start_load={start_load}): " +
                " -> ".join(step_strs) +
                f" | total_pick={total_pick}, total_drop={total_drop}"
            )

        # Write enriched routes.txt
        with open(f"{output_dir}/routes.txt", "w", encoding="utf-8") as f:
            f.write(f"Status: {status_str}\n")
            f.write(f"Balanced before (baseline): {initial_balanced}\n")
            if obj_val is not None:
                f.write(f"Objective (balanced stations): {int(obj_val)}\n")
            if status_code == GRB.TIME_LIMIT and obj_val is not None:
                f.write(f"MIPGap: {model.MIPGap:.4f}\n")
            f.write("\n".join(routes_lines))

        return True, obj_val, status_str
    else:
        return False, None, status_str

import concurrent.futures

def solve_wrapper(args):
    """包裝 process_instance 以便多進程呼叫"""
    (idx, stations_path, distances_path, K, T, output_dir,
            scenario_dir, time_period, instance_num) = args    
    try:
        success, obj_val, status_str = process_instance(stations_path, distances_path, K, T, output_dir)
        return (idx, scenario_dir, time_period, instance_num,
                success, obj_val, status_str, output_dir, None)
    except Exception as e:
        # 若有例外，回傳失敗
        return (idx, scenario_dir, time_period, instance_num, False, None, "EXCEPTION", output_dir, str(e))


def main():
    base_dir = "generated_instances"
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)

    # 準備所有 instance 的任務列表
    task_list = []
    for scenario_dir in sorted(os.listdir(base_dir)):
        if not os.path.isdir(os.path.join(base_dir, scenario_dir)):
            continue
        K, T = get_scenario_params(scenario_dir)
        scenario_path = os.path.join(base_dir, scenario_dir)
        for time_period in sorted(os.listdir(scenario_path)):
            if not os.path.isdir(os.path.join(scenario_path, time_period)):
                continue
            time_path = os.path.join(scenario_path, time_period)
            for instance_file in sorted(os.listdir(time_path)):
                if not instance_file.endswith('.csv'):
                    continue
                instance_num = instance_file.split('_')[1].split('.')[0]
                stations_path = os.path.join(time_path, instance_file)
                distances_path = "./distance_output/distance_matrix.csv"
                output_dir = os.path.join(results_dir, scenario_dir, time_period, f"instance_{instance_num}")
                os.makedirs(output_dir, exist_ok=True)
                # 包裝所有必要資訊
                task_list.append((
                    len(task_list),                 # idx
                    stations_path, distances_path, K, T, output_dir,
                    scenario_dir, time_period, instance_num
                ))

    print(f"\n總共需要處理 {len(task_list)} 個實例，開始平行求解 ...")

    total_tasks = len(task_list)
    optimal_cnt = time_limit_cnt = error_cnt = 0 

    # 利用 ProcessPoolExecutor 平行處理所有任務
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=11) as executor:
        # 提交所有任務
        for args in task_list:
            futures.append(executor.submit(solve_wrapper, args))

        # 收集與顯示結果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            # unpack
            idx, scenario_dir, time_period, instance_num, success, obj_val, status_str, output_dir, err = result

            # 印出進度與結果
            print(f"\n[{idx+1}/{total_tasks}] {scenario_dir} | {time_period} | instance_{instance_num} -> {status_str} ", end="")

            if status_str == "OPTIMAL":
                optimal_cnt += 1
            elif status_str == "TIME_LIMIT":
                time_limit_cnt += 1
            else:
                error_cnt += 1

            if success:
                print(f"✓ 最佳化完成：總平衡站數 = {obj_val} (結果於 {output_dir})")
            else:
                if err:
                    print(f"✗ 發生例外: {err}")
                else:
                    print("✗ 模型未在時限內找到最適解")

    # 印出總結
    print("\n=== Summary ===")
    print(f"Optimal      : {optimal_cnt}")
    print(f"Time-limit   : {time_limit_cnt}")
    print(f"Errors/Other : {error_cnt}")


if __name__ == "__main__":
    main()
