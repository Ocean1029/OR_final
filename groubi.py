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
    MIN_LNG = 121.58498      # 西邊界
    MAX_LNG = 123            # 東邊界（臨時設個 123°E，比台北再東一些）
    MIN_LAT = 25.04615       # 南邊界
    MAX_LAT = 25.08550       # 北邊界
    
    # 篩選站點
    stations = stations[(stations.longitude > MIN_LNG) & (stations.longitude < MAX_LNG) &
                       (stations.latitude > MIN_LAT) & (stations.latitude < MAX_LAT)]

    
    # 讀取距離矩陣
    dist_df = pd.read_csv(distance_file, index_col=0)
    
    # 確保距離矩陣的索引和欄位名稱與站點 ID 一致
    dist_df.index = dist_df.index.astype(str)
    dist_df.columns = dist_df.columns.astype(str)
    stations['id'] = stations['id'].astype(str)
    
    # 確保距離矩陣包含所有需要的站點
    all_stations = set(stations['id'])
    dist_stations = set(dist_df.index)
    missing_stations = all_stations - dist_stations
    
    if missing_stations:
        # 使用平均距離填充缺失的站點
        avg_dist = dist_df.values.mean()
        for station in missing_stations:
            dist_df.loc[station] = avg_dist
            dist_df[station] = avg_dist
        
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
                Q: int = 14,
                L: float = 2   # 實際裝卸時間/車 (分鐘)
               ) -> gp.Model:

    N = stations["id"].tolist()  # 站點 ID 列表 
    N0 = [0] + N                 # 加入 depot (0)
    print(f"\n建立模型：")
    print(f"- 站點數：{len(N)}")
    print(f"- 卡車數：{K}")
    print(f"- 時間限制：{T} 分鐘")
    
    # 建立包含 depot 的完整距離矩陣
    dist = distances.to_dict()   # {(i,j): 距離(公里)}
    
    # 計算 depot 到各站點的距離（使用平均距離）
    avg_dist = distances.values.mean()
    
    d = {}
    missing_pairs = []
    for i in N0:
        for j in N0:
            if i == j:
                d[(i, j)] = 0
            elif i == 0 or j == 0:
                d[(i, j)] = 0.0005
            else:
                try:
                    d[(i, j)] = dist[i][j]
                except KeyError:
                    missing_pairs.append((i, j))
                    d[(i, j)] = avg_dist
    
    if missing_pairs:
        print(f"\n警告：找不到以下站點對的距離，使用平均距離代替：")
        for i, j in missing_pairs[:5]:  # 只顯示前5個
            print(f"- {i} → {j}")
        if len(missing_pairs) > 5:
            print(f"... 還有 {len(missing_pairs)-5} 個站點對")
    
    # 轉換為時間（分鐘）
    t = {k: v / 30 * 60 for k, v in d.items()}  # 30 km/h → 轉成分鐘

    C = {row.id: row.C for _, row in stations.iterrows()}
    B = {row.id: row.B for _, row in stations.iterrows()}

    bigM = max(C.values())                   # 站點容量上界 (for balanced range)

    m = gp.Model("YouBike_Rebalancing")

    # === Decision variables === #
    x = m.addVars(N0, N0, range(K), vtype=GRB.BINARY, name="x")  # 路徑
    a = m.addVars(N,  range(K), vtype=GRB.INTEGER, name="a")     # 取車
    b = m.addVars(N,  range(K), vtype=GRB.INTEGER, name="b")     # 放車
    y = m.addVars(N,              vtype=GRB.BINARY,  name="y")   # 平衡指標
    W = m.addVars(N0, range(K), vtype=GRB.INTEGER, name="W")     # 載重
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
            m.addConstr(a[i,k] <= Q * gp.quicksum(x[h,i,k] for h in N0 if h != i))
            m.addConstr(b[i,k] <= Q * gp.quicksum(x[h,i,k] for h in N0 if h != i))

    # ---------- Truck route constraints --------- #
    for k in range(K):
        # Start / end at depot
        m.addConstr(gp.quicksum(x[0,j,k] for j in N) == 1)
        m.addConstr(gp.quicksum(x[i,0,k] for i in N) == 1)

        for i in N:
            # At most visit once
            m.addConstr(gp.quicksum(x[h,i,k] for h in N0 if h != i) <= 1)
            m.addConstr(gp.quicksum(x[i,j,k] for j in N0 if j != i) <= 1)
            # Flow conservation
            m.addConstr(gp.quicksum(x[h,i,k] for h in N0 if h != i)
                        == gp.quicksum(x[i,j,k] for j in N0 if j != i))

        # Total time window (travel + load/unload)
        travel = gp.quicksum(t[i,j] * x[i,j,k] for i in N0 for j in N0 if i!=j)
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
        # Depot loading decision: W[0,k] free in [0,Q]
        m.addConstr(0 <= W[0,k])
        m.addConstr(W[0,k] <= Q)

        for i in N0:
            for j in N0:
                if i != j:
                    if j == 0:  # 到達 depot
                        m.addConstr(W[j,k] >= W[i,k] - Q*(1 - x[i,j,k]))
                        m.addConstr(W[j,k] <= W[i,k] + Q*(1 - x[i,j,k]))
                    elif i == 0:  # 從 depot 出發
                        m.addConstr(W[j,k] >= W[i,k] + a[j,k] - b[j,k] - Q*(1 - x[i,j,k]))
                        m.addConstr(W[j,k] <= W[i,k] + a[j,k] - b[j,k] + Q*(1 - x[i,j,k]))
                    else:  # 站點間移動
                        m.addConstr(W[j,k] >= W[i,k] + a[j,k] - b[j,k] - Q*(1 - x[i,j,k]))
                        m.addConstr(W[j,k] <= W[i,k] + a[j,k] - b[j,k] + Q*(1 - x[i,j,k]))

        for i in N:
            m.addConstr(0 <= W[i,k])
            m.addConstr(W[i,k] <= Q)

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
    
    # 建模並求解
    model = build_model(model_stations, dist_df, K=K, T=T)
    model.setParam("TimeLimit", 300)   # 5 分鐘上限
    model.optimize()
    
    # 準備輸出結果
    results = []
    if model.Status == GRB.OPTIMAL:
        # 收集各站點狀態
        for i in model_stations["id"]:
            balanced = int(model.getVarByName(f"y[{i}]").X + 0.5)
            final_bikes = model_stations.loc[model_stations.id==i,"B"].values[0] \
                        + sum(model.getVarByName(f"b[{i},{k}]").X -
                              model.getVarByName(f"a[{i},{k}]").X
                              for k in range(K))
            results.append({
                "station_id": i,
                "balanced": balanced,
                "final_bikes": int(final_bikes),
                "latitude": stations.loc[stations.id==i, "latitude"].values[0],
                "longitude": stations.loc[stations.id==i, "longitude"].values[0]
            })
        
        # 收集卡車路徑
        routes = []
        for k in range(K):
            route = [0]
            current = 0
            while True:
                next_nodes = [j for j in [0]+model_stations["id"].tolist()
                              if j != current and
                              model.getVarByName(f"x[{current},{j},{k}]").X > 0.5]
                if not next_nodes:
                    break
                current = next_nodes[0]
                route.append(current)
                if current == 0:
                    break
            routes.append(route)
        
        # 儲存結果
        result_df = pd.DataFrame(results)
        result_df.to_csv(f"{output_dir}/station_results.csv", index=False)
        
        # 儲存路徑
        with open(f"{output_dir}/routes.txt", "w") as f:
            for k, route in enumerate(routes):
                f.write(f"Truck {k}: Route {route}\n")
        
        return True, model.ObjVal
    else:
        return False, None

import concurrent.futures

def solve_wrapper(args):
    """包裝 process_instance 以便多進程呼叫"""
    stations_path, distances_path, K, T, output_dir, scenario_dir, time_period, instance_num = args
    try:
        success, obj_val = process_instance(stations_path, distances_path, K, T, output_dir)
        return (scenario_dir, time_period, instance_num, success, obj_val, output_dir)
    except Exception as e:
        # 若有例外，回傳失敗
        return (scenario_dir, time_period, instance_num, False, None, output_dir, str(e))


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
                    stations_path, distances_path, K, T, output_dir,
                    scenario_dir, time_period, instance_num
                ))

    print(f"\n總共需要處理 {len(task_list)} 個實例，開始平行求解 ...")

    # 利用 ProcessPoolExecutor 平行處理所有任務
    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交所有任務
        for args in task_list:
            futures.append(executor.submit(solve_wrapper, args))

        # 收集與顯示結果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            # unpack
            if len(result) == 7:
                scenario_dir, time_period, instance_num, success, obj_val, output_dir, err = result
            else:
                scenario_dir, time_period, instance_num, success, obj_val, output_dir = result
                err = None
            # 印出進度與結果
            print(f"\n[{scenario_dir} | {time_period} | instance_{instance_num}] ", end="")
            if success:
                print(f"✓ 最佳化完成：總平衡站數 = {obj_val} (結果於 {output_dir})")
            else:
                if err:
                    print(f"✗ 發生例外: {err}")
                else:
                    print("✗ 模型未在時限內找到最適解")

if __name__ == "__main__":
    main()
