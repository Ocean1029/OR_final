from heuristic_solver import solve_instance

stations_path = "stations.csv"      # 你的站點資料檔案
distances_path = "dist_df.csv"      # 你的距離矩陣檔案
K = 2                               # 車輛數量
T = 120                             # 每台車的時間限制（單位同距離矩陣）

routes = solve_instance(stations_path, distances_path, K, T)
for i, route in enumerate(routes):
    print(f"Truck {i+1} route: {route}")