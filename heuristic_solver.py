import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
import math

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

    # 加入 depot (0) df
    depot = pd.DataFrame({
        'id': ['0'],
        'C': [100],  # 假設 depot 容量很大
        'B': [50],
        'latitude': [25.05326],
        'longitude': [121.60656]
    })
    stations = pd.concat([depot, stations], ignore_index=True)
    
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
    
    if len(stations_set.intersection(dist_stations)) == 0:
        raise ValueError("錯誤：距離矩陣中沒有任何與站點資料匹配的站點 ID。請檢查距離矩陣檔案是否正確。")
    
    missing_stations = stations_set - dist_stations
    
    if missing_stations:
        avg_dist = dist_df.values.mean()
        # 先補 column
        for station in missing_stations:
            dist_df[station] = avg_dist
        # 再補 row
        for station in missing_stations:
            dist_df.loc[station] = avg_dist
        # 重新排序 row/column
        dist_df = dist_df.loc[station_ids, station_ids]

    return stations, dist_df

class HeuristicSolver:
    def __init__(self, stations: pd.DataFrame, distances: pd.DataFrame, K: int, T: int, Q: int = 14, L: float = 2):
        """
        Initialize the heuristic solver
        
        Args:
            stations: Station data
            distances: Distance matrix
            K: Number of trucks
            T: Time limit
            Q: Truck capacity
            L: Loading/unloading time
        """
        self.stations = stations
        self.distances = distances
        self.K = K
        self.T = T
        self.Q = Q
        self.L = L
        self.depot = '0'
        self.n = len(stations)
        
    def _find_nearest_station(self, current_pos: str, station_list: List[str]) -> Tuple[str, float]:
        """Find the nearest station"""
        min_dist = float('inf')
        nearest_station = None
        
        for station in station_list:
            dist = self.distances.at[current_pos, station]
            if dist < min_dist:
                min_dist = dist
                nearest_station = station
                
        return nearest_station, min_dist
    
    def solve(self) -> List[List[str]]:
        """
        Execute the heuristic algorithm
        
        Returns:
            List[List[str]]: The path of each truck
        """
        # Initialize the station status
        current_bikes = {row['id']: row['B'] for _, row in self.stations.iterrows()}
        capacities = {row['id']: row['C'] for _, row in self.stations.iterrows()}
        
        # Calculate the target number of vehicles for each station
        target_bikes = {i: 0.5 * capacities[i] for i in current_bikes.keys()}
        
        # Initialize the surplus and deficit lists
        surplus_list = []
        deficit_list = []
        surplus_amount = {}
        deficit_amount = {}
        
        for i in current_bikes.keys():
            if i == self.depot:  # Skip depot
                continue
            if current_bikes[i] < 0.3 * capacities[i]:
                deficit_amount[i] = target_bikes[i] - current_bikes[i]
                deficit_list.append(i)
            elif current_bikes[i] > 0.7 * capacities[i]:
                surplus_amount[i] = current_bikes[i] - target_bikes[i]
                surplus_list.append(i)
        
        # Plan the path for each truck
        routes = []
        for _ in range(self.K):
            position = self.depot
            remaining_time = self.T
            load = 0
            route = [self.depot]
            
            while remaining_time > 0 and (surplus_list or deficit_list):
                if load < self.Q and surplus_list:
                    # Find the nearest surplus station to collect vehicles
                    next_station, move_time = self._find_nearest_station(position, surplus_list)
                    if next_station is None:
                        break
                        
                    op_time = self.L * min(surplus_amount[next_station], self.Q - load)
                    
                    if move_time + op_time > remaining_time:
                        break
                        
                    route.append(next_station)
                    bikes_to_collect = min(surplus_amount[next_station], self.Q - load)
                    load += bikes_to_collect
                    surplus_amount[next_station] -= bikes_to_collect
                    current_bikes[next_station] -= bikes_to_collect
                    
                    if surplus_amount[next_station] == 0:
                        surplus_list.remove(next_station)
                        
                    remaining_time -= (move_time + op_time)
                    position = next_station
                    
                elif load > 0 and deficit_list:
                    # Find the nearest deficit station to replenish vehicles
                    next_station, move_time = self._find_nearest_station(position, deficit_list)
                    if next_station is None:
                        break
                        
                    op_time = self.L * min(deficit_amount[next_station], load)
                    
                    if move_time + op_time > remaining_time:
                        break
                        
                    route.append(next_station)
                    bikes_to_deliver = min(deficit_amount[next_station], load)
                    load -= bikes_to_deliver
                    deficit_amount[next_station] -= bikes_to_deliver
                    current_bikes[next_station] += bikes_to_deliver
                    
                    if deficit_amount[next_station] == 0:
                        deficit_list.remove(next_station)
                        
                    remaining_time -= (move_time + op_time)
                    position = next_station
                else:
                    break
            
            route.append(self.depot)
            routes.append(route)
            
        return routes

def solve_instance(stations_path: str, distances_path: str, K: int, T: int) -> List[List[str]]:
    """
    Solve a single instance
    
    Args:
        stations_path: The path to the station data file
        distances_path: The path to the distance matrix file
        K: The number of trucks
        T: The time limit
        
    Returns:
        List[List[str]]: The path of each truck
    """
    # Load data
    stations, distances = load_data(stations_path, distances_path)
    
    # Create solver and solve
    solver = HeuristicSolver(stations, distances, K=K, T=T)
    return solver.solve() 