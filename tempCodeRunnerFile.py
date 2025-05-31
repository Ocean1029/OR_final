    # 確保距離矩陣包含所有需要的站點
    stations_set = set(stations['id'])
    dist_stations = set(dist_df.index)
    