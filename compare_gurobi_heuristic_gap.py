import os
import pandas as pd

def sum_balanced_in_dir(base_dir, scenario, time_slot):
    """加總某 scenario/time_slot 下所有 instance 的 balanced 數量與總站點數"""
    total_balanced = 0
    total_count = 0
    slot_path = os.path.join(base_dir, scenario, time_slot)
    if not os.path.isdir(slot_path):
        return 0, 0
    for instance in os.listdir(slot_path):
        instance_path = os.path.join(slot_path, instance)
        if not os.path.isdir(instance_path):
            continue
        csv_path = os.path.join(instance_path, "station_results.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'balanced' in df.columns:
                    total_balanced += (df['balanced'] == True).sum()
                    total_count += len(df)
            except Exception as e:
                print(f"處理 {csv_path} 時出錯: {e}")
    return total_balanced, total_count

def compare_balanced_gap(
    gurobi_dir="optimization_results/190114_新小南港_limit300s_時速60",
    heuristic_dir="optimization_results_for_heuristic/20250531_231106"
):
    results = []
    scenario_results = {}
    
    for scenario in os.listdir(gurobi_dir):
        scenario_path = os.path.join(gurobi_dir, scenario)
        if not os.path.isdir(scenario_path):
            continue
            
        # 初始化此scenario的加總值
        scenario_results[scenario] = {
            "g_balanced": 0,
            "g_total": 0,
            "h_balanced": 0,
            "h_total": 0,
            "time_slot_count": 0
        }
        
        for time_slot in os.listdir(scenario_path):
            time_slot_path = os.path.join(scenario_path, time_slot)
            if not os.path.isdir(time_slot_path):
                continue
                
            # Gurobi
            g_balanced, g_total = sum_balanced_in_dir(gurobi_dir, scenario, time_slot)
            # Heuristic
            h_balanced, h_total = sum_balanced_in_dir(heuristic_dir, scenario, time_slot)
            
            # 加總到scenario結果中
            scenario_results[scenario]["g_balanced"] += g_balanced
            scenario_results[scenario]["g_total"] += g_total
            scenario_results[scenario]["h_balanced"] += h_balanced
            scenario_results[scenario]["h_total"] += h_total
            scenario_results[scenario]["time_slot_count"] += 1
    
    # 計算每個scenario的平均結果
    for scenario, data in scenario_results.items():
        if data["time_slot_count"] > 0:
            # 計算平均比例
            g_ratio = data["g_balanced"] / data["g_total"] if data["g_total"] > 0 else 0
            h_ratio = data["h_balanced"] / data["h_total"] if data["h_total"] > 0 else 0
            # 計算gap
            gap = (h_ratio - g_ratio) / g_ratio * 100 if g_ratio > 0 else None
            
            results.append({
                "scenario": scenario,
                "gurobi_balanced_avg": data["g_balanced"] / data["time_slot_count"],
                "gurobi_total_avg": data["g_total"] / data["time_slot_count"],
                "gurobi_ratio": g_ratio,
                "heuristic_balanced_avg": data["h_balanced"] / data["time_slot_count"],
                "heuristic_total_avg": data["h_total"] / data["time_slot_count"],
                "heuristic_ratio": h_ratio,
                "gap(%)": gap
            })
    
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("gurobi_heuristic_balanced_gap.csv", index=False)
    print("\n已輸出詳細結果到 gurobi_heuristic_balanced_gap.csv")

if __name__ == "__main__":
    compare_balanced_gap() 