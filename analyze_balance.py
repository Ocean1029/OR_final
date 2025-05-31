import pandas as pd
import os
import matplotlib.pyplot as plt

def analyze_balance_ratios():
    # 設定基本路徑
    base_path = "optimization_results/190114_新小南港_limit300s_時速60"
    
    # 儲存每個scenario、每個time slot的balance比例
    balance_ratios = {}  # {scenario: {time_slot: [ratios]}}

    # 遍歷6個scenario資料夾
    for scenario in os.listdir(base_path):
        scenario_path = os.path.join(base_path, scenario)
        if not os.path.isdir(scenario_path):
            continue

        balance_ratios[scenario] = {}
        # 遍歷 time slot 資料夾
        for time_slot in os.listdir(scenario_path):
            time_slot_path = os.path.join(scenario_path, time_slot)
            if not os.path.isdir(time_slot_path):
                continue
            scenario_ratios = []
            # 遍歷 instance 資料夾
            for instance in os.listdir(time_slot_path):
                instance_path = os.path.join(time_slot_path, instance)
                if not os.path.isdir(instance_path):
                    continue
                csv_path = os.path.join(instance_path, "station_results.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if 'balanced' in df.columns:
                            balance_ratio = (df['balanced'] == True).mean()
                            scenario_ratios.append(balance_ratio)
                    except Exception as e:
                        print(f"處理文件 {csv_path} 時出錯: {str(e)}")
            if scenario_ratios:
                balance_ratios[scenario][time_slot] = scenario_ratios

    # 輸出結果
    print("\n各Scenario各時間的Balance比例分析：")
    print("-" * 50)
    best = {}
    for time_slot in set(ts for s in balance_ratios.values() for ts in s):
        # 收集所有scenario在這個time_slot的平均
        slot_means = {scenario: (sum(data[time_slot])/len(data[time_slot]) if time_slot in data else 0)
                      for scenario, data in balance_ratios.items()}
        best_scenario = max(slot_means.items(), key=lambda x: x[1])
        worst_scenario = min(slot_means.items(), key=lambda x: x[1])
        best[time_slot] = best_scenario
        print(f"\n【{time_slot}】")
        for scenario, ratios in slot_means.items():
            print(f"  {scenario}: 平均Balance比例 {ratios:.2%}")
        print(f"  最佳: {best_scenario[0]} ({best_scenario[1]:.2%})")
        print(f"  最差: {worst_scenario[0]} ({worst_scenario[1]:.2%})")

    # 繪製每個 time slot 的箱型圖
    for time_slot in set(ts for s in balance_ratios.values() for ts in s):
        plt.figure(figsize=(10, 6))
        data_to_plot = [data[time_slot] for data in balance_ratios.values() if time_slot in data]
        labels = [scenario for scenario, data in balance_ratios.items() if time_slot in data]
        plt.boxplot(data_to_plot, labels=labels)
        plt.title(f'{time_slot} 各Scenario的Balance比例分布')
        plt.ylabel('Balance比例')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'balance_analysis_{time_slot}.png')
        print(f"分析圖表已保存為 'balance_analysis_{time_slot}.png'")

if __name__ == "__main__":
    analyze_balance_ratios()