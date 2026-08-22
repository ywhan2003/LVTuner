import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 基本配置
result_path = '../result'
benchmark_name = 'sift'
M_LIST = [2, 4, 8, 16, 32, 64]

horizontal_metric = 'recall'
vertical_metric = 'QPS'

NUM_POINTS = 10
RECALL_THRESHOLD = 0.8

plt.figure()

for M in M_LIST:
    csv_path = f"{result_path}/{benchmark_name}_M{M}.csv"
    df = pd.read_csv(csv_path)

    # 只保留需要的列
    df = df[[horizontal_metric, vertical_metric]]

    # 转为数值
    df[horizontal_metric] = pd.to_numeric(df[horizontal_metric])
    df[vertical_metric] = pd.to_numeric(df[vertical_metric])

    # 按 recall 排序
    df = df.sort_values(by=horizontal_metric).reset_index(drop=True)

    # === 1️⃣ 只保留 recall >= 0.8 ===
    df = df[df[horizontal_metric] >= RECALL_THRESHOLD].reset_index(drop=True)

    if df.empty:
        print(f"[WARN] M={M}: no points with recall >= {RECALL_THRESHOLD}")
        continue

    # === 2️⃣ 等间隔抽样 10 个点 ===
    if len(df) > NUM_POINTS:
        idx = np.linspace(0, len(df) - 1, NUM_POINTS, dtype=int)
        df_sampled = df.iloc[idx]
    else:
        df_sampled = df

    # === 3️⃣ 画折线 ===
    plt.plot(
        df_sampled[horizontal_metric],
        df_sampled[vertical_metric],
        marker="o",
        label=f"M={M}"
    )

# 图形设置
plt.xlabel(horizontal_metric)
plt.ylabel(vertical_metric)
plt.grid(True)
plt.legend()

# 保存并显示
plt.savefig(f"{benchmark_name}.png", dpi=150)
plt.show()
