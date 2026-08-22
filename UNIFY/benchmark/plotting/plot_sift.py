import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取数据
result_path = '../result'
benchmark_name = 'sift'
df = pd.read_csv(f"{result_path}/{benchmark_name}.csv")

# 2. 按 ef 分组绘制 Recall-QPS 曲线
plt.figure(figsize=(6, 4))

for i, (ef, group) in enumerate(df.groupby("ef")):
    if i >= 5:
        break

    group_sorted = group.sort_values("recall")
    plt.plot(
        group_sorted["recall"],
        group_sorted["QPS"],
        marker="o",
        label=f"ef={ef}"
    )

# 3. 图形设置
plt.xlabel("Recall")
plt.ylabel("QPS")
plt.legend()
plt.grid(True)

# 4. 显示 / 保存
plt.tight_layout()
plt.savefig(f"{benchmark_name}_recall_qps_by_ef.png")
plt.show()