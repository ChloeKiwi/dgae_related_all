import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 数据准备
data = {
    "Model": ["Janus-Pro-7B", "TokenFlow-XL", "Janus-Pro-1B", "LaVA-v1.5-7B*", "Show-o-512", "Janus", "Snow-o-256"],
    "LLM Parameters (Billions)": [7, 64, 1, 7, 512, 8, 256],  # 参数量
    "Average Performance": [58, 64, 58, 58, 512, 10, 256]   # 平均性能
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 设置 Seaborn 风格
sns.set(style="whitegrid")

# 创建散点图
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(
    x="LLM Parameters (Billions)", 
    y="Average Performance", 
    hue="Model", 
    size="LLM Parameters (Billions)",  # 用点的大小表示参数量
    sizes=(100, 400),  # 点的大小范围
    palette="viridis",  # 颜色方案
    data=df
)

# 添加标签
for i, row in df.iterrows():
    plt.text(
        row["LLM Parameters (Billions)"] + 5,  # X 轴偏移
        row["Average Performance"] + 5,       # Y 轴偏移
        row["Model"], 
        fontsize=12, 
        ha="left"
    )

# 设置标题和标签
plt.title("Comparison of LLM Parameters and Average Performance", fontsize=16)
plt.xlabel("LLM Parameters (Billions)", fontsize=14)
plt.ylabel("Average Performance", fontsize=14)

# 显示图例
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

# 显示图表
plt.tight_layout()
plt.savefig("plot2.png")

plt.show()