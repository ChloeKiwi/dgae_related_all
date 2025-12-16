import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 数据准备
data = {
    "Model": ["DeepGMG", "GraphRNN", "EdgePredict", "GraphGen", "NodeNAR"],
    "Representation Size": [266, 15625, 282, 625, 125],  # 表示大小
    "Generation Steps": [266, 15625, 282, 625, 125],     # 生成步骤
    "Complexity": ["O(NM^2)", "O(N^2)", "O(M)", "O(N+M)", "O(N)"]  # 复杂度
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 设置 Seaborn 风格
sns.set(style="whitegrid")

# 创建散点图
plt.figure(figsize=(10, ))
scatter = sns.scatterplot(
    x="Representation Size", 
    y="Generation Steps", 
    hue="Model", 
    size="Complexity",  # 用点的大小表示复杂度
    sizes=(100, 400),  # 点的大小范围
    palette="viridis",  # 颜色方案
    data=df
)

# 突出显示 NodeNAR
node_nar = df[df["Model"] == "NodeNAR"]
plt.scatter(
    node_nar["Representation Size"], 
    node_nar["Generation Steps"], 
    color="red", 
    s=200,  # 点的大小
    label="NodeNAR (Ours)", 
    edgecolor="black", 
    linewidth=2
)

# 添加标签
for i, row in df.iterrows():
    plt.text(
        row["Representation Size"] + 50,  # X 轴偏移
        row["Generation Steps"] + 50,     # Y 轴偏移
        row["Model"], 
        fontsize=12, 
        ha="left"
    )

# 设置标题和标签
plt.title("Comparison of Representation Size and Generation Steps", fontsize=16)
plt.xlabel("Representation Size", fontsize=14)
plt.ylabel("Generation Steps", fontsize=14)

# 显示图例
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

# 显示图表
plt.tight_layout()

plt.savefig('rep.pdf')

plt.show()