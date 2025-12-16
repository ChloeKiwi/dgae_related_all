import matplotlib.pyplot as plt
import pandas as pd

# 数据准备（假设N=100, M=200）
data = {
    "Model": ["DeepGMG", "GraphRNN", "GRAN", "EdgePredict", "GraphGen", "NodeNAR"],
    "Sequence Length": [100+200, 100**2, (100+200)/5, 2*200, 5*100, 100],  # 根据#Sequence Length列计算
    "Generation Steps": [100+200, 100+200, (100+200)/5, 2*200, 100, 100/2],  # 根据#Gen. Steps列计算
    "Complexity": [100**2/200, 100*200, 100*(200+100), 200, 200, 100]  # 根据Complexity列假设N=100, M=200
}

df = pd.DataFrame(data)

# 绘制散点图
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    x=df["Sequence Length"],
    y=df["Generation Steps"],
    c=df["Complexity"],  # 颜色表示复杂度
    s=df["Complexity"]*0.1,  # 大小表示复杂度（缩放比例）
    cmap="viridis",
    alpha=0.8
)

# 标注模型名称
for i, row in df.iterrows():
    plt.annotate(
        row["Model"],
        (row["Sequence Length"] + 50, row["Generation Steps"] + 50),
        fontsize=10
    )

# 突出NodeNAR
plt.scatter(
    df[df["Model"] == "NodeNAR"]["Sequence Length"],
    df[df["Model"] == "NodeNAR"]["Generation Steps"],
    color="red",
    s=200,
    edgecolor="black",
    label="NodeNAR (Ours)"
)

# 图表美化
plt.title("Model Efficiency Comparison: Representation vs Generation", fontsize=14)
plt.xlabel("Sequence Length (Lower = Better)", fontsize=12)
plt.ylabel("Generation Steps (Lower = Better)", fontsize=12)
plt.colorbar(scatter, label="Complexity (Lower = Better)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()