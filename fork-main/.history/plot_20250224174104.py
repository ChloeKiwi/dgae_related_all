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

# 设置整体风格和颜色
plt.style.use('seaborn-v0_8-paper')  # 使用更专业的风格
sns.set_palette("husl")  # 使用更鲜明的颜色方案

# 创建图形，设置更大的尺寸和更高的DPI
plt.figure(figsize=(12, 8), dpi=300)

# 创建散点图，使用更优雅的参数
scatter = sns.scatterplot(
    x="Representation Size", 
    y="Generation Steps", 
    hue="Model",
    size="Complexity",
    sizes=(150, 500),  # 增加点的大小范围
    alpha=0.7,  # 添加透明度
    data=df
)

# 突出显示 NodeNAR，使用更醒目的样式
node_nar = df[df["Model"] == "NodeNAR"]
plt.scatter(
    node_nar["Representation Size"], 
    node_nar["Generation Steps"], 
    color="#FF3366",  # 使用更醒目的红色
    s=300,  # 增大点的大小
    label="NodeNAR (Ours)", 
    edgecolor="white",  # 白色边框更显眼
    linewidth=2,
    zorder=5  # 确保显示在最上层
)

# 优化标签位置和样式
for i, row in df.iterrows():
    x_offset = row["Representation Size"] * 0.05  # 动态计算偏移量
    y_offset = row["Generation Steps"] * 0.05
    plt.text(
        row["Representation Size"] + x_offset,
        row["Generation Steps"] + y_offset,
        row["Model"],
        fontsize=11,
        ha="left",
        va="bottom",
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)  # 添加文本背景
    )

# 设置更专业的标题和标签
plt.title("Model Comparison: Representation Size vs. Generation Steps", 
          fontsize=16, pad=20, fontweight='bold')
plt.xlabel("Representation Size (log scale)", fontsize=12)
plt.ylabel("Generation Steps (log scale)", fontsize=12)

# 设置对数刻度
plt.xscale('log')
plt.yscale('log')

# 优化图例
plt.legend(title="Models", 
          title_fontsize=12,
          fontsize=10,
          bbox_to_anchor=(1.15, 1),
          loc='upper left',
          frameon=True,
          edgecolor='none')

# 调整布局和边距
plt.tight_layout(rect=[0, 0, 0.9, 1])  # 为图例留出空间

# 保存图片，设置更高质量
plt.savefig('rep.pdf', dpi=300, bbox_inches='tight')

plt.show()