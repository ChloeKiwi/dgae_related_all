import numpy as np
import matplotlib.pyplot as plt

# 数据
metrics = ['Recon MMD', 'Prior Loss', 'Prior MMD', 'Generation MMD']
values = np.array([
    [0.024, 0.084, 0.013, 0.031],  # AR w Sort
    [0.024, 0.083, 0.013, 0.031],  # AR w/o Sort
    [0.023, 0.163, 0.037, 0.029],  # NAR w Sort
    [0.023, 0.153, 0.022, 0.031]   # NAR w/o Sort
])

# 设置柱状图位置
x = np.arange(len(metrics))
width = 0.2  # 柱的宽度

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
rects1 = ax.bar(x - width*1.5, values[0], width, label='AR w Sort', color='#1f77b4')
rects2 = ax.bar(x - width/2, values[1], width, label='AR w/o Sort', color='#aec7e8')
rects3 = ax.bar(x + width/2, values[2], width, label='NAR w Sort', color='#ff7f0e')
rects4 = ax.bar(x + width*1.5, values[3], width, label='NAR w/o Sort', color='#ffbb78')

# 自定义图表
ax.set_ylabel('Value')
ax.set_title('Comparison of Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45)
ax.legend()

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, size=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# 调整布局
plt.tight_layout()

plt.savefig('sortcompare.pdf')

# 显示图表
plt.show()