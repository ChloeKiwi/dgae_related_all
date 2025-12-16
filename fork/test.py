# import torch

# def get_mask_from_indices(indices, mask_indice):
#     """
#     为每个序列创建mask，在第一个mask_indice之前的位置为1，之后为0
#     Args:
#         indices: (bs, n_max, nc) 或 (bs, n_max)
#         mask_indice: 标记位置的值（比如10）
#     Returns:
#         mask: (bs, n_max) 布尔tensor
#     """
#     device = indices.device 
#     if indices.dim() == 3:
#         indices = indices.squeeze(-1)  # (bs, n_max)
#     bs, n_max = indices.shape
    
#     # 找到每个序列中第一个mask_indice的位置
#     first_mask = (indices == mask_indice).float()  # 将bool转为float
#     first_mask_pos = first_mask.argmax(dim=1)  # (bs,)
    
#     # 如果某行没有mask_indice，argmax会返回0
#     # 需要处理这种情况
#     has_mask = first_mask.sum(dim=1) > 0  # (bs,)
    
#     # 创建位置索引
#     pos_idx = torch.arange(n_max, device=device).unsqueeze(0)  # (1, n_max)
    
#     # 创建mask：位置小于first_mask_pos的为1，大于的为0
#     mask = pos_idx < first_mask_pos.unsqueeze(1)  # (bs, n_max)
    
#     # 对于没有mask_indice的行，设置全1
#     mask = torch.where(has_mask.unsqueeze(1), mask, torch.ones_like(mask))
    
#     return mask

# indices = torch.randint(0, 11, (10, 10, 1))
# print("indices:", indices.squeeze(-1))
# mask_indice = 10
# mask = get_mask_from_indices(indices, mask_indice)
# print("mask:", mask)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 准备数据
metrics = ['Recon MMD', 'Prior Loss', 'Prior MMD', 'Generation MMD']
ar_w_sort = [0.024, 0.084, 0.013, 0.031]
ar_wo_sort = [0.024, 0.083, 0.013, 0.031]
nar_w_sort = [0.023, 0.163, 0.037, 0.029]
nar_wo_sort = [0.023, 0.153, 0.022, 0.031]

# 创建DataFrame
data = pd.DataFrame({
    'Metric': metrics * 4,
    'Value': ar_w_sort + ar_wo_sort + nar_w_sort + nar_wo_sort,
    'Model': ['AR w Sort']*4 + ['AR w/o Sort']*4 + ['NAR w Sort']*4 + ['NAR w/o Sort']*4
})

# 设置样式
plt.style.use('seaborn')
plt.figure(figsize=(10, 6))

# 绘制柱状图
sns.barplot(
    x='Metric', 
    y='Value', 
    hue='Model',
    data=data,
    palette=['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']  # 设置颜色
)

# 自定义图表
plt.title('Comparison of Metrics Across Different Models and Sorting Strategies', pad=20)
plt.xlabel('Metrics', labelpad=10)
plt.ylabel('Value', labelpad=10)

# 旋转x轴标签
plt.xticks(rotation=45)

# 调整图例位置
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 添加数值标签
ax = plt.gca()
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, rotation=90, size=8)

# 调整布局
plt.tight_layout()

plt.savefig('sort compare.pdf')
# 显示图表
plt.show()