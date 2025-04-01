import numpy as np
import matplotlib.pyplot as plt

# 数据
categories = ["Category 1", "Category 2", "Category 3"]  # 大类
subcategories = [
    ["A1", "A2"],   # Category 1 下的小类
    ["B1", "B2"],   # Category 2 下的小类
    ["C1", "C2"]    # Category 3 下的小类
]
values = [
    [10, 15],   # Category 1 下的小类数据
    [20, 25],   # Category 2 下的小类数据
    [30, 35]    # Category 3 下的小类数据
]

# 计算 x 轴刻度位置
x_positions = []  # 存储每个柱子的 x 坐标
labels = []  # 存储每个柱子的小类标签
category_positions = []  # 存储大类的中心位置
current_x = 0

for i, subs in enumerate(subcategories):
    category_positions.append(current_x + (len(subs) - 1) / 2)  # 计算大类的中心位置
    for sub in subs:
        x_positions.append(current_x)
        labels.append(sub)  # 记录小类标签
        current_x += 1

# 画柱形图
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x_positions, np.concatenate(values), width=0.6)

# 设置 x 轴
ax.set_xticks(x_positions)
ax.set_xticklabels(labels, rotation=0)  # 小类标签

# 添加大类标签
for i, category in enumerate(categories):
    ax.text(category_positions[i], -2, category, ha='center', fontsize=12, fontweight='bold')

# 其他图表设置
ax.set_ylabel("Score")
ax.set_title("Average Total Reward")

plt.show()

