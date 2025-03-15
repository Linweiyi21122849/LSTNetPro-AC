import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 配置 Matplotlib 支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为 SimHei（黑体）
rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

# 1. 读取保存的 NumPy 文件
file_path = r'data\arr.npy'  # 替换为你的 .npy 文件路径
data = np.load(file_path, allow_pickle=True)

data = data[:300]

# 2. 筛选出不全为零的维度，并且每个维度的时间段长度为 500
filtered_data = []
count = 0
for t in range(data.shape[1]):
    dimension = data[:, t]
    # 如果该维度的前 500 个元素中没有全零的情况，则加入 filtered_data
    if np.sum(dimension) == 0.0:
        print("去除时间段: ",t)  
    else:
        count += 1
        filtered_data.append(dimension)  # 保存该维度的前 500 个元素
        if count == 500:
            break

# 将 filtered_data 转换为 numpy 数组
data = np.array(filtered_data, dtype=object)
data = data.transpose()

# 检查数据格式
print("数据形状:", data.shape)
print("数据内容:", data)

# np.save(r'data\Video300_Time500.npy', data)
np.savetxt(r'动态规划方法\Video300_Time500.txt', data, "%d")


# # 每页显示的子图数量 (5 x 5)
# num_per_page = 25
# # 计算一共有多少页
# num_pages = (data.shape[0] // num_per_page) + (1 if data.shape[0] % num_per_page != 0 else 0)

# # 绘制每页的图表
# for page in range(num_pages):
#     # 创建一个新的图形
#     fig, axes = plt.subplots(5, 5, figsize=(15, 15))
#     axes = axes.flatten()  # 将二维数组转换为一维数组，方便索引

#     start_idx = page * num_per_page
#     end_idx = min((page + 1) * num_per_page, data.shape[0])  # 计算当前页要显示的结束索引

#     for i, ax in enumerate(axes[:end_idx - start_idx]):
#         idx = start_idx + i
#         ax.plot(data[idx])  # 绘制第 idx 个视频的播放量折线图
#         ax.set_title(f"视频 {idx + 1}", fontsize=10)
#         ax.set_xlabel('时间', fontsize=8)
#         ax.set_ylabel('播放量变化', fontsize=8)

#     # 调整布局，防止标题重叠
#     plt.tight_layout()
#     plt.suptitle(f"播放量折线图 - 第 {page + 1} 页", fontsize=16, y=1.03)

#     # 显示图表
#     plt.show()

