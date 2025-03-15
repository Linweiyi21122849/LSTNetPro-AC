import numpy as np

# 随机生成符合正态分布的300个视频文件大小
np.random.seed(42)  # 设置随机种子以便结果可复现
mean_size = (500 + 4048) / 2  # 平均值
std_dev = (4048 - 500) / 6  # 假设99.7%的值落在范围内，6倍标准差

# 生成正态分布随机数
sizes = np.random.normal(loc=mean_size, scale=std_dev, size=300)

# 限制范围在4000KB到32768KB
sizes = np.clip(sizes, 500, 4048)
sizes = np.round(sizes).astype(int)
print(sizes)
# # 保存数组到文件
# file_path = "data/video_sizes.npy"
# np.savetxt(r"动态规划方法\video_sizes.txt", sizes, fmt="%d")
np.save("data/video_sizes.npy", sizes)
