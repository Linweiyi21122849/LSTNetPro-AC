import numpy as np
import math
import matplotlib.pyplot as plt

Wl = 100000
SNR1 = 72.27
e1 = 100

Ws = 20000
SNR2 = 28.83
e2 = 50
esen = 0.00015
est = 0.0734

C = 102400 #KB
sizes = np.load('data/video_sizes.npy') #前300个视频文件的大小 4000Bytes-32768Bytes
y_i = np.load('data/Video300_Time500.npy',allow_pickle=True) #前300个文件的500个时间段的播放量
print(y_i[:, 274])
y_i = y_i / (np.sum(y_i, axis=0, keepdims=True)+ 1e-11)  # L1 归一化
np.savetxt(r'动态规划方法\Video300_Time500.txt', y_i, "%.10f")

t1 = sizes * 8.0 /(Wl * math.log2(1 + 10**(SNR1/10.0)))
E1 = t1 * e1

t2 = sizes * 8.0 /(Ws * math.log2(1 + 10**(SNR2/10.0)))
E2 = E1 + t2 * e2 + esen * sizes * 8 + est
E1 = E1.reshape(-1,1)
E2 = E2.reshape(-1,1)

r_del = y_i * (E2 - E1)

# r_del =  r_del[:,0]
# import pandas as pd
# # 创建 DataFrame
# df = pd.DataFrame({
#     "id": [f"item{i+1}" for i in range(len(sizes))],  # 生成 item1, item2, ...
#     "value": r_del.astype(float),  # 确保 value 为整数
#     "weight": sizes.astype(int)  # 确保 weight 为整数
# })
# # 保存为 CSV 文件
# df.to_csv("items.csv", index=False)

# print("CSV 文件已保存！")
# """""
# DQN 算法
# state: 这个游戏的状态用300个数字表示,表示视频文件是否已经存入服务器,形状为(300,),0是未存入,1是已经存入,一个state可以由y_i[:,t]取出一个时间片得到
# action: 表示改变视频文件存放在服务器中的状态, 形状为(300,),不是0就是1
# reward:
#     如果说action导致文件装不下,reward为-10000
#     反之则计算改变值
#     if sum(action *sizes) >= C:
#         reward = -10000
#         over = True #同时游戏被迫结束
#     else:
#         reward = (action - state) * r_del[:,t]
# next_state: 因为游戏的特殊性, 直接改变成了action的样子, next_state = action
# over: 表示游戏是否结束
# """""

# # plt.plot(np.arange(300), y_i[:,1], label="yi")
# # plt.plot(np.arange(300), r_del[:,3], label="ri")
# # plt.xlabel("Index")
# # plt.ylabel("Value")
# # plt.title("Curve of r_del[1, :]")
# # plt.legend()
# # plt.show()

# """""
# 动态规划算法
# 状态定义：
#     dp[i][c]：表示前 i 个视频，在 存储容量 c 下的最大收益（减少的能耗）。
#     decision[i][c]：是否选择存储第 i 个视频（用于回溯最优解）。

# 状态转移方程：
#     不存储视频 i:
#         dp[i][c]=dp[i-1][c]
#     存储视频 i(如果有足够存储空间):
#         dp[i][c]=max(dp[i-1][c],dp[i-1][c-size(i)]+rdel[i,t])
# """""

# # 初始化 DP 数组
# n = y_i.shape[0]  # 视频数量 (300)
# timestep = y_i.shape[1]  # 时间步长 (500)

# selected_videos_list = []  # 存放每个时间段选中的视频索引
# max_rewards_list = []      # 存放每个时间段的最大收益

# # 动态规划求解
# for t in range(1):  # 遍历每个时间步
#     dp = np.zeros((C + 1))  # DP 状态数组
#     decision = np.zeros((n + 1, C + 1), dtype=int)  # 记录选择的视频

#     r = r_del[:, t]
#     for i in range(1, n + 1):  # 遍历每个视频
#         print(t,":",i)
#         for c in range(C, sizes[i-1] - 1, -1):  # 遍历存储容量，从大到小
#             if dp[c - sizes[i-1]] + r[i-1] > dp[c]: 
#                 dp[c] = dp[c - sizes[i-1]] + r[i-1]
#                 decision[i][c] = 1  # 选择存储该视频
#             else:
#                 decision[i][c] = 0  # 不存储

#     # 找到最优存储策略
#     selected_videos = []
#     c = C
#     for i in range(n, 0, -1):
#         if decision[i][c] == 1:
#             selected_videos.append(i - 1)
#             c -= sizes[i - 1]

#     print("选中的视频索引:", selected_videos)
#     print("最大收益:", dp[C])

#     # 存储当前时间段的结果
#     selected_videos_list.append(selected_videos)
#     max_rewards_list.append(dp[C])

# # 保存文件
# np.save("result/dp_index.npy", selected_videos_list)
# np.save("result/dp_rewards.npy", max_rewards_list)
