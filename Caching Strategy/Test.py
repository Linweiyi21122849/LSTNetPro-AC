import numpy as np
import matplotlib.pyplot as plt

reward_dqn = np.load("results/Dqn/reward.npy")
reward_ddqn = np.load("results/DDqn/reward.npy")
reward_reinforce = np.load("results/Reinforce/reward.npy")
reward_actor_critic = np.load("results/Actor_critic/reward.npy")
reward_dp = np.loadtxt("动态规划方法/dp_rewards.txt")

timecost_dqn = np.load("results/Dqn/timecost.npy")
timecost_ddqn = np.load("results/DDqn/timecost.npy")
timecost_reinforce = np.load("results/Reinforce/timecost.npy")
timecost_actor_critic = np.load("results/Actor_critic/timecost.npy")
time_dp = np.loadtxt("动态规划方法/dp_times.txt")

# 加载 .npy 文件
hitrate_dqn = np.load("results/Dqn/hitrate.npy")
hitrate_ddqn = np.load("results/DDqn/hitrate.npy")
hitrate_reinforce = np.load("results/Reinforce/hitrate.npy")
hitrate_actor_critic = np.load("results/Actor_critic/hitrate.npy")
hitrate_dp = np.loadtxt("动态规划方法/dp_hitrate.txt")


# # 输出数据长度
# print(f"DQN 数据量: {len(reward_dqn)}, DDQN 数据量: {len(reward_ddqn)}, Reinforce 数据量: {len(reward_reinforce)}, AC 数据量: {len(reward_actor_critic)}, DP 数据量: {len(reward_dp)}")

# # 计算统计信息
# print("\n统计信息（完整数据）：")
# print(f"DQN: 平均值={np.mean(reward_dqn):.4f}, 方差={np.var(reward_dqn):.4f}, 最大值={np.max(reward_dqn):.4f}, 最小值={np.min(reward_dqn):.4f}")
# print(f"DDQN: 平均值={np.mean(reward_ddqn):.4f}, 方差={np.var(reward_ddqn):.4f}, 最大值={np.max(reward_ddqn):.4f}, 最小值={np.min(reward_ddqn):.4f}")
# print(f"Reinforce: 平均值={np.mean(reward_reinforce):.4f}, 方差={np.var(reward_reinforce):.4f}, 最大值={np.max(reward_reinforce):.4f}, 最小值={np.min(reward_reinforce):.4f}")
# print(f"Actor-Critic: 平均值={np.mean(reward_actor_critic):.4f}, 方差={np.var(reward_actor_critic):.4f}, 最大值={np.max(reward_actor_critic):.4f}, 最小值={np.min(reward_actor_critic):.4f}")
# print(f"DP: 平均值={np.mean(reward_dp):.4f}, 方差={np.var(reward_dp):.4f}, 最大值={np.max(reward_dp):.4f}, 最小值={np.min(reward_dp):.4f}")
# DQN 数据量: 500, DDQN 数据量: 500, Reinforce 数据量: 500, AC 数据量: 500, DP 数据量: 500
# 统计信息（完整数据）：
# DQN: 平均值=2.5153, 方差=0.2069, 最大值=3.7385, 最小值=0.8564
# DDQN: 平均值=2.5121, 方差=0.2321, 最大值=3.8929, 最小值=0.9332        
# Reinforce: 平均值=5.4514, 方差=0.0990, 最大值=6.0207, 最小值=2.0878   
# Actor-Critic: 平均值=6.0168, 方差=0.0765, 最大值=6.4170, 最小值=2.4333
# DP: 平均值=6.6001, 方差=0.0207, 最大值=6.9100, 最小值=5.4683

# # ， 就是要体现出DP虽然是最优算法，但是推理需要得时间也很多
# 计算统计信息
# methods = {
#     "DQN": timecost_dqn,
#     "DDQN": timecost_ddqn,
#     "Reinforce": timecost_reinforce,
#     "Actor-Critic": timecost_actor_critic,
#     "DP": time_dp
# }
# for name, data in methods.items():
#     print(f"{name}: 数据量={len(data)}, 平均值={np.mean(data):.4f}, 方差={np.var(data):.4f}, 最大值={np.max(data):.4f}, 最小值={np.min(data):.4f}")
# DQN: 数据量=500, 平均值=0.0196, 方差=0.0000, 最大值=0.0328, 最小值=0.0130
# DDQN: 数据量=500, 平均值=0.0192, 方差=0.0000, 最大值=0.0319, 最小值=0.0150
# Reinforce: 数据量=500, 平均值=0.0442, 方差=0.0000, 最大值=0.0691, 最小值=0.0320
# Actor-Critic: 数据量=500, 平均值=0.0429, 方差=0.0000, 最大值=0.0730, 最小值=0.0318
# DP: 数据量=500, 平均值=0.3140, 方差=0.0001, 最大值=0.3610, 最小值=0.2980


# 计算统计信息
methods = {
    "DQN": hitrate_dqn,
    "DDQN": hitrate_ddqn,
    "Reinforce": hitrate_reinforce,
    "Actor-Critic": hitrate_actor_critic,
    "DP": hitrate_dp
}
for name, data in methods.items():
    print(f"{name}: 数据量={len(data)}, 平均值={np.mean(data):.4f}, 方差={np.var(data):.4f}, 最大值={np.max(data):.4f}, 最小值={np.min(data):.4f}")

# ################################
# # # 绘制每个算法的 reward 曲线
# ################################

# plt.figure(figsize=(10, 6))
# # reward_dp_real = np.loadtxt("动态规划方法/dp_rewards_real.txt")
# reward_dp_opt = np.loadtxt("动态规划方法/optdp_rewards.txt")
# plt.plot(reward_dp_opt, label="ACL Opt", color='m')  # 灰色0虚线
# # 绘制每个算法的 reward 曲线
# plt.plot(reward_dqn, label="DQN", color='b')
# plt.plot(reward_ddqn, label="DDQN", color='g')
# plt.plot(reward_reinforce, label="Reinforce", color='r')
# plt.plot(reward_actor_critic, label="Actor-Critic", color='c')
# # plt.plot(reward_dp, label="DP", color='m')
# # 添加标题和标签
# plt.title("Rewards of Different Methods", fontsize=14)
# plt.xlabel("Time Stamp", fontsize=12)
# plt.ylabel("Reward", fontsize=12)
# # 添加图例
# plt.legend()
# # plt.savefig(f"results/Pictures/cache_reward.png", format="png")
# plt.savefig(f"results/Pictures/cache_reward.eps", format="eps")
# # 显示图形
# plt.show()

# ###############################
# # 绘制每个算法的 timecost 曲线
# ###############################
# plt.figure(figsize=(10, 6))
# plt.plot(timecost_dqn, label="DQN", color='b')
# plt.plot(timecost_ddqn, label="DDQN", color='g')
# plt.plot(timecost_reinforce, label="Reinforce", color='r')
# plt.plot(timecost_actor_critic, label="Actor-Critic", color='c')
# plt.plot(time_dp, label="ACL Opt", color='m')
# # 添加标题和标签
# plt.title("Timecosts of Different Methods", fontsize=14)
# plt.xlabel("Time Stamp", fontsize=12)
# plt.ylabel("Timecost", fontsize=12)
# # 添加图例
# plt.legend(loc='upper right')
# # plt.savefig(f"results/Pictures/cache_timecost.png", format="png")
# # plt.savefig(f"results/Pictures/cache_timecost.eps", format="eps")
# # 显示图形
# plt.show()

# ###############################
# # 绘制每个算法的 hitrate 曲线
# ###############################
# plt.figure(figsize=(10, 6))
# plt.plot(hitrate_dqn, label="DQN", color='b')
# plt.plot(hitrate_ddqn, label="DDQN", color='g')
# plt.plot(hitrate_reinforce, label="Reinforce", color='r')
# plt.plot(hitrate_actor_critic, label="Actor-Critic", color='c')
# plt.plot(hitrate_dp, label="ACL Opt", color='m')
# # 添加标题和标签
# plt.title("hitrates of Different Methods", fontsize=14)
# plt.xlabel("Hitrate", fontsize=12)
# plt.ylabel("hitrate", fontsize=12)
# # 添加图例
# plt.legend(loc='upper right')
# plt.savefig(f"results/Pictures/cache_Hitrate.png", format="png")
# # plt.savefig(f"results/Pictures/cache_Hitrate.eps", format="eps")
# # 显示图形
# plt.show()
