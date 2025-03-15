#什么时候发生参数更新
#为什么rt上升了，loss也在上升
#loss很大是为什么
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import gymnasium as gym
from gymnasium import spaces
import torch.nn.functional as F
from typing import Tuple
import time

# 定义游戏相关的常量
Wl = 100000  # 大带宽 
SNR1 = 72.27  # 信噪比1
e1 = 100  # 能耗1

Ws = 20000  # 小带宽
SNR2 = 28.83  # 信噪比2
e2 = 50  # 能耗2
esen = 0.00015  # 额外能耗
est = 0.0734  # 传输延迟

C = 102400  # 服务器缓存容量 (KB)

# 加载数据
sizes = np.load('data/video_sizes.npy')  # 300个视频文件的大小 (4000Bytes-32768Bytes)
# y_i = np.load('data/Video300_Time500.npy', allow_pickle=True)  # 300个文件在500个时间段的播放量
y_i = np.load('data/LSTNet_predict.npy', allow_pickle=True)  # 300个文件在500个时间段的播放量
y_i = y_i[:,:500]
# 归一化播放量，使得每个时间段所有视频的播放量之和为1
y_i = y_i / (np.sum(y_i, axis=0, keepdims=True) + 1e-11)  # L1 归一化

y_i_real = np.load('data/LSTNet_predict.npy', allow_pickle=True)  #引入真实标签仅仅用来计算cache_hit_rate
y_i_real = y_i_real[:,:500]
y_i_real = y_i_real / (np.sum(y_i_real, axis=0, keepdims=True) + 1e-11)  # L1 归一化

# 归一化播放量，使得每个时间段所有视频的播放量之和为1
y_i = y_i / (np.sum(y_i, axis=0, keepdims=True) + 1e-11)  # L1 归一化

# 计算两个带宽下的传输时间和能耗
t1 = sizes * 8.0 / (Wl * math.log2(1 + 10**(SNR1 / 10.0)))
E1 = t1 * e1

t2 = sizes * 8.0 / (Ws * math.log2(1 + 10**(SNR2 / 10.0)))
E2 = E1 + t2 * e2 + esen * sizes * 8 + est  # 计算总能耗

# 重新调整形状以适应后续计算
E1 = E1.reshape(-1, 1)
E2 = E2.reshape(-1, 1)

# 计算奖励变化量 (E2 - E1)
r_del = y_i * (E2 - E1)

# 定义游戏环境类
class VideoCacheEnv(gym.Env):
    def __init__(self, sizes, values, t=None):
        super(VideoCacheEnv, self).__init__()
        #定义当前游戏的时间戳
        if t == None:
            self.t = random.randint(0, 499)  # 选择500个时间步中的随机一个作为初始时间步
        else:
            self.t = t
        # 初始化物品的价值和重量，以及背包的容量
        self.capacities = np.array([102400])  # 背包的容量列表
        self.values = np.array(values[:,self.t]) # 视频存放的奖励
        self.sizes = np.array(sizes) # 视频的大小
        self.num_items = len(sizes) #视频的数量

        # 定义动作空间
        self.action_space = spaces.Discrete(self.num_items)

        # 定义观察空间，包含以下几个部分：
        # - 物品的价值
        # - 物品的重量
        # - 背包的剩余容量
        # - 物品是否被选择

        self.observation_space = spaces.Dict({
            'item_values': spaces.Box(low=0, high=np.inf, shape=(self.num_items,), dtype=np.float32),
            'item_weights': spaces.Box(low=0, high=np.inf, shape=(self.num_items,), dtype=np.float32),
            'remaining_capacities': spaces.Box(low=0, high=np.max(capacities), shape=(1,), dtype=np.float32),
            'selection_status': spaces.MultiBinary(self.num_items)  # 每个物品是否被选择
        })

        self.time_step = -1  # 初始化时间步
        self.state = None  # 初始化状态
        self.reset()  # 调用重置函数

    # 准备当前状态的观测信息
    def prepare_state(self):
        # 获取状态中每一部分的信息
        item_values = np.array(self.state['item_values'], dtype=np.float32)
        item_weights = np.array(self.state['item_weights'], dtype=np.float32)
        remaining_capacities = np.array(self.state['remaining_capacities'], dtype=np.float32)
        selection_status = np.array(self.state['selection_status'], dtype=np.float32)
        
        # 将这些信息拼接成一个一维数组并返回
        ret = np.concatenate([item_values, item_weights, remaining_capacities, selection_status])
        # ret = ret.reshape(1, len(ret)) #fixed
        return ret
    
    # 重置环境，初始化状态
    def reset(self, t=None):
        if t == None:
            self.t = random.randint(0, 499)  # 选择500个时间步中的随机一个作为初始时间步
            # print(f"当局游戏进行数据的的时间戳为{self.t}")
        else:
            self.t = t
        self.values = np.array(r_del[:, self.t]) # 视频存放的奖励
        # self.values = np.array(r_del[:, 0]) # 视频存放的奖励
        self.sizes = np.array(sizes) # 视频的大小
        # 初始化每个物品的价值、重量、背包的剩余容量和每个物品的选择状态
        self.state = {
            'item_values': self.values,  # 物品的价值, 静态状态
            'item_weights': self.sizes,  # 物品的重量， 静态态状态
            'remaining_capacities': self.capacities.copy(),  # 每个背包的剩余容量， 动态状态
            'selection_status': np.zeros(self.num_items, dtype=int)  # 物品是否被选择，初始时全为 0
        }
        self.time_step = -1  # 重置时间步
        return self.prepare_state()  # 返回当前状态
    
    # 表示所有物品存放的有效性的掩码
    def valid_actions(self):
        mask = np.zeros(self.num_items, dtype=bool)
        # 检查当前物品是否可以放入当前背包
        for i in range(self.num_items):
                if self.state["selection_status"][i] == 0 and self.state['remaining_capacities'] >= self.sizes[i]:
                    mask[i] = True  # 如果可以放入，标记该动作为有效
        # return mask.reshape(1, len(mask))  # 返回有效动作的掩码 fixed
        return mask  # 返回有效动作的掩码
    
    # 执行动作，并返回新的状态、奖励、是否完成、有效动作的掩码
    def step(self, action):
        # 根据动作计算选择的物品和背包
        # item_idx = action[0] #fixed
        item_idx = action
        
        # 获取该物品的价值和重量
        item_value = self.values[item_idx]
        item_weight = self.sizes[item_idx]
        
        # 如果物品还没有被选择，并且当前背包有足够容量来放入该物品
        if self.state['selection_status'][item_idx] == 0 and self.state['remaining_capacities'] >= item_weight:
            # 更新背包的剩余容量和物品的选择状态
            self.state['remaining_capacities'] -= item_weight
            self.state['selection_status'][item_idx] = 1
            reward = item_value  # 如果成功放入物品，奖励为物品的价值
        else:
            reward = 0  # 如果物品不能放入背包，则奖励为 0
        
        # 检查是否所有物品都已被选择，或者没有足够空间放入更多物品
        no_more_fits = np.all(self.state['remaining_capacities'] < np.min(self.sizes[self.state['selection_status'] == 0]))
        done = np.all(self.state['selection_status']) or no_more_fits  # 如果所有物品都选择了，或者没有更多物品可以放入背包，则结束

        # 获取有效的动作掩码
        mask = self.valid_actions()

        return self.prepare_state(), reward, done, mask

# ReplayBuffer用于存储经验回放数据
class ReplayBuffer:
# obs: 当前状态 s_t，形状 (batch_size, obs_dim)
# action: 执行动作 a_t，形状 (batch_size,)
# next_obs: 下一状态 s_{t+1}，形状 (batch_size, obs_dim)
# reward: 奖励 r_t，形状 (batch_size,)
# over: 是否终止（终止=1，未终止=0），形状 (batch_size,)
    def __init__(self, obs_dim: int, max_size: int, batch_size: int) -> None:
        # 初始化ReplayBuffer的属性
        self.max_size = max_size
        self.batch_size = batch_size
        self.size = 0
        self.idx = 0
        
        # 为每种数据（obs, action, next_obs, reward, terminated）分配缓冲区
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((max_size,), dtype=np.int64)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size,), dtype=np.float32)
        self.over_buf = np.zeros((max_size,), dtype=np.float32)
        
    def store(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, over: np.ndarray):
        # 将新的经验存入缓冲区
        batch_size = obs.shape[0]
        idxs = np.arange(self.idx, self.idx + batch_size) % self.max_size #生成 batch_size 个连续索引（从 self.idx 开始）
        
        self.obs_buf[idxs] = obs
        self.action_buf[idxs] = action
        self.next_obs_buf[idxs] = next_obs
        self.reward_buf[idxs] = reward
        self.over_buf[idxs] = over
        
        self.idx = (self.idx + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
        
    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 从缓冲区中随机抽取一个batch的经验
        idxs = np.random.choice(self.size, self.batch_size, replace=False) #随机抽取 batch_size 个索引，从 0 ~ self.size-1 之间采样
        obs = self.obs_buf[idxs]
        # print(obs.shape)
        action = self.action_buf[idxs]
        next_obs = self.next_obs_buf[idxs]
        reward = self.reward_buf[idxs]
        over = self.over_buf[idxs]
        
        return obs, action, next_obs, reward, over

# QNetwork类定义了Q值网络
class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=256) -> None:
        super(QNetwork, self).__init__()
        
        # 定义一个简单的前馈神经网络
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 前向传播，输出Q值
        return self.model(obs)

# DQN类实现了深度Q网络
# 3 * num_items + num_bags, num_bags * num_items, num_items
class DDQN:
    def __init__(self, obs_dim, n_actions) -> None:
        # 初始化DQN的各种超参数和网络
        self.obs_dim = obs_dim  # 3 * 300 + 1
        self.n_actions = n_actions # 1 * 300
        self.device = torch.device("cpu")
        self.epoch = 3
        self.lr = 1e-3 
        self.gamma = 0.99 #因为reward很小所以要缩小折扣函数
        # self.gamma = 0.99 #因为reward很小所以要缩小折扣函数
        self.batch_size = 32 
        self.eps = 1.0
        self.eps_decay = 0.995
        self.min_eps = 0.01
        self.target_net_update_freq = 100
        self.time_step = -1
        self.max_size = 1000 #缓冲区的容量
        # 初始化Q网络和目标网络
        self.model = QNetwork(obs_dim, n_actions).to(self.device)
        self.next_model = QNetwork(obs_dim, n_actions).to(self.device)
        self.next_model.load_state_dict(self.model.state_dict())
        
        # 初始化优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(obs_dim, self.max_size, self.batch_size)
        
    @torch.no_grad() #得到动作不需要梯度计算
    def select_action(self, obs: np.ndarray, mask) -> np.ndarray:
        # epsilon-贪心策略: 以一定的概率选择随机动作
        if np.random.rand() < self.eps:
            rand_logits = np.random.rand(self.n_actions) #生成(batch, self.n_action)的形状
            rand_logits[(~torch.BoolTensor(mask))] = -float('inf') #会将无效动作的概率设为负无穷，确保这些无效动作不能被选择。
            return np.argmax(rand_logits, axis=-1)
    
        # 否则根据Q网络选择动作
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        else:
            obs = obs.clone().detach().to(self.device)
        q_values = self.model(obs)
        # q_values[~torch.BoolTensor(mask.reshape(1, 300))] = -float('inf') #fixed
        q_values[~torch.BoolTensor(mask)] = -float('inf') 
        return torch.argmax(q_values, dim=-1).cpu().numpy()
    
    def update(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, over: np.ndarray):
        self.time_step += 1
        self.replay_buffer.store(obs, action, next_obs, reward, over)
        if self.replay_buffer.size < self.replay_buffer.batch_size: # 还达不到训练要求的最小batch数量
            return
        td_losses = self._train()
        return {
            "td_losses": td_losses,
            "eps": self.eps,
        }
        
    def _train(self):
        td_losses = []
        for _ in range(self.epoch):
            # 从经验回放缓冲区中抽取数据
            obs, action, next_obs, reward, over = self.replay_buffer.sample()
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            over = torch.tensor(over, dtype=torch.float32).to(self.device)

            q_values = self.model(obs)  # 预测当前状态的 Q 值
            q_values = q_values.gather(dim=1, index=action.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_actions = torch.argmax(self.model(next_obs), dim=-1, keepdim=True)  # 选择最佳动作
                next_q_values = self.next_model(next_obs).gather(dim=1, index=next_actions).squeeze(-1)
                # 计算 Q 目标值
                q_target_values = reward + (1 - over) * self.gamma * next_q_values
            

            td_loss = F.mse_loss(q_values, q_target_values)
            # 反向传播优化
            self.optimizer.zero_grad()
            td_loss.backward()
            self.optimizer.step()

            td_losses.append(td_loss.item())
        
        # 更新目标网络
        self._update_target_net()
        self.eps = max(self.eps * self.eps_decay, self.min_eps)
        
        return td_losses
    
    def _update_target_net(self):
        # 每隔一定步数更新目标网络的权重
        if self.time_step % self.target_net_update_freq == 0:
            self.next_model.load_state_dict(self.model.state_dict())

# 训练函数，执行多个训练回合
def train(env, dqn, episodes):
    td_loss_list = []
    cumulative_reward_list = []
    epsilons = []
    start_time = time.time()
    for e in range(episodes): #玩一百局游戏, 每一局用不同的视频参数r_del[:,t]
        obs = env.reset()
        over = False
        cumulative_reward = 0
        # mask = np.zeros((1, 300), dtype=bool) #fixed
        mask = np.zeros(300, dtype=bool)
        while not over:
            # 选择动作
            action = dqn.select_action(obs, mask)
            
            # 执行动作并观察下一个状态和奖励
            next_obs, reward, over, _mask = env.step(action)
            
            # 更新DQN
            dqn_info = dqn.update(obs, action, next_obs, reward, over) #每次更新time_step+1, 加入一条小数据(使用当局游戏的time),同时从缓冲区提取出一组batch进行更新
            if dqn_info is not None:
                tmploss = np.mean(dqn_info["td_losses"])
                td_loss_list.append(tmploss)
                epsilons.append(dqn_info["eps"])
            # 转移到下一个状态
            obs = next_obs
            mask = _mask
            # cumulative_reward += reward
        
        cumulative_reward = sum([test(dqn, ti=t) for t in range(20)]) / 20
        cumulative_reward_list.append(cumulative_reward)
        print(f'episode: {e}, cumulative_reward : {cumulative_reward},loss:{tmploss}')
        # print(f'episode: {e}, cumulative_reward : {cumulative_reward[0]},loss:{tmploss}')
    return cumulative_reward_list, td_loss_list, epsilons

# calc = True 表示额外进行命中率推理
def test(dqn, ti, calc = False):
    #初始化游戏
    obs = env.reset(t=ti)
    over = False
    cumulative_reward = 0
    mask = np.zeros(300, dtype=bool)
    cache = np.zeros(300, dtype=bool)
    while not over:
        # 选择动作
        action = dqn.select_action(obs, mask)
        # 执行动作并观察下一个状态和奖励
        next_obs, reward, over, _mask = env.step(action)
        if reward != 0.0:
            cache[action] = 1
        
        # 转移到下一个状态
        obs = next_obs
        mask = _mask
        cumulative_reward += reward
    
    if calc == False:
        return cumulative_reward
    else:
        return cumulative_reward, calc_rate(cache, ti)

def calc_rate(mask, ti):
    global y_i_real  # 确保 y_i_real 变量已定义
    prob = y_i_real[:, ti]  # (300,)

    # 采样这么nums次, 蒙特卡洛法计算
    # nums = 100000
    # sampled_indices = np.random.choice(300, size=nums, p=prob)
    # hit_count = np.sum(mask[sampled_indices] == 1)  # 统计命中次数
    # print(hit_count / nums)
    # hit_rate = hit_count / nums  # 计算命中率
    
    # 概率公式计算
    indices = np.where(mask == 1)[0]
    hit_rate = np.sum(prob[indices])
    return hit_rate
    
# 执行推理，获取模型在环境中的表现
def inference(agent):
    reward_list = []
    time_list = []
    hitrate_list = []
    
    for t in range(500):
        start_time = time.time()
        reward, hitrate = test(agent, ti=t, calc=True)
        end_time = time.time()
        T = end_time - start_time
        print(f'Time Stamp: {t}, Reward : {reward}, CacheHitRate : {hitrate}, time : {T}')
        reward_list.append(reward)
        time_list.append(T)
        hitrate_list.append(hitrate)
    
    # 绘制训练结果图
    plt.plot(reward_list)
    plt.title("Rewards")
    plt.xlabel('Time Stamp')
    plt.ylabel('Reward')
    plt.savefig(f"results/Ddqn/inference.png")
    plt.close()

    plt.plot(time_list)
    plt.title("Timecost")
    plt.xlabel('Time Stamp')
    plt.ylabel('Timecost')
    plt.savefig(f"results/Ddqn/inference_timecost.png")
    plt.close()

    plt.plot(hitrate_list)
    plt.title("Cache Hit Rate")
    plt.xlabel('Time Stamp')
    plt.ylabel('Cache Hit Rate')
    plt.savefig(f"results/Ddqn/inference_hitrate.png")
    plt.close()

    np.save("results/Ddqn/reward.npy", reward_list)
    np.save("results/Ddqn/timecost.npy", time_list)
    np.save("results/Ddqn/hitrate.npy", hitrate_list)
    return

# 主函数，解析命令行参数并启动训练或推理
if __name__ == '__main__':
    episodes = 200
    eps_decay = 0.995
    # inference_problem = False
    inference_problem = True
    problem_name = "test"
    
    num_items = 300 #视频数量
    num_bags = 1 #服务器数量

    # Values = r_del[:,0]
    Values = r_del
    Sizes = sizes
    capacities = 102400

    # 初始化环境
    env = VideoCacheEnv(values=Values, sizes=Sizes)
    if not inference_problem:
        #初始化 DDQN 模型
        dqn = DDQN(3 * num_items + 1, num_items)

        start_time = time.time()
        # 返回训练过程中的累计奖励, TD损失,  epsilon 值
        cumulative_reward_list, td_loss_list, epsilons = train(env, dqn, episodes=episodes)
        end_time = time.time()
        train_time = end_time - start_time

        # 保存训练结果
        directory = "results/Ddqn/"
        os.makedirs(directory)
        ckpt_dict = {
            "agent": dqn.model.state_dict(),
            "train_time": train_time,
            "episodes": episodes,
        }
        torch.save(ckpt_dict, f"{directory}/checkpoint.pt")

        # 绘制训练结果图
        plt.plot(cumulative_reward_list)
        plt.title("Cumulative Rewards")
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Reward')
        plt.savefig(f"{directory}/cumulative_rewards.png")
        plt.close()
        
        plt.plot(td_loss_list)
        plt.title("TD Losses")
        plt.xlabel('Steps')
        plt.ylabel('TD Loss')
        plt.savefig(f"{directory}/td_losses.png")
        plt.close()
        
        plt.plot(epsilons)
        plt.title(f"Epsilon Decay {eps_decay}")
        plt.xlabel('Steps')
        plt.ylabel('Epsilon')
        plt.savefig(f"{directory}/epsilons.png")
        plt.close()

    else:
        # 加载训练好的模型并进行推理
        dqn = DDQN(3 * num_items + 1, num_items)
        directory = f"results/DDqn/checkpoint.pt"
        ckpt_dict = torch.load(directory)
        dqn.model.load_state_dict(ckpt_dict["agent"])
        dqn.next_model.load_state_dict(ckpt_dict["agent"])
        
        total_value = inference(dqn)