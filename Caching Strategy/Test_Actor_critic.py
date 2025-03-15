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

from torch.distributions import Categorical

np.set_printoptions(suppress=True)

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
y_i = np.load('data/LSTNet_predict.npy', allow_pickle=True)  # 300个文件在500个时间段的播放量
y_i = y_i[:,:500]
print(y_i.shape)
# 归一化播放量，使得每个时间段所有视频的播放量之和为1
y_i = y_i / (np.sum(y_i, axis=0, keepdims=True) + 1e-11)  # L1 归一化

y_i_real = np.load('data/LSTNet_predict.npy', allow_pickle=True)  #引入真实标签仅仅用来计算cache_hit_rate
y_i_real = y_i_real[:,:500]
y_i_real = y_i_real / (np.sum(y_i_real, axis=0, keepdims=True) + 1e-11)  # L1 归一化

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
        self.num_items = len(values) #视频的数量

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
        ret = ret.reshape(1, len(ret)) #fixed
        return ret
    
    # 重置环境，初始化状态
    def reset(self, t=None):
        if t == None:
            self.t = random.randint(0, 499)  # 选择500个时间步中的随机一个作为初始时间步
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
    
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim , action_dim , hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x):
        x = self.network(x) 
        x = torch.tanh(x) * 10 # 限制 logits 取值在 [-10, 10] 之间
        return x

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出一个值
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)  # 输出形状 (batch,)
    
class REINFORCEAgent:
    def __init__(
        self, 
        state_dim: int, # 定义状态的维度
        item_dim: int, # 每个物品的特征维度
        gamma: float = 1.0,  # 折扣因子，用于折扣回报值
        entropy_coef: float = 0.001, # 熵系数，用于控制策略的探索度，避免过早收敛到一个局部最优解
        device: str = "cpu",
        inference: bool = False,
    ):
        self.item_dim = item_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        self.inference = inference
        
        self.model_actor = ActorNetwork(state_dim, item_dim).to(self.device)
        self.model_critic = CriticNetwork(state_dim).to(self.device) # 添加值网络
        self.model_critic_delay = CriticNetwork(state_dim).to(self.device) # 添加值网络
        self.model_critic_delay.load_state_dict(self.model_critic.state_dict())

        self.optimizer_actor = optim.RMSprop(self.model_actor.parameters(), lr=1e-5)
        self.optimizer_critic = optim.Adam(self.model_critic.parameters(), lr=1e-5)  # 值网络优化器
        
        self._state_buffer = []  # 记录状态
        self._action_log_prob_buffer = []
        self._reward_buffer = []
        self._next_state_buffer = []
        self._over_buffer = []
        self._entropy_buffer = []

    def select_action(self, obs: torch.Tensor, mask) -> torch.Tensor:
        """
        根据当前的观察值选择一个动作。

        Args:
            obs 形状为 (batch, state_items)

        Returns:
            action (Tensor): `(batch,)`, the index of the selected knapsack and item
        """
        # 否则根据Q网络选择动作
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        else:
            obs = obs.clone().detach().to(self.device)
        obs = (obs - obs.mean()) / (obs.std() + 1e-8)
        logits = self.model_actor(obs.to(self.device))
        index = np.where(mask == 0)
        logits[0][index] = -float('inf')
        # print(logits[0][:50])
        policy_dist = Categorical(logits=logits) # 代表类别的 未归一化对数概率
        action = policy_dist.sample()
        # print(policy_dist.probs[0][:50])

        if not self.inference:
            self._action_log_prob = policy_dist.log_prob(action) #  计算选定动作的对数概率,就是把softmax之后的概率取ln
            self._entropy = policy_dist.entropy() # 当前 策略分布 的 熵
            # print("选定的动作的采样概率为",policy_dist.probs[0][action],"当前策略分布的熵为: ",policy_dist.entropy())
        return action.detach().cpu()
        
    def update(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, reward: torch.Tensor, terminated: torch.Tensor):
        """
        更新policy网络。

        Args:
            obs (Tensor): `(batch, state_dim)`
            action (Tensor): `(batch,)`
            next_obs (Tensor): `(batch, state_dim)`
            reward (Tensor): `(batch,)`
            terminated (Tensor): `(batch,)`
        """
        if self.inference: #在推理模式下，update 方法不会做任何更新
            return
        
        # 否则，会将当前奖励、动作的对数概率和熵保存到缓冲区, 直到回合结束时, 调用 _train() 方法进行训练
        self._state_buffer.append(obs)  # 记录状态
        self._action_log_prob_buffer.append(self._action_log_prob) # 策略概率分布
        self._reward_buffer.append(reward)
        self._next_state_buffer.append(next_obs)
        self._over_buffer.append(terminated)
        self._entropy_buffer.append(self._entropy)
        if terminated.item(): # 每一局游戏结束之后发生更新
            loss, entropy = self._train()     
            return {
                "policy_loss": loss,
                "entropy": entropy,
            }
    
    def requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad_(value)

    def train_critic(self, obs, reward, next_obs, over):
        self.requires_grad(self.model_actor, False) #actor函数就是拿来用用，不用计算梯度
        self.requires_grad(self.model_critic, True)

        #计算values和targets
        value = self.model_critic(obs)
        with torch.no_grad():
            target = self.model_critic_delay(next_obs)
        target = target * self.gamma * (1 - over) + reward

        #时序差分误差,也就是tdloss
        loss = F.mse_loss(value, target)

        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        #减去value相当于去基线
        return (target - value).detach()
    
    def train_actor(self, state, action_log_prob, advantage, entropy):
        self.requires_grad(self.model_actor, True)
        self.requires_grad(self.model_critic, False)

        # 策略网络损失
        policy_loss = (advantage * action_log_prob).mean()
        entropy = entropy.mean()

        # print(-policy_loss)
        # 添加熵正则项
        loss = -policy_loss + self.entropy_coef * entropy  # 熵项鼓励探索

        self.optimizer_actor.zero_grad()
        loss.backward()
        self.optimizer_actor.step()

        return loss.item(), entropy.item()
    
    def _train(self):
        # return  obs, action_log_prob, reward, next_obs, over, entropy
        # reward, action_log_prob, entropy, obs = self._buffer_to_tensor()
        obs, action_log_prob, reward, next_obs, over, entropy  = self._buffer_to_tensor()
        obs = obs.squeeze(0)
        action_log_prob = action_log_prob.squeeze(0)
        reward = reward.squeeze(0)
        next_obs = next_obs.squeeze(0)
        over = over.squeeze(0)
        ret = self._compute_return(reward)

        # std = ret.std() + 1e-8 if len(ret) > 1 else 1.0
        # ret = (ret - ret.mean()) / std # 归一化之后加速收敛

        td_error = self.train_critic(obs, reward, next_obs, over) # 把td_error作为优势函数
        loss, entropy = self.train_actor(obs, action_log_prob, td_error, entropy)
 
        return loss, entropy
    
    def _buffer_to_tensor(self): # 将缓冲区中的数据转换为张量，返回奖励、动作对数概率和熵的堆叠张量。
        obs = torch.stack(self._state_buffer, dim=1).to(self.device)  # 获取状态
        action_log_prob = torch.stack(self._action_log_prob_buffer, dim=1).to(self.device)
        reward = torch.stack(self._reward_buffer, dim=1).to(self.device)
        next_obs = torch.stack(self._next_state_buffer, dim=1).to(self.device)
        over = torch.stack(self._over_buffer, dim=1).to(self.device)
        entropy = torch.stack(self._entropy_buffer, dim=1).to(self.device)

        self._state_buffer.clear()
        self._action_log_prob_buffer.clear()
        self._reward_buffer.clear()
        self._next_state_buffer.clear()
        self._over_buffer.clear()
        self._entropy_buffer.clear()
        # return reward, action_log_prob, entropy, obs
        return  obs, action_log_prob, reward, next_obs, over, entropy
    
    def _compute_return(self, reward: torch.Tensor) -> torch.Tensor: # 计算未来奖励的回报
        ret = torch.empty_like(reward)
        G = 0.0
        for t in reversed(range(len(ret))):
            G = reward[t] + self.gamma * G
            ret[t] = G
        return ret

def train(env, agent: REINFORCEAgent, episodes: int, summary_freq: int):
    cumulative_reward_list = []
    policy_losses = []
    entropies = []
    total_values = []
    _start_time = time.time()
    
    for e in range(episodes): # 玩episodes局游戏
        obs = env.reset()
        obs = torch.from_numpy(obs)
        over = False
        cumulative_reward = 0.0
        mask = np.ones(300, dtype=bool)

        while not over:
            action = agent.select_action(obs, mask)
            next_obs, reward, over, _mask = env.step(action.item())
            
            next_obs = torch.from_numpy(next_obs)
            reward = torch.tensor([reward], dtype=torch.float32)
            terminated = torch.tensor([over], dtype=torch.float32)

            agent_info = agent.update( # 每一次都更新
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward=reward,
                terminated=terminated
            )
            
            mask = _mask
            obs = next_obs
            # cumulative_reward += reward.item()
            
            if agent_info is not None:
                policy_losses.append(agent_info["policy_loss"])
                entropies.append(agent_info["entropy"])

            #复制参数
            for param, param_delay in zip(agent.model_critic.parameters(),
                                        agent.model_critic_delay.parameters()):
                value = param_delay.data * 0.7 + param.data * 0.3
                param_delay.data.copy_(value)

        # if e % summary_freq == 0:
        if e % 100 == 0:
            cumulative_reward = sum([test(agent, ti=t) for t in range(20)]) / 20
            cumulative_reward_list.append(cumulative_reward)
            print(f"Training time: {time.time() - _start_time:.2f}, Episode: {e}, Cumulative Reward: {cumulative_reward}, loss: {policy_losses[-1]}, entropy: {entropies[-1]} ")
            
    return cumulative_reward_list, policy_losses, entropies

# calc = True 表示额外进行命中率推理
def test(agent, ti, calc = False):
    #初始化游戏
    obs = env.reset(t=ti)
    over = False
    cumulative_reward = 0
    mask = np.ones(300, dtype=bool)
    cache = np.zeros(300, dtype=bool)
    while not over:
        # 选择动作
        action = agent.select_action(obs, mask)
        
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
    plt.savefig(f"results/Actor_critic/inference.png")
    plt.close()

    plt.plot(time_list)
    plt.title("Timecost")
    plt.xlabel('Time Stamp')
    plt.ylabel('Timecost')
    plt.savefig(f"results/Actor_critic/inference_timecost.png")
    plt.close()

    plt.plot(hitrate_list)
    plt.title("Cache Hit Rate")
    plt.xlabel('Time Stamp')
    plt.ylabel('Cache Hit Rate')
    plt.savefig(f"results/Actor_critic/inference_hitrate.png")
    plt.close()

    np.save("results/Actor_critic/reward.npy", reward_list)
    np.save("results/Actor_critic/timecost.npy", time_list)
    np.save("results/Actor_critic/hitrate.npy", hitrate_list)
    return 

# 主函数，解析命令行参数并启动训练或推理
if __name__ == '__main__':
    episodes = 25000
    summary_freq = 100
    entropy_coef = 0.01
    # inference_problem = False
    inference_problem = True
    problem_name = "test"

    # 加载背包问题数据并初始化环境
    num_items = 300 #视频数量
    num_bags = 1 #服务器数量

    #这里到时候要改一下
    Values = r_del
    Sizes = sizes
    capacities = 102400

    # 初始化环境
    env = VideoCacheEnv(values=Values, sizes=Sizes)

    if not inference_problem:

        #初始化 DQN 模型
        agent = REINFORCEAgent(
            state_dim= num_items * 3 + 1,
            item_dim= num_items,
        )

        start_time = time.time()
        # 返回训练过程中的累计奖励, TD损失,  epsilon 值
        cumulative_reward_list, policy_losses, entropies = train(env, agent, episodes, summary_freq)
        end_time = time.time()
        train_time = end_time - start_time

        # 保存训练结果
        directory = "results/Actor_critic"
        os.makedirs(directory)
        ckpt_dict = {
            "agent": agent.model_actor.state_dict(),
            "train_time": train_time,
            "episodes": episodes,
        }
        torch.save(ckpt_dict, f"{directory}/checkpoint.pt")

        # 绘制训练结果图
        plt.plot(cumulative_reward_list)
        plt.title("Cumulative Rewards")
        plt.xlabel('Episodes * 100')
        plt.ylabel('Cumulative Reward')
        plt.savefig(f"{directory}/cumulative_rewards.png")
        plt.close()
        
        plt.plot(policy_losses)
        plt.title("Policy Losses")
        plt.xlabel('Steps')
        plt.ylabel('Policy Losses')
        plt.savefig(f"{directory}/policy_losses.png")
        plt.close()
        
        plt.plot(entropies)
        plt.title(f"Entropies")
        plt.xlabel('Steps')
        plt.ylabel('Entropy')
        plt.savefig(f"{directory}/entropies.png")
        plt.close()
    else:
        # 加载训练好的模型并进行推理
        agent = REINFORCEAgent(
            state_dim= num_items * 3 + 1,
            item_dim= num_items,
        )
        directory = f"results/Actor_critic/checkpoint.pt"
        ckpt_dict = torch.load(directory)
        agent.model_actor.load_state_dict(ckpt_dict["agent"])
        
        total_value = inference(agent)
        # test(agent, ti=377)
    
