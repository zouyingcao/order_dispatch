import pandas as pd
from algorithm.AC import *
from simulator.envs import *
from algorithm import RL_utils
import matplotlib.pyplot as plt

actor_lr = 1e-3  # actor网络学习率
critic_lr = 1e-2  # critic网络学习率
num_episode = 30   # 30天数据训练
n_step = 168    # 早上8点到晚上10点, 每五分钟一个时间区间, 从0到167
batch_size = int(5e+2)  # 5*10^2=500, 每次取出500条经验来更新网络
capacity = int(1e+6)  # 1.0*10^6=1000000, 经验池的capacity
gamma = 0.98  # reward中订单效率的折扣因子
alpha = 0.3  # reward中骑手公平的权重
beta = 0.3  # reward中商家公平的权重

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
# CUDA_LAUNCH_BLOCKING = "1"

# 订单包括订单id，订单起点经度，起点纬度，终点经度，终点纬度，产生时间
shop_data = pd.read_csv("finalOrderData/shopInitialization.csv")  # 从30天数据中提取出来的相关商家（包括经纬度、所在路网网格ID）
mapped_matrix_int = np.arange(0, 100, 1).reshape([10, 10])  # 用一个矩形来网格化配送地图，不可送达地（湖泊、海洋等）标记-1
env_name = 'order_dispatch'
env = Environment(shop_data, mapped_matrix_int, 10, 10)
# env = torch.load("../models/Environment/env_after_2_episode.pth")
# torch.manual_seed(0)

# actor网络输入层维数：env.n_valid_nodes * 3 + 4 + 7, 订单、骑手、商家、用户分布, 骑手位置, 时间, 订单数, action (7)
# critic网络输入层维数：env.n_valid_nodes * 3 + 4, 无action
action_dim = 7 + 40  # 起点、终点(经度, 纬度), cost, 距离, 同节点订单数目, route
state_dim = env.n_valid_nodes * 3 + 4

agent = ActorCritic(state_dim, action_dim, actor_lr, critic_lr, gamma, device)
replay = RL_utils.ReplayMemory(capacity, batch_size)
# agent = torch.load("../models/AC/model_after_2_episode.pth")
# replay = torch.load("../models/Buffer/buffer_after_2_episode.pth")
reward_list, return_list = RL_utils.train_on_policy_agent(env, agent, n_step, num_episode, replay,
                                                          alpha, beta)

steps_list = list(range(len(reward_list)))
plt.figure(1)
plt.plot(steps_list, reward_list)
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

mv_reward = RL_utils.moving_average(reward_list, 9)
plt.figure(2)
plt.plot(steps_list, mv_reward)
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

episodes_list = list(range(len(return_list)))
plt.figure(3)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

mv_return = RL_utils.moving_average(return_list, 9)
plt.figure(4)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()
