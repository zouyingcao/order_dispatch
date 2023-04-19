"""
owner: Zou ying Cao
data: 2023-04-03
description: DQN with cost penalty
"""
import torch.nn.functional as F

from models import Actor
from monitor import Monitor
from CPO import ReplayMemory
from envs.envs import Environment
from env_utils.neighbours import *
from envs.order_dispatching import *

save_dir = 'save-dir'


class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, simulator, capacity_size, batch_size, Monitor,
                 epsilon, target_update, continue_from_file=True, save_every=1, print_updates=True):
        self.env = simulator
        self.q_net = Actor(state_dim, action_dim)  # Q网络
        # 目标网络
        self.target_q_net = Actor(state_dim, action_dim)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.replay = ReplayMemory(capacity_size, batch_size)
        self.monitor = Monitor
        self.mean_rewards = []
        self.mean_costs = []
        self.mean_value_loss = []
        self.episode_num = 19
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.continue_from_file = continue_from_file
        self.save_every = save_every
        self.print_updates = print_updates
        self.if_update = False
        self.batch_state = None
        self.batch_reward = None
        self.batch_action = None

        if self.continue_from_file:
            self.load_session()

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        state_ = torch.tensor(state, dtype=torch.float32)
        action_prob = self.q_net(state_)
        if np.random.random() < self.epsilon:
            c_id = torch.distributions.Categorical(action_prob).sample().item()
        else:
            maxi = max(action_prob)
            c_id = np.random.choice([index for index, t in enumerate(action_prob) if t == maxi])
        return c_id

    def update(self, states, actions, next_states, rewards):
        q_values = torch.tensor([])
        for i in range(len(states)):
            s = torch.tensor(states[i], dtype=torch.float32)
            q_value = self.q_net(s).gather(0, actions[i])
            q_values = torch.cat((q_values, q_value), 0)  # Q值

        # 下个状态的最大Q值
        next_states = torch.tensor(next_states, dtype=torch.float32)
        max_next_q_values = max(self.target_q_net(next_states)).view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values  # TD误差目标

        dqn_loss = torch.mean(F.mse_loss(q_values.view(-1, 1), q_targets))  # 均方误差损失函数

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
        return dqn_loss

    def train(self, n_episodes, n_step, alpha, beta, penalty_rate):
        trajectory_value_loss = []

        while self.episode_num < n_episodes:
            fileName = "../datasets/orderData" + str(self.episode_num + 1) + ".csv"
            orders_data = pd.read_csv(fileName)
            self.env.reset_env(orders_data)  # 出现新的订单与用户

            trajectory_value_loss.clear()
            dispatch_action = {}
            trajectory_rewards = []
            trajectory_costs = []

            for i_step in range(1, n_step + 1):
                dispatch_action.clear()
                trajectory_rewards.clear()
                trajectory_costs.clear()
                for i_order in self.env.day_orders[self.env.time_slot_index]:
                    self.env.users_dict[i_order.user_loc].create_order(i_order)  # 用户下单
                    self.env.set_node_one_order(i_order)  # 商家所在路网节点订单+1
                    state = self.env.get_region_state()  # 路网中骑手、用户、商家、订单分布(其中商家分布不变)
                    shop_node_id = i_order.begin_p  # 订单所在商家的对应node_id
                    x, y = ids_1dto2d(shop_node_id, self.env.M, self.env.N)
                    order_num = state[2][x][y]  # 附近的订单个数

                    id_list, couriers, wait_time, action = self.env.action_collect(x, y, i_order, order_num)
                    if len(couriers) == 0:  # 8邻域无可用的骑手, 多一层邻域查找
                        for layer in range(6):
                            id_list, couriers, wait_time, action = self.env.more_action_collect(layer + 2, x, y,
                                                                                                i_order, order_num)
                            if len(couriers):
                                break
                    if len(couriers):
                        state_couriers = get_courier_state(self.env, state, couriers)
                        state_input = get_state_input(state_couriers, action)  # state, action拼接

                        # if self.if_update:
                        #     loss = self.update(self.batch_state, self.batch_action, state_input, self.batch_reward)
                        #     trajectory_value_loss.append(loss)
                        #     self.if_update = False

                        c_id = self.take_action(state_input)  # , softmax_V

                        i_order.set_order_accept_time(self.env.time_slot_index)  # 订单设置接单时间
                        self.env.shops_dict[i_order.shop_loc].add_order(i_order)  # 相应商家order_list添加该订单
                        # courier部分状态更新包含num_order与接待订单的时间点属性
                        self.env.couriers_dict[id_list[c_id]].take_order(i_order, self.env)
                        self.env.nodes[i_order.begin_p].order_num -= 1

                        d = DispatchPair(i_order, self.env.couriers_dict[id_list[c_id]])
                        d.set_state_input(state_input)  # [c_id]
                        d.set_state(state_couriers[c_id])
                        d.set_action(c_id)  # int
                        d.set_cost(wait_time[c_id])  # 0:不超时, 1:超时

                        record = False
                        if i_step == n_step and self.env.time_slot_index == 167:
                            record = True
                        d.set_penalty_reward(self.env, alpha, beta, penalty_rate, self.episode_num, record)

                        trajectory_rewards.append(d.reward)
                        trajectory_costs.append(d.cost)
                        dispatch_action[i_order] = d
                    else:
                        if self.env.time_slot_index != 167:
                            self.env.day_orders[self.env.time_slot_index + 1].append(i_order)  # 放到下一时刻再分配
                        else:
                            print('one order dismissed')

                if len(dispatch_action):
                    dispatch_result = self.env.step(dispatch_action)  # state为路网的next_state
                    # state_inputs, states, actions, rewards, next_states, costs = process_memory(
                    #     self.env.state, dispatch_result)
                    # self.replay.add(state_inputs, states, actions, rewards, next_states, costs)
                    self.mean_rewards.append(torch.mean(torch.Tensor(trajectory_rewards)))
                    self.mean_costs.append(torch.mean(torch.Tensor(trajectory_costs)))
                    self.monitor.update_reward(self.episode_num * n_step + i_step, self.mean_rewards[-1])
                    self.monitor.update_cost(self.episode_num * n_step + i_step, self.mean_costs[-1])

                # if i_step >= 80 and i_step != 167 and i_step % 5 == 0:
                #     self.batch_state, _, batch_reward, _, _, batch_action = self.replay.sample()
                #
                #     self.batch_action = torch.LongTensor(batch_action).view(-1, 1)
                #     self.batch_reward = torch.tensor(batch_reward, dtype=torch.float32).view(-1, 1)
                #     self.if_update = True

            self.episode_num += 1

            # self.mean_value_loss.append(torch.mean(torch.Tensor(trajectory_value_loss)))
            # self.monitor.record_loss(self.episode_num, self.mean_value_loss[-1], False)  # true/false:cost/value

            self.env.update_env()

            if self.print_updates:
                self.print_update()
            # if self.save_every and not self.episode_num % self.save_every:
            #     self.save_session()

    def save_session(self):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, 'dqn_model0403' + '.pt')

        ptFile = dict(policy_state_dict=self.q_net.state_dict(),
                      mean_rewards=self.mean_rewards,
                      mean_value_loss=self.mean_value_loss,
                      episode_num=self.episode_num)

        torch.save(ptFile, save_path)

    def load_session(self):
        actor_load_path = os.path.join(save_dir, 'dqn_model_0403' + '.pt')
        actor_ptFile = torch.load(actor_load_path)

        self.q_net.load_state_dict(actor_ptFile['policy_state_dict'])

    def print_update(self):
        update_message = '[Episode]: {0} | [Avg. Reward]: {1}'
        format_args = (self.episode_num, self.mean_rewards[-1])
        print(update_message.format(*format_args))


if __name__ == '__main__':
    # setup the environment
    shop_data = pd.read_csv("../shopInitialization.csv")  # 从30天数据中提取出来的相关商家（包括经纬度、所在路网网格ID）
    mapped_matrix_int = np.arange(0, 100, 1).reshape([10, 10])  # 用一个矩形来网格化配送地图，不可送达地（湖泊、海洋等）标记-1
    environment = Environment(shop_data, mapped_matrix_int, 10, 10)
    monitor = Monitor(train=True, spec="DQN_Dispatch")

    # get state / action size
    state_size = environment.n_valid_nodes * 3 + 4
    action_size = 7 + 40  # 起点、终点(经度, 纬度), cost, 距离, 同节点订单数目, route
    batch = int(5e+2)  # 5*10^2=500, 每次取出500条经验来更新网络
    capacity = int(1e+4)  # 1.0*10^4=10000, 经验池的capacity

    dqn = DQN(state_dim=state_size, action_dim=action_size, simulator=environment,
              capacity_size=capacity, batch_size=batch, Monitor=monitor,
              learning_rate=1e-2, gamma=0.995, epsilon=0.02, target_update=5)

    # for train
    n_episodes = 20
    n_steps = 168  # 一天内
    alpha = 0.3  # reward中骑手公平的权重
    beta = 0.3  # reward中商家公平的权重
    penalty_rate = 1

    dqn.train(n_episodes, n_steps, alpha, beta, penalty_rate)
