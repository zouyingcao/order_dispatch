import random
import time as t
import pandas as pd
from tqdm import tqdm

from dispatch.utility import *
from simulator.utilities import *
from dispatch.dispatch_action import *

"""
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        if batch_size <= len(self.buffer):
            transitions = random.sample(self.buffer, batch_size)
        else:
            transitions = random.sample(self.buffer, len(self.buffer)-1)
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)

    def size(self):
        return len(self.buffer)
"""


class ReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.policy = []

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0

    def add(self, s, a, r, next_s, p):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]
            self.policy = p

        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
            self.policy = np.concatenate((self.policy, p), axis=0)
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)

            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s
            self.policy[index:(index + new_sample_lens)] = p

    def sample(self):
        if self.curr_lens <= self.batch_size:
            return [self.states, self.actions, self.rewards, self.next_states, self.policy]
        indices = random.sample(range(0, self.curr_lens), self.batch_size)
        batch_s = self.states[indices]
        batch_a = self.actions[indices]
        batch_r = self.rewards[indices]
        batch_next_s = self.next_states[indices]
        batch_p = self.policy[indices]
        return [batch_s, batch_a, batch_r, batch_next_s, batch_p]


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, n_step, num_episode, replay, alpha, beta):
    step_reward_list = []  # 30*168个时间步，每个时间步的平均reward
    episode_reward_list = []  # 30天，每天的平均reward
    recorder = open("../rewards/total.txt", 'a+')  # ’a+‘方式写不覆盖，不存在会文件新建
    for i_episode in range(1, num_episode + 1):
        i_recorder_name = "../rewards/episode" + str(i_episode) + ".txt"
        i_recorder = open(i_recorder_name, 'a+')
        fileName = "finalOrderData/orderData" + str(i_episode) + ".csv"
        orders_data = pd.read_csv(fileName)
        env.reset_env(orders_data)  # 出现新的订单与用户
        episode_reward = []
        with tqdm(total=n_step, desc='Day %d' % i_episode) as progress_bar:
            for i_step in range(1, n_step + 1):
                dispatch_action = {}
                step_reward = []
                i_recorder.write("step " + str(i_step) + "\n")
                for i_order in env.day_orders[env.time_slot_index]:
                    env.users_dict[i_order.user_loc].create_order(i_order)  # 用户下单
                    env.set_node_one_order(i_order)
                    state = env.get_region_state()  # 路网中骑手、用户、商家、订单分布(其中商家分布不变)
                    shop_node_id = i_order.begin_p  # 订单所在商家的对应node_id
                    x, y = ids_1dto2d(shop_node_id, env.M, env.N)
                    order_num = state[2][x][y]

                    couriers, cost_distance, cost_time, courier_distance, route = env.action_collect(i_order)
                    if len(couriers) == 0:
                        couriers, cost_distance, cost_time, route = env.action_collect_force(i_order, courier_distance)
                        print('i_step %d：' % i_step)
                        print('一次强制派单')
                        if len(couriers) == 0:
                            print("还是没有骑手")
                    state_couriers, id_list = get_courier_state(env, state, couriers)
                    action = get_action(i_order, cost_distance, cost_time, order_num, route)
                    state_input = get_state_input(state_couriers, action)
                    # epsilon-greedy 策略
                    # c_id, softmax_V = agent.take_action_epsilon(state_input, 0.9)  # softmax_V为softmax(actor输出值)
                    c_id, softmax_V = agent.take_action(state_input)

                    i_order.set_order_accept_time(env.time_slot_index)  # 订单设置接单时间
                    env.shops_dict[i_order.shop_loc].add_order(i_order)  # 相应商家order_list添加该订单
                    env.couriers_dict[id_list[c_id]].take_order(i_order, env)  # courier部分状态更新包含了num_order与接待订单的时间点属性
                    env.nodes[i_order.begin_p].order_num -= 1

                    d = DispatchPair(i_order, env.couriers_dict[id_list[c_id]])
                    d.set_state_input(state_input[c_id])
                    d.set_state(state_couriers[c_id])
                    d.set_action(action[c_id])
                    d.set_reward(i_episode, env, agent.gamma, alpha, beta)
                    d.set_policy(softmax_V.gather(0, torch.LongTensor([[c_id]])))
                    dispatch_action[i_order] = d
                    step_reward.append(d.get_reward())
                    episode_reward.append(d.get_reward())
                    i_recorder.write(str(d.get_reward()) + "\n")

                state, dispatch_result = env.step(dispatch_action)  # state为路网的next_state

                if len(dispatch_result) != 0:
                    states, actions, rewards, next_states, policy = process_memory(state, dispatch_result)
                    replay.add(states, actions, rewards, next_states, policy)
                mean_step_reward = np.mean(step_reward)
                step_reward_list.append(mean_step_reward)
                i_recorder.write("average_reward_per_step:" + str(mean_step_reward) + "\n")

                if i_step % 10 == 0:  # 用经验池
                    for _ in range(10):
                        batch_state, batch_action, batch_reward, batch_next_s, batch_policy = replay.sample()
                        agent.update(batch_state, batch_reward, batch_next_s, batch_policy)

                progress_bar.set_postfix({'time_step': '%d' % i_step,
                                          'reward': '%.3f' % mean_step_reward})
                progress_bar.update(1)

        env.update_env()  # 重置骑手新的一天上线时间、出现位置、清空路线、工作天数+1, 从而重置day_couriers; 商家运营天数+1，
        re = np.mean(episode_reward)
        episode_reward_list.append(re)
        recorder.write(str(re) + "\n")
        i_recorder.close()
        torch.save(agent, "../models/AC/model_after_" + str(i_episode) + "_episode.pth")
    recorder.close()
    return step_reward_list, episode_reward_list
