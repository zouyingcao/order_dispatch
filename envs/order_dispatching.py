"""
owner: Zou ying Cao
data: 2023-02-01
description:
"""
import torch
import numpy as np
import pandas as pd


def get_courier_state(env, state, couriers):
    state = np.array(state).flatten()
    state_couriers = np.zeros((len(couriers), env.n_valid_nodes * 3 + 4))
    state_couriers[:, :env.n_valid_nodes * 3] = np.stack([state] * len(couriers))  # state：n_valid_nodes*3=10*10*3=300
    state_couriers[:, env.n_valid_nodes * 3:env.n_valid_nodes * 3 + 4] = couriers
    return state_couriers


def get_action(i_order, cost_distance, cost_time, order_num, route):
    shop_lat = i_order.shop_latitude
    shop_long = i_order.shop_longitude
    user_lat = i_order.user_latitude
    user_long = i_order.user_longitude
    actions = []
    for i in range(len(cost_distance)):
        action = [shop_lat, shop_long, user_lat, user_long, cost_distance[i][1], cost_time[i][1], order_num]
        # print(len(route[i][1]))
        action.extend(route[i][1])
        actions.append(action)

    return np.array(actions)


def get_state_input(state, action):
    state_inputs = np.zeros((len(state), 351))
    state_inputs[:, :304] = state
    state_inputs[:, -47:] = action  # capacity*4=40,7+40
    return state_inputs


def get_next_state(state, couriers):
    state = np.array(state).flatten()
    state_couriers = np.zeros((len(couriers), 100 * 3 + 4))  # env.n_valid_nodes = 100
    state_couriers[:, :100 * 3] = np.stack([state] * len(couriers))  # state：n_valid_nodes*3=10*10*3=300
    state_couriers[:, 100 * 3:100 * 3 + 4] = couriers
    return state_couriers


def process_memory(next_state, dispatch_result):
    states = []
    state_inputs = []
    actions = []
    rewards = []
    next_states = []
    costs = []
    # policy = []
    for order, d_pair in dispatch_result.items():
        state_inputs.append(d_pair.state_input)
        states.append(d_pair.state)
        actions.append(int(d_pair.action))
        rewards.append(d_pair.reward)
        costs.append(int(d_pair.cost))
        # policy.append(d_pair.policy)
        next_state = np.array(next_state.flatten())  # 更新后的路网特征
        next_state1 = np.hstack((next_state, d_pair.next_state))  # 骑手特征+路网特征:critic网络输入
        # next_state1 = get_next_state(next_state, next_courier_state)
        next_states.append(next_state1)
    return state_inputs, states, actions, rewards, next_states, costs  # , np.array(policy


class DispatchPair:
    def __init__(self, order, courier):
        self.index = 0  # int
        self.order = order
        self.courier = courier
        self.state_input = None
        self.state = None
        self.next_state = None
        self.action = None
        self.reward = 0  # float
        self.cost = None
        # self.policy = None
        self.next_action = None

    def set_index(self, index):
        self.index = index

    def set_state_input(self, state):
        self.state_input = state

    def set_state(self, state):
        self.state = state

    def set_action(self, action):
        self.action = np.array(action)

    def set_next_state(self, next_state):
        self.next_state = next_state

    def set_next_action(self, next_action):
        self.next_action = np.array(next_action)

    def set_cost(self, cost):
        if cost <= 0:
            self.cost = 0
        else:
            self.cost = 1

    def get_reward(self):
        return self.reward

    def set_reward(self, env, alpha, beta, i_episode, flag):
        Loss = torch.nn.MSELoss()
        delivery_efficiency = []
        shop_radio_sum = []
        courier_efficiency = 0.0
        resultList = []

        for courier_id, courier in env.couriers_dict.items():
            if courier.online is True and courier.occur_time <= env.time_slot_index:
                delivery_fee = 0.0
                if len(courier.order_list):
                    for order in courier.order_list:
                        if env.shops_dict[order.shop_loc] not in env.involved_shop:
                            env.involved_shop.append(env.shops_dict[order.shop_loc])
                        delivery_fee += order.price  # 订单效益
                        if flag:
                            resultList.append([order.order_id, courier_id, env.shops_dict[order.shop_loc].shop_id,
                                               order.shop_loc.longitude, order.shop_loc.latitude,
                                               order.user_loc.longitude, order.user_loc.latitude,
                                               order.promise_delivery_duration, order.price,
                                               order.order_create_time, order.order_accept_time,
                                               order.order_pickup_time, order.order_delivery_time,
                                               order.real_delivery_duration])
                if delivery_fee != 0.0:
                    total_fee = env.couriers_dict[courier_id].sum_fee + delivery_fee
                    work_time = env.couriers_dict[courier_id].work_day + int(
                        env.time_slot_index - courier.occur_time + 1)
                    delivery_efficiency.append(total_fee / work_time)
                else:
                    delivery_efficiency.append(0)
                if self.courier is not None and courier_id == self.courier.courier_id:
                    courier_efficiency = delivery_efficiency[-1]

        efficiency = torch.from_numpy(np.array(delivery_efficiency))
        mean_efficiency = torch.tensor([np.mean(np.array(delivery_efficiency))] * efficiency.shape[0])
        courier_fairness = Loss(efficiency, mean_efficiency).item()

        for shop in env.involved_shop:
            shopRadioMean = 0
            for r in shop.order_time_distance_radio:
                if r:
                    shopRadioMean += r[0]
            # if shopRadioMean != 0:
            shop_radio_sum.append(shopRadioMean / shop.order_num)
        if len(shop_radio_sum) == 0:
            shop_fairness = 0
        else:
            fairness = torch.from_numpy(np.array(shop_radio_sum))
            mean_fairness = torch.tensor([np.mean(np.array(shop_radio_sum))] * fairness.shape[0])
            shop_fairness = Loss(fairness, mean_fairness).item()

        if flag:
            df = pd.DataFrame(resultList,
                              columns=['order_id', 'rider_id', 'shop_id', 'shop_longitude', 'shop_latitude',
                                       'user_longitude', 'user_latitude', 'promise_delivery_time', 'delivery_fee',
                                       'order_create_time', 'rider_accept_order_time',
                                       'rider_arrive_restaurant_time', 'rider_delivery_time', 'real_delivery_duration'])
            fileName = 'save-dir/result/episode' + str(i_episode) + '.csv'
            df.to_csv(fileName)  # 追加数据

        # three parts: order price + courier fairness + shop fairness
        self.reward = (1 - alpha - beta) * courier_efficiency - alpha * courier_fairness - beta * shop_fairness

    def set_penalty_reward(self, env, alpha, beta, rate, i_episode, flag):
        Loss = torch.nn.MSELoss()
        delivery_efficiency = []
        shop_radio_sum = []
        courier_efficiency = 0.0
        resultList = []

        for courier_id, courier in env.couriers_dict.items():
            if courier.online is True and courier.occur_time <= env.time_slot_index:
                delivery_fee = 0.0
                if len(courier.order_list):
                    for order in courier.order_list:
                        if env.shops_dict[order.shop_loc] not in env.involved_shop:
                            env.involved_shop.append(env.shops_dict[order.shop_loc])
                        delivery_fee += order.price  # 订单效益
                        if flag:
                            resultList.append([order.order_id, courier_id, env.shops_dict[order.shop_loc].shop_id,
                                               order.shop_loc.longitude, order.shop_loc.latitude,
                                               order.user_loc.longitude, order.user_loc.latitude,
                                               order.promise_delivery_duration, order.price,
                                               order.order_create_time, order.order_accept_time,
                                               order.order_pickup_time, order.order_delivery_time,
                                               order.real_delivery_duration])
                if delivery_fee != 0.0:
                    total_fee = env.couriers_dict[courier_id].sum_fee + delivery_fee
                    work_time = env.couriers_dict[courier_id].work_day + int(
                        env.time_slot_index - courier.occur_time + 1)
                    delivery_efficiency.append(total_fee / work_time)
                else:
                    delivery_efficiency.append(0)
                if self.courier is not None and courier_id == self.courier.courier_id:
                    courier_efficiency = delivery_efficiency[-1]

        efficiency = torch.from_numpy(np.array(delivery_efficiency))
        mean_efficiency = torch.tensor([np.mean(np.array(delivery_efficiency))] * efficiency.shape[0])
        courier_fairness = Loss(efficiency, mean_efficiency).item()

        for shop in env.involved_shop:
            shopRadioMean = 0
            for r in shop.order_time_distance_radio:
                if r:
                    shopRadioMean += r[0]
            # if shopRadioMean != 0:
            shop_radio_sum.append(shopRadioMean / shop.order_num)
        if len(shop_radio_sum) == 0:
            shop_fairness = 0
        else:
            fairness = torch.from_numpy(np.array(shop_radio_sum))
            mean_fairness = torch.tensor([np.mean(np.array(shop_radio_sum))] * fairness.shape[0])
            shop_fairness = Loss(fairness, mean_fairness).item()

        if flag:
            df = pd.DataFrame(resultList,
                              columns=['order_id', 'rider_id', 'shop_id', 'shop_longitude', 'shop_latitude',
                                       'user_longitude', 'user_latitude', 'promise_delivery_time', 'delivery_fee',
                                       'order_create_time', 'rider_accept_order_time',
                                       'rider_arrive_restaurant_time', 'rider_delivery_time', 'real_delivery_duration'])
            fileName = '../save-dir/result/episode' + str(i_episode) + '.csv'
            df.to_csv(fileName)  # 追加数据

        # three parts: order price + courier fairness + shop fairness
        self.reward = (1 - alpha - beta) * courier_efficiency \
                      - alpha * courier_fairness - beta * shop_fairness - rate * self.cost
