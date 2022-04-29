import torch
import numpy as np
import pandas as pd

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")


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
        self.next_action = None
        self.policy = None

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

    def set_policy(self, policy):
        self.policy = policy

    def get_reward(self):
        return self.reward

    def set_reward(self, i_episode, env, gamma, alpha, beta):
        Loss = torch.nn.MSELoss()
        involved_shop = []
        delivery_efficiency = []
        shop_radio_sum = []
        courier_efficiency = gamma ** self.order.real_delivery_duration * self.order.price
        resultList = []

        for courier_id, courier in env.couriers_dict.items():
            if courier.online is True and courier.occur_time <= env.time_slot_index:
                delivery_fee = []
                for i in range(courier.work_day_index + 1):
                    if len(courier.order_list[i]):
                        for order in courier.order_list[i]:
                            # if order == 0:
                            #     continue
                            if env.shops_dict[order.shop_loc] not in involved_shop:
                                involved_shop.append(env.shops_dict[order.shop_loc])
                            r_price = gamma ** order.real_delivery_duration * order.price  # 订单效益
                            delivery_fee.append(r_price)
                            resultList.append([order.order_id, courier_id, env.shops_dict[order.shop_loc].shop_id,
                                               order.shop_loc.longitude, order.shop_loc.latitude,
                                               order.user_loc.longitude, order.user_loc.latitude,
                                               order.promise_delivery_duration, order.price, r_price,
                                               order.order_create_time, order.order_accept_time,
                                               order.order_pickup_time, order.order_delivery_time,
                                               order.real_delivery_duration])
                # if courier.courier_id == c_id:
                #     courier_efficiency = sum(delivery_fee)
                if len(delivery_fee) != 0:
                    work_time = 0
                    for t in courier.work_day:  # 时间步为单位
                        work_time += t
                    delivery_efficiency.append(sum(delivery_fee) /
                                               (work_time + int(env.time_slot_index - courier.occur_time + 1)))
                else:
                    delivery_efficiency.append(0)

        efficiency = torch.from_numpy(np.array(delivery_efficiency)).to(device)
        mean_efficiency = torch.tensor([np.mean(np.array(delivery_efficiency))] * efficiency.shape[0]).to(device)
        courier_fairness = Loss(efficiency, mean_efficiency).item()

        for shop in involved_shop:
            shopRadioMean = 0
            num = 0
            for r in shop.order_time_distance_radio:
                if r:
                    for rr in r:
                        num += 1
                        shopRadioMean += rr
            if shopRadioMean != 0:
                shop_radio_sum.append(shopRadioMean / num)
        if len(shop_radio_sum) == 0:
            shop_fairness = 0
        else:
            fairness = torch.from_numpy(np.array(shop_radio_sum)).to(device)
            mean_fairness = torch.tensor([np.mean(np.array(shop_radio_sum))] * fairness.shape[0]).to(device)
            shop_fairness = Loss(fairness, mean_fairness).item()

        df = pd.DataFrame(resultList,
                          columns=['order_id', 'rider_id', 'shop_id', 'shop_longitude', 'shop_latitude',
                                   'user_longitude', 'user_latitude', 'promise_delivery_time', 'delivery fee',
                                   'discounted fee', 'order_create_time', 'rider_accept_order_time',
                                   'rider_arrive_restaurant_time', 'rider_delivery_time', 'real_delivery_duration'])
        fileName = '../rewards/result'+str(i_episode)+'.csv'
        df.to_csv(fileName)  # 追加数据
        self.reward = (1 - alpha - beta) * courier_efficiency + alpha * (-courier_fairness) + beta * (-shop_fairness)
