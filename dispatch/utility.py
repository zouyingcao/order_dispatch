import numpy as np


def get_courier_state(env, state, couriers):
    state = np.array(state).flatten()
    state_couriers = np.zeros((len(couriers), env.n_valid_nodes * 3 + 4))
    # print(len(couriers))
    state_couriers[:, :env.n_valid_nodes * 3] = np.stack([state] * len(couriers))  # state：n_valid_nodes*3 = 10*10*3=300
    in_states = []
    id_list = []
    for ic in couriers:
        lat = ic.latitude
        long = ic.longitude
        t = ic.record_time_slot - ic.occur_time
        num = ic.num_order
        in_state = [lat, long, num, t]  # 可以直接in_state = ic.get_state()
        in_states.append(in_state)
        id_list.append(ic.courier_id)
    in_states = np.array(in_states)
    state_couriers[:, env.n_valid_nodes * 3:env.n_valid_nodes * 3 + 4] = in_states[:, 0:4]
    return state_couriers, id_list


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


def process_memory(next_state, dispatch_result):
    states = []
    actions = []
    rewards = []
    next_states = []
    policy = []
    for order, d_pair in dispatch_result.items():
        states.append(d_pair.state)
        actions.append(d_pair.action)
        rewards.append(d_pair.reward)
        policy.append(d_pair.policy)
        next_state = np.array(next_state.flatten())   # 更新后的路网特征
        next_state1 = np.hstack((next_state, d_pair.next_state))   # 骑手特征+路网特征:critic网络输入
        next_states.append(next_state1)
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(policy)


