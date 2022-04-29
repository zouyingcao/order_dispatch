import torch
import numpy as np
import torch.nn.functional as F

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")


# actor
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim_, action_dim_):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim_
        self.action_dim = action_dim_
        self.S = torch.nn.Linear(state_dim_, 128).to(device)
        self.A = torch.nn.Linear(action_dim_, 8).to(device)
        self.l1 = torch.nn.Linear(128 + 8, 16).to(device)
        self.l2 = torch.nn.Linear(16, 8).to(device)
        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X):
        s1 = X[:, :self.state_dim]
        a1 = X[:, -self.action_dim:]
        S1 = F.relu(self.S(s1))
        A1 = F.relu(self.A(a1))
        Y1 = torch.cat((S1, A1), dim=1)
        l1 = F.relu(self.l1(Y1))
        l2 = F.relu(self.l2(l1))
        return F.relu(self.f(l2)) + 1


# critic
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim_):
        super(ValueNet, self).__init__()
        self.S = torch.nn.Linear(state_dim_, 128).to(device)
        self.l1 = torch.nn.Linear(128, 64).to(device)
        self.l2 = torch.nn.Linear(64, 32).to(device)
        self.f = torch.nn.Linear(32, 1).to(device)

    def forward(self, X):
        S1 = F.relu(self.S(X))
        l1 = F.relu(self.l1(S1))
        l2 = F.relu(self.l2(l1))
        return F.relu(self.f(l2))


class ActorCritic:
    def __init__(self, state_dim, action_dim_, actor_lr_, critic_lr_, gamma_, device_):
        self.actor = PolicyNet(state_dim, action_dim_).to(device_)
        self.critic = ValueNet(state_dim).to(device_)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr_)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr_)
        self.gamma = gamma_

    # actor:采取动作
    def take_action(self, _state):
        state_ = torch.tensor(_state, dtype=torch.float).to(device)
        v_output = self.actor(state_)
        action_prob = torch.softmax(v_output, dim=0)
        action_dist = torch.distributions.Categorical(action_prob)
        c_id_ = np.argmax(action_dist.sample().cpu())
        # c_id_ = action_dist.sample()
        # print(c_id_)
        return c_id_.item(), action_prob.detach()

    def take_action_epsilon(self, _state, epsilon):
        state_ = torch.tensor(_state, dtype=torch.float).to(device)
        v_output = self.actor(state_)
        softmax_V = torch.softmax(v_output, dim=0)
        c_id = np.argmax(np.array(v_output.cpu().detach().numpy()))
        # epsilon-greedy 策略
        action_prob = np.ones(len(_state))
        action_prob = action_prob * (1 - epsilon) / (len(_state))
        action_prob[c_id] += epsilon
        c_id_ = np.argmax(np.random.multinomial(1, action_prob))
        return c_id_.item(), softmax_V.detach()

    def update(self, states, rewards, next_states, policy):
        states = torch.tensor(states, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        policy = torch.tensor(policy, dtype=torch.float, requires_grad=True).to(device)  # 对应的softmax_V_output

        td_target = torch.unsqueeze(rewards, 1) + self.gamma * self.critic(next_states).to(device)
        V = self.critic(states)
        critic_loss = torch.mean(F.mse_loss(V, td_target.detach())).to(device)  # TD error
        td_delta = td_target - V  # 时序差分误差 TD_error
        log_prob = torch.log(policy)  # softmax后对应的选的action那一值组成的tensor, requires_grad=True
        actor_loss = torch.mean(-log_prob * td_delta.detach()).to(device)

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        critic_loss.backward()  # 计算critic网络的梯度
        actor_loss.backward()  # 计算actor网络的梯度
        self.critic_optimizer.step()  # 更新critic网络参数
        self.actor_optimizer.step()  # 更新actor网络参数

    def actor_update(self, states, rewards, next_states, policy):
        states = torch.tensor(states, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)  # TD error
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        policy = torch.tensor(policy, dtype=torch.float, requires_grad=True).to(device)

        # log_softmax = torch.nn.LogSoftmax()
        # log_softmax_prob = log_softmax(softmax_log_prob)
        # loss = torch.mean(torch.sum(log_softmax_prob * advantages, dim=1))
        # entropy = - torch.mean(softmax_log_prob * log_softmax_prob)
        # actor_loss = loss - 0.01 * entropy
        td_target = torch.unsqueeze(rewards, 1) + self.gamma * self.critic(next_states).to(device)
        td_delta = td_target - self.critic(states)  # 时序差分误差 TD_error
        log_prob = torch.log(policy)
        actor_loss = torch.mean(-log_prob * td_delta.detach()).to(device)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # 计算actor网络的梯度
        self.actor_optimizer.step()  # 更新actor网络参数

    def critic_update(self, states, rewards, next_states):
        states = torch.tensor(states, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)  # TD error:Torch.size([200])
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)

        td_target = torch.unsqueeze(rewards, 1) + self.gamma * self.critic(next_states).to(device)

        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach())).to(device)  # TD error

        self.critic_optimizer.zero_grad()
        critic_loss.backward()  # 计算critic网络的梯度
        self.critic_optimizer.step()  # 更新critic网络参数
