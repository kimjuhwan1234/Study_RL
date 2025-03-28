import gymnasium as gym
import torch
import warnings
import numpy as np
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

warnings.filterwarnings("ignore")
mp.set_start_method("spawn", force=True)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.tensor([0], dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # ✅ Shared Memory 설정
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_pi = nn.Linear(32, action_dim)  # Policy network

    def forward(self, x, softmax_dim=-1):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_v = nn.Linear(32, 1)  # Value network

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.fc_v(x)
        return value


class A3CWorker:
    def __init__(self, env, global_actor, global_critic, actor_optimizer, critic_optimizer, state_dim, action_dim,
                 gamma=0.99, K=3):
        self.env = env
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.local_actor = Actor(state_dim, action_dim)
        self.local_actor.load_state_dict(self.global_actor.state_dict())

        self.local_critic = Critic(state_dim)
        self.local_critic.load_state_dict(self.global_critic.state_dict())
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.K = K
        self.data = []
        self.reward_log = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.local_actor(state_tensor)

        # 기존 방식대로 확률적으로 샘플링
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.item(), log_prob  # 액션과 log 확률 반환

    def put_data(self, item):
        self.data.append(item)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for s, a, r, s_prime, done in self.data:
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([0 if done else 1])

        s = torch.tensor(np.array(s_lst), dtype=torch.float)
        a = torch.tensor(a_lst)
        r = torch.tensor(r_lst)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
        done_mask = torch.tensor(done_lst, dtype=torch.float)

        self.data = []
        return s, a, r, s_prime, done_mask

    def train(self, n_epi):
        s, a, r, s_prime, done_mask = self.make_batch()

        if n_epi % 200 == 0:
            values = self.local_critic(s).squeeze()

            # ✅ TD-Target (Discounted Return) 계산
            G = torch.zeros_like(r, dtype=torch.float)
            G[-1] = r[-1] + self.gamma * self.local_critic(s_prime[-1]) * done_mask[-1]

            for i in range(len(r) - 2, -1, -1):
                G[i] = r[i] + self.gamma * G[i + 1]

            returns = G.unsqueeze(1)

            # ✅ Advantage 계산 및 정규화
            advantage = returns - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            pi = self.local_actor(s)
            pi_a = pi.gather(1, a)
            pi_a = torch.clamp(pi_a, min=1e-8)  # ✅ log(0) 방지

            actor_loss = (-torch.log(pi_a) * advantage).mean()
            critic_loss = F.mse_loss(values, returns)


            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            for global_param, local_param in zip(self.global_actor.parameters(), self.local_actor.parameters()):
                if local_param.grad is not None:
                    global_param._grad = local_param.grad.clone()
            self.actor_optimizer.step()


            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            for global_param, local_param in zip(self.global_critic.parameters(), self.local_critic.parameters()):
                if local_param.grad is not None:
                    global_param._grad = local_param.grad.clone()
            self.critic_optimizer.step()


            self.local_actor.load_state_dict(self.global_actor.state_dict())
            self.local_critic.load_state_dict(self.global_critic.state_dict())

    def run(self):
        for n_epi in tqdm(range(1000), desc=f"Worker {id(self)}", position=None, leave=True):
            state, _ = self.env.reset()
            total_reward = 0
            step_count = 0

            for _ in range(500):
                a, log_prob = self.select_action(state)
                s_prime, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated

                decreasing_reward = r * (0.99 ** step_count)
                step_count += 1

                self.put_data((state, a, decreasing_reward, s_prime, done))
                state = s_prime
                total_reward += r

                if done:
                    break

            if len(self.reward_log) >= 10 and np.mean(self.reward_log[-10:]) < 450:
                self.train(n_epi)

            self.reward_log.append(total_reward)

        self.env.close()


def train_worker(global_actor, global_critic, actor_optimizer, critic_optimizer, state_dim, action_dim):
    env = gym.make("CartPole-v1")
    worker = A3CWorker(env, global_actor, global_critic, actor_optimizer, critic_optimizer, state_dim, action_dim)
    worker.run()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')


def plot_weight_distribution(model):
    plt.figure(figsize=(12, 4), dpi=400)

    for name, param in model.named_parameters():
        if "weight" in name:
            sns.histplot(param.data.cpu().numpy().flatten(), bins=100, kde=True, label=name)

    plt.title("Weight Distribution of DQN")
    plt.xlabel("Weight Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    state_dim = 4
    action_dim = 2

    global_actor = Actor(state_dim, action_dim)
    global_actor.apply(initialize_weights)
    global_actor.share_memory()
    actor_optimizer = SharedAdam(global_actor.parameters(), lr=5e-4)

    global_critic = Critic(state_dim)
    global_critic.apply(initialize_weights)
    global_critic.share_memory()
    critic_optimizer = SharedAdam(global_critic.parameters(), lr=1e-4)

    workers = []
    for _ in range(5):
        worker = mp.Process(target=train_worker, args=(
            global_actor, global_critic, actor_optimizer, critic_optimizer, state_dim, action_dim))
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()

    env = gym.make("CartPole-v1")
    rewards_log = []

    # ✅ 성능 평가 (확률 높은 액션만 수행)
    for _ in tqdm(range(1000)):
        s, _ = env.reset()
        total_reward = 0

        for _ in range(500):
            state_tensor = torch.FloatTensor(s).unsqueeze(0)

            # ✅ NaN 방지 및 확률 값 확인
            probs = torch.clamp(global_actor(state_tensor), min=1e-8)

            action = torch.argmax(probs).item()

            # ✅ 액션 수행
            s_prime, r, terminated, truncated, _ = env.step(action)
            s = s_prime
            done = terminated or truncated
            total_reward += r

            if done:
                break

        rewards_log.append(total_reward)

    env.close()

    # ✅ 학습된 모델의 가중치 분포 시각화
    plot_weight_distribution(global_actor.fc2)
    plot_weight_distribution(global_critic.fc2)

    # ✅ 보상 그래프 출력
    plt.figure(figsize=(15, 3), dpi=400)
    plt.plot(rewards_log, label="Global Network")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole-v1 Training Rewards")
    plt.legend()
    plt.show()

    # ✅ 학습 결과 요약 출력
    print("First 10 episode rewards:", rewards_log[:10])
    print("Last 10 episode rewards:", rewards_log[-10:])
    print(f"Average reward over last 100 episodes: {np.mean(rewards_log[-100:]):.2f}")
