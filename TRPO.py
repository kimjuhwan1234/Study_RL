import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_v = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.pi = nn.Linear(32, action_dim)

    def forward(self, x, softmax_dim=-1):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        logits = self.pi(x)
        probs = F.softmax(logits, dim=softmax_dim)
        return probs


class TRPO(nn.Module):
    def __init__(self, state_dim, action_dim, value_lr=0.003, gamma=0.99, lmbda=0.95, num_value_updates=3, delta=0.01,
                 damping=0.1, cg_iters=10):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.v = Critic(state_dim).to(self.device)
        self.pi = Actor(state_dim, action_dim).to(self.device)
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=value_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.K = num_value_updates
        self.delta = delta
        self.damping = damping
        self.cg_iters = cg_iters

        self.data = []
        self.alpha = 0.5
        self.max_kl = 0.05

    def select_action(self, state, rewards_log=None):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.pi(state)
        dist = torch.distributions.Categorical(probs)

        if rewards_log is not None and len(rewards_log) >= 10 and np.mean(rewards_log[-10:]) >= 450:
            action = torch.argmax(probs)
        else:
            action = dist.sample()

        action_prob = probs[0, action].item()
        return action.item(), action_prob

    def put_data(self, item):
        self.data.append(item)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for item in self.data:
            s, a, r, s_prime, prob_a, done = item
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_lst.append([0 if done else 1])

        s = torch.tensor(s_lst, dtype=torch.float).to(self.device)
        a = torch.tensor(a_lst).to(self.device)
        r = torch.tensor(r_lst).to(self.device)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        prob_a = torch.tensor(prob_a_lst).to(self.device)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def fisher_vector_product(self, vector, state):
        """FIM(Fisher Information Matrix)와 벡터 곱"""
        pi_new = self.pi(state)
        dist_new = torch.distributions.Categorical(probs=pi_new)
        dist_old = torch.distributions.Categorical(probs=self.prob_a.detach())
        kl = torch.distributions.kl_divergence(dist_old, dist_new).mean()

        # 1️⃣ KL-divergence의 Gradient 계산
        grads = torch.autograd.grad(kl, self.pi.parameters(), create_graph=True, retain_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl * vector).sum()

        # 2️⃣ Fisher Information Matrix와 벡터 곱 (2차 미분)
        grads = torch.autograd.grad(kl_v, self.pi.parameters(), retain_graph=True)
        flat_grad_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads]).detach()

        return flat_grad_grad_kl + self.damping * vector  # Damping term 추가 (안정성 향상)

    def conjugate_gradient(self, policy_grad, state, residual_tol=1e-10):
        """Conjugate Gradient (CG)로 F^-1 g 계산"""
        x = torch.zeros_like(policy_grad)
        r = policy_grad.clone()
        p = r.clone()
        r_dot_r = torch.dot(r, r)

        for _ in range(self.cg_iters):
            fisher_p = self.fisher_vector_product(p, state)
            alpha = r_dot_r / (torch.dot(p, fisher_p) + 1e-8)

            x += alpha * p
            r -= alpha * fisher_p
            new_r_dot_r = torch.dot(r, r)

            beta = new_r_dot_r / (r_dot_r + 1e-8)

            p = r + beta * p
            r_dot_r = new_r_dot_r

            if r_dot_r < residual_tol:
                break

        return x

    def line_search(self, new_params, states):
        # Get new policy distribution
        with torch.no_grad():
            pi_old = self.pi(states).detach()

        # Temporarily set new parameters
        offset = 0
        old_params_backup = {i: param.data.clone() for i, param in enumerate(self.pi.parameters())}

        for i, param in enumerate(self.pi.parameters()):
            old_params_backup[i] = param.data.clone()
            numel = param.numel()
            param.data.copy_(new_params[offset:offset + numel].view(param.shape))
            offset += numel

        pi_new = self.pi(states)

        dist_new = torch.distributions.Categorical(probs=pi_new)
        dist_old = torch.distributions.Categorical(probs=pi_old.detach())
        kl = torch.distributions.kl_divergence(dist_new, dist_old).mean()

        # Restore original parameters
        for i, param in enumerate(self.pi.parameters()):
            param.data.copy_(old_params_backup[i])

        return kl

    def update(self, policy_grad, state):
        # 2️⃣ Fisher Information Matrix의 역행렬 곱 계산
        step_direction = self.conjugate_gradient(policy_grad, state)
        old_gHg = torch.dot(policy_grad, step_direction)
        gHg = torch.max(old_gHg, torch.tensor(0.001, device=old_gHg.device))

        # 3️⃣ Step Size 조정 (Line Search)
        step_size = torch.sqrt(2 * self.delta / gHg)
        update_vector = step_size * step_direction

        # 옵티마이저에서 기울기 초기화
        for param in self.pi.parameters():
            param.grad = None  # 기울기 초기화 (zero_grad 효과)

        with torch.no_grad():
            param_vector = torch.cat([p.view(-1) for p in self.pi.parameters()])
            new_param_vector = param_vector + self.alpha * update_vector

            # KL 제약 만족할 때까지 beta 감소
            while self.line_search(new_param_vector, state) > self.max_kl:
                print(old_gHg, gHg)
                self.alpha *= self.alpha
                new_param_vector = param_vector + self.alpha * update_vector

            # 새로운 파라미터 적용
            offset = 0
            for param in self.pi.parameters():
                numel = param.numel()
                param.data.copy_(new_param_vector[offset:offset + numel].view(param.shape))
                offset += numel

    def train(self):
        s, a, r, s_prime, done_mask, self.prob_a = self.make_batch()

        for _ in range(self.K):  # K번 학습 반복
            td_target = r + self.gamma * self.v(s_prime) * done_mask  # TD 타겟 계산
            delta = td_target - self.v(s)  # TD 오차 계산
            delta = delta.detach().cpu().numpy()  # NumPy 변환 (GAE 계산 용이하게)

            # Generalized Advantage Estimation (GAE) 계산
            advantage_lst = []
            advantage = 0
            for delta_t in delta[::-1]:  # 역순으로 계산
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()

            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)


            # 정책 업데이트를 위한 손실 계산
            pi = self.pi(s, softmax_dim=-1)  # 현재 정책 확률
            pi_a = pi.gather(1, a)  # 실행한 액션의 확률
            ratio = torch.exp(torch.log(pi_a) - torch.log(self.prob_a))  # 확률 비율 계산

            # PPO 클리핑 적용
            policy_loss = - ratio * advantage
            value_loss = F.smooth_l1_loss(td_target.detach(), self.v(s))

            # 가치 함수 손실 계산 (MSE)
            self.v_optimizer.zero_grad()
            value_loss.backward()
            self.v_optimizer.step()

        grads = torch.autograd.grad(policy_loss.mean(), self.pi.parameters())
        grad_vector = torch.cat([g.view(-1) for g in grads]).detach()
        self.update(grad_vector, s)
