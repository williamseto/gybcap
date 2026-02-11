import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ====== Toy 2-State MDP without hard violation limit ======
class TwoStateEnv(gym.Env):
    """
    State 0: A (can trade with small reward + violation)
    State 1: B (hold-only, zero reward)
    Actions: 0=hold, 1=trade
    No hard termination on violations; agent continues indefinitely
    """
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.state = 0  # always start in A
        self.violation_count = 0
        return self._obs()

    def _obs(self):
        # One-hot encoding of state
        return np.array([1.0, 0.0]) if self.state == 0 else np.array([0.0, 1.0])

    def step(self, action):
        # Compute reward and violation
        reward = 1.0 if (self.state == 0 and action == 1) else 0.0
        violation = 1.0 if (self.state == 0 and action == 1) else 0.0
        self.violation_count += violation

        # Transition: A -> B -> A ...
        self.state = 1 - self.state

        done = False  # no hard limit
        info = {'violation': violation}
        return self._obs(), reward, done, info

# ====== Simple Actor-Critic ======
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=2, hidden_dim=16, act_dim=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh()
        )
        self.pi = nn.Linear(hidden_dim, act_dim)
        self.v  = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return nn.Softmax(dim=-1)(self.pi(x)), self.v(x)

# ====== PPO + Lagrangian ======
class PPOLagAgent:
    def __init__(self, env, lr=1e-2, gamma=0.99, clip_eps=0.2, lagr_lr=1e-2, critic_coeff=0.5):
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.critic_coeff = critic_coeff

        self.model = ActorCritic()
        # include both actor and critic in optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # Lagrange multiplier
        self.lagr = torch.tensor(1.0, requires_grad=True)
        self.lagr_opt = torch.optim.Adam([self.lagr], lr=lagr_lr)

    def collect(self, steps=200):
        obs_list, act_list, rew_list, logp_list, vio_list = [], [], [], [], []
        obs = self.env.reset()
        for _ in range(steps):
            o = torch.FloatTensor(obs)
            pi, _ = self.model(o)
            dist = Categorical(pi)
            a = dist.sample().item()
            next_obs, r, done, info = self.env.step(a)

            obs_list.append(o)
            act_list.append(a)
            rew_list.append(r)
            logp_list.append(dist.log_prob(torch.tensor(a)).detach())
            vio_list.append(info['violation'])

            obs = next_obs

        print(f"np.sum(vio_list): {np.sum(vio_list)}")
        return obs_list, act_list, rew_list, logp_list, vio_list

    def update(self, data):
        obs_list, act_list, rew_list, logp_list, vio_list = data
        # Compute discounted returns
        R, G = [], 0
        for r in reversed(rew_list):
            G = r + self.gamma * G
            R.insert(0, G)
        returns = torch.tensor(R)

        # Prepare tensors
        obs = torch.stack(obs_list)
        acts = torch.tensor(act_list)

        old_logp = torch.stack(logp_list)
        vio = torch.tensor(vio_list)

        # Forward pass for current policy
        pi, values = self.model(obs)
        dist = Categorical(pi)
        logp = dist.log_prob(acts)
        values = values.squeeze(1)

        # Advantages (using value baseline)
        advantages = returns - values.detach()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO surrogate loss
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss (MSE)
        critic_loss = 0.5 * (returns - values).pow(2).mean()

        w = next(self.model.pi.parameters())
        w_old = w.data.clone()

        self.optimizer.zero_grad()
        self.lagr_opt.zero_grad()
        
        logp.retain_grad()

        # Lagrangian penalty on average violation
        # penalty = self.lagr * vio.mean()
        penalty = (self.lagr * (vio * logp)).mean()

        # Total loss
        # total_loss = actor_loss + self.critic_coeff * critic_loss + penalty
        total_loss = penalty

        # Backward pass and updates
        total_loss.backward(retain_graph=True)


        print(f"penalty {penalty.item()} d logp_sample =", logp.grad)
        exit()

        # After loss.backward(), before opt.step()
        # actor_norm = 0.0
        # critic_norm = 0.0
        # for name, p in self.model.named_parameters():
        #     if p.grad is None:
        #         print(f"{name}: NO GRAD")
        #         continue
        #     grad_norm = p.grad.data.norm().item()
        #     if name.startswith("pi") or name.startswith("fc"):   # adjust to your actor layers
        #         actor_norm   += grad_norm
        #     elif name.startswith("v"):                             # your critic layer
        #         critic_norm  += grad_norm
        #     else:
        #         # If you have shared layers, you can bucket them separately
        #         print(f"{name}: grad norm = {grad_norm:.3e} (shared?)")
        # print(f"Gradient Norms → Actor: {actor_norm:.3e}, Critic: {critic_norm:.3e}")

        self.optimizer.step()


        # Dual ascent on lambda
        # (-penalty).backward()

        vio_mean = vio.mean()                 # a constant
        lagr_term = self.lagr * vio_mean      # new scalar
        dual_loss = -lagr_term                # maximize over λ
        dual_loss.backward()                  # gradient only flows into self.lagr

        self.lagr_opt.step()
        with torch.no_grad():
            self.lagr.clamp_(0)

        mask_A = (obs[:,0] == 1.0)
        p_trade_A = dist.probs[mask_A, 1].mean().item()

        # Logging
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'penalty': penalty.item(),
            'lambda': self.lagr.item(),
            'p_trade_A': p_trade_A
        }

# ====== Training Loop ======
if __name__ == '__main__':
    env = TwoStateEnv()
    agent = PPOLagAgent(env)
    for epoch in range(1, 51):
        data = agent.collect(steps=200)
        logs = agent.update(data)
        print(f"Epoch {epoch:02d} | ActorLoss={logs['actor_loss']:.3f} | CriticLoss={logs['critic_loss']:.3f} | "
              f"Penalty={logs['penalty']:.3f} | λ={logs['lambda']:.3f} | P(trade|A)={logs['p_trade_A']:.3f}")
