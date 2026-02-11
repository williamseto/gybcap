import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque
import random
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1) Execution environment
class ExecutionEnv(gym.Env):
    def __init__(self, price_series=None, sigma_noise=0.02,
                 lambda_risk=0.1, max_steps=1000):
        super().__init__()
        self.max_steps = max_steps
        self.lambda_risk = lambda_risk
        self.sigma_noise = sigma_noise
        if price_series is None:
            self.price_series = self._generate_price_series()
        else:
            self.price_series = price_series
        self.n_steps = len(self.price_series) - 1
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        # obs: [price, position, drawdown]
        self.observation_space = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.reset()

    def _generate_price_series(self, mu=0.0, sigma=0.01,
                               S0=100, T=1.0, dt=0.001):
        steps = int(T / dt)
        prices = [S0]
        for _ in range(steps):
            ret = np.random.normal(mu * dt, sigma * np.sqrt(dt))
            prices.append(prices[-1] * np.exp(ret))
        return np.array(prices)

    def reset(self):
        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = 1.0
        self.peak = 1.0
        self.drawdown = 0.0
        self.price = self.price_series[0]
        return self._get_obs()

    def _get_obs(self):
        future_idx = min(self.current_step + 1, len(self.price_series)-1)
        trueH = self.price_series[future_idx] - self.price_series[self.current_step] + np.random.normal(0, 0.001)
        return np.array([trueH, self.position, self.drawdown], dtype=np.float32)

    def step(self, action):
        action = float(np.clip(action, -1, 1))
        prev_price = self.price_series[self.current_step]
        new_price = self.price_series[self.current_step + 1]
        ret = np.log(new_price / prev_price)
        pnl = action * ret
        self.portfolio_value *= np.exp(pnl)
        self.peak = max(self.peak, self.portfolio_value)
        self.drawdown = (self.peak - self.portfolio_value) / self.peak

        risk_penalty = self.lambda_risk * (action**2) * abs(ret)
        reward = pnl - risk_penalty

        self.current_step += 1
        done = self.current_step >= min(self.n_steps, self.max_steps)
        self.price = new_price
        self.position = action
        info = {'ret': ret, 'reward': reward, 'action': action, 'portfolio_value': self.portfolio_value}
        return self._get_obs(), reward, done, info


# 2) Collect full episodes for sequence training
def collect_episodes(env, model, max_steps=1000, n_episodes=50):
    episodes = []
    for _ in range(n_episodes):
        obs = env.reset(); episode = []
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, done, info = env.step(action)
            episode.append((obs, action, reward, next_obs))
            obs = next_obs
            if done:
                break
        episodes.append(episode)
    return episodes

# 3) Latent-state RNN surrogate model
class LatentDynamics(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.enc = nn.Linear(obs_dim + act_dim + latent_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, latent_dim)
        self.dec_obs = nn.Linear(latent_dim, obs_dim)
        self.dec_rew = nn.Linear(latent_dim, 1)

    def forward(self, obs, act, h):
        x = torch.cat([obs, act, h], dim=-1)
        h_next = self.rnn(torch.relu(self.enc(x)), h)
        obs_pred = self.dec_obs(h_next)
        rew_pred = self.dec_rew(h_next)
        return obs_pred, rew_pred.squeeze(-1), h_next

# 4) Train dynamics on sequences of length seq_len
def train_dynamics(model, episodes, seq_len=10, epochs=50, batch_size=32, device='cpu'):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    model.to(device)
    for _ in range(epochs):
        # flatten possible start points
        starts = []
        for ep in episodes:
            L = len(ep)
            for i in range(L - seq_len):
                starts.append((ep, i))
        random.shuffle(starts)
        for idx in range(0, len(starts), batch_size):
            batch = starts[idx:idx+batch_size]
            # initialize latent
            h = torch.zeros(len(batch), model.rnn.hidden_size, device=device)
            loss = 0.0
            # iterate over sequence
            for t in range(seq_len):
                obs_b = torch.tensor([ep[i+t][0] for ep, i in batch], dtype=torch.float32, device=device)
                act_b = torch.tensor([ep[i+t][1] for ep, i in batch], dtype=torch.float32, device=device)
                next_b= torch.tensor([ep[i+t][3] for ep, i in batch], dtype=torch.float32, device=device)
                rew_b = torch.tensor([ep[i+t][2] for ep, i in batch], dtype=torch.float32, device=device)

                obs_pred, rew_pred, h = model(obs_b, act_b, h)
                loss += mse(obs_pred, next_b) + mse(rew_pred, rew_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

# 5) MPC planning with RNN over latent sequences
def mpc_with_rnn(model, cur_obs, plan_horizon=10, n_cand=256, device='cpu'):
    obs0 = torch.tensor(cur_obs, dtype=torch.float32, device=device)
    best_ret, best_a = -1e9, 0.0
    for _ in range(n_cand):
        h = torch.zeros(1, model.rnn.hidden_size, device=device)
        obs_t = obs0.clone()
        cum_rew = 0.0
        a0 = None
        for t in range(plan_horizon):
            a = torch.rand(1,1, device=device)*2 - 1
            if t == 0:
                a0 = a.item()
            
            obs_t, rew_t, h = model(obs_t, a, h)
            cum_rew += rew_t.item()
        if cum_rew > best_ret:
            best_ret, best_a = cum_rew, a0
    return np.array([best_a], dtype=np.float32)

# 5) CEM Planner using RNN surrogate

def cem_with_rnn(model, cur_obs, plan_horizon=10, n_cand=100, n_iter=5, elite_frac=0.2, device='cpu'):
    obs0 = torch.tensor(cur_obs, dtype=torch.float32, device=device)
    latent_dim = model.rnn.hidden_size
    # initialize mean/std for sequence of actions
    mean = torch.zeros(plan_horizon, 1, device=device)
    std = torch.ones(plan_horizon, 1, device=device)
    n_elite = int(n_cand * elite_frac)
    for _ in range(n_iter):
        # sample candidate sequences: [n_cand, H, 1]
        actions = mean + std * torch.randn(n_cand, plan_horizon, 1, device=device)
        actions = torch.clamp(actions, -1, 1)
        returns = torch.zeros(n_cand, device=device)
        for k in range(n_cand):
            h = torch.zeros(1, latent_dim, device=device)
            obs_t = obs0.clone()
            ret_sum = 0
            for t in range(plan_horizon):
                a = actions[k, t].unsqueeze(0)
                obs_t, rew_t, h = model(obs_t, a, h)
                ret_sum += rew_t.item()
            returns[k] = ret_sum
        # select elites
        topk = returns.topk(n_elite).indices
        elite_actions = actions[topk]  # [n_elite, H,1]
        # update mean/std
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0) + 1e-6
    # after iter, return first action of mean
    a0 = mean[0].cpu().numpy()
    return a0

# 6) Full flow
if __name__ == "__main__":
    # warm-start DDPG
    env = ExecutionEnv()
    noise = NormalActionNoise(mean=np.zeros(1), sigma=0.2*np.ones(1))

    vec_env = DummyVecEnv([lambda: ExecutionEnv()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    ddpg = DDPG('MlpPolicy', env, verbose=1)
    ddpg.learn(total_timesteps=50000)

    # # Post-training evaluation
    # obs = env.reset()
    # done = False
    # rewards, positions, portfolio_values = [], [], []
    # while not done:
    #     action, _ = ddpg.predict(obs, deterministic=True)
    #     print(f"Action: {action} obs: {obs}")
    #     obs, reward, done, info = env.step(action)
    #     positions.append(info['action'])
    #     rewards.append(reward)
    #     portfolio_values.append(info['portfolio_value'])

    # # Plot evaluation results
    # fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    # axes[0].plot(positions);
    # axes[0].set_title('Eval: Position over Time'); axes[0].set_ylabel('Position')
    # axes[1].plot(portfolio_values);
    # axes[1].set_title('Eval: Portfolio Value over Time'); axes[1].set_ylabel('Value')
    # axes[2].plot(env.price_series);
    # axes[2].set_title('prices'); axes[2].set_ylabel('price'); axes[2].set_xlabel('Time Step')
    # fig.tight_layout()
    # plt.show()


    # collect data
    data_buffer = collect_episodes(env, ddpg, max_steps=1000, n_episodes=30)
    # train RNN surrogate
    dyn = LatentDynamics(obs_dim=3, act_dim=1)
    train_dynamics(dyn, data_buffer)


    # # run planner
    # obs = env.reset(); done=False
    # positions, values = [], []
    # while not done:
    #     a = mpc_with_rnn(dyn, obs)
    #     obs, rew, done, info = env.step(a)
    #     positions.append(env.position)
    #     values.append(env.portfolio_value)
    # # plot
    # plt.subplot(2,1,1); plt.plot(positions); plt.title('Planned Positions')
    # plt.subplot(2,1,2); plt.plot(values); plt.title('Portfolio Value'); plt.tight_layout(); plt.show()

    prices = env._generate_price_series()


    env_baseline = DummyVecEnv([lambda: ExecutionEnv(prices)])
    env_baseline = VecNormalize(env_baseline, norm_obs=True, norm_reward=False)

    env_planner = DummyVecEnv([lambda: ExecutionEnv(prices)])
    env_planner = VecNormalize(env_planner, norm_obs=True, norm_reward=False)


    # run baseline
    obs = env_baseline.reset()
    vals_base = []
    done_base = False
    while not done_base:
        a, _ = ddpg.predict(obs, deterministic=True)
        obs, _, done_base, _ = env_baseline.step(a)
        vals_base.append(env_baseline.get_attr('portfolio_value'))

    # run planner
    obs = env_planner.reset()
    vals_mpc = []
    done_planner = False
    idx = 0
    while not done_planner:
        # a = mpc_with_rnn(dyn, obs)
        a = cem_with_rnn(dyn, obs)
        obs, _, done_planner, _ = env_planner.step(a)
        vals_mpc.append(env_planner.get_attr('portfolio_value'))
        idx += 1
        print(f"idx: {idx}")


    # plot
    plt.plot(vals_base, label='DDPG')
    plt.plot(vals_mpc,  label='RNN‑MPC')
    plt.legend(); plt.title('Portfolio Value Over Time')
    plt.show()