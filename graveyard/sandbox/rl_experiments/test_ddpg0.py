import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from collections import deque
import random
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# —————————————————————————————————————————————————
# 1) Execution Environment (same as before, continuous action)
# —————————————————————————————————————————————————
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

        # continuous action in [-1,1]
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        # obs: [price_t, position, drawdown]
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

        # PnL and portfolio update
        pnl = action * ret
        self.portfolio_value *= np.exp(pnl)
        self.peak = max(self.peak, self.portfolio_value)
        self.drawdown = (self.peak - self.portfolio_value) / self.peak

        # risk penalty
        risk_penalty = self.lambda_risk * (action**2) * abs(ret)
        reward = pnl - risk_penalty

        # update state
        self.current_step += 1
        done = self.current_step >= min(self.n_steps, self.max_steps)
        self.price = new_price
        self.position = action

        obs = self._get_obs()
        info = {'ret': ret, 'reward': reward, 'action': action, 'portfolio_value': self.portfolio_value}
        return obs, reward, done, info

# —————————————————————————————————————————————————
# 2) DDPG for data collection (warm‑start) 
# —————————————————————————————————————————————————
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

def collect_data(env, model, steps=20000):
    buffer = []
    obs = env.reset()
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, done, info = env.step(action)
        buffer.append((obs, action, reward, next_obs))
        obs = next_obs if not done else env.reset()
    return buffer

# —————————————————————————————————————————————————
# 3) Surrogate Dynamics Model 
# —————————————————————————————————————————————————
class DynamicsModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, obs_dim + 1)  # predict next_obs and immediate reward
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        out = self.net(x)
        next_obs = out[..., :-1]
        reward   = out[..., -1:]
        return next_obs, reward

def train_dynamics(model, buffer, epochs=50, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    for _ in range(epochs):
        random.shuffle(buffer)
        for i in range(0, len(buffer), batch_size):
            batch = buffer[i:i+batch_size]
            obs_b = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            act_b = torch.tensor([b[1] for b in batch], dtype=torch.float32)
            rew_b = torch.tensor([[b[2]] for b in batch], dtype=torch.float32)
            next_b = torch.tensor([b[3] for b in batch], dtype=torch.float32)

            pred_next, pred_rew = model(obs_b, act_b)
            loss = mse_loss(pred_next, next_b) + mse_loss(pred_rew, rew_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

# —————————————————————————————————————————————————
# 4) MPC Planner (random shooting)
# —————————————————————————————————————————————————
def mpc_action(model, cur_obs, plan_horizon=5, n_candidates=256):
    # sample random action sequences, evaluate imagined cumulative reward
    obs_tile = torch.tensor(cur_obs, dtype=torch.float32).unsqueeze(0)
    best_return = -1e9
    best_first = 0.0
    for _ in range(n_candidates):
        seq = torch.rand(plan_horizon, 1)*2 - 1  # uniform in [-1,1]
        sim_obs = obs_tile.clone()
        cum_rew = 0.0
        for t in range(plan_horizon):
            a = seq[t].unsqueeze(0)
            sim_obs, rew = model(sim_obs, a)
            cum_rew += rew.item()
        if cum_rew > best_return:
            best_return = cum_rew
            best_first  = seq[0].item()
    return np.array([best_first], dtype=np.float32)

# —————————————————————————————————————————————————
# 5) Main training & evaluation loop
# —————————————————————————————————————————————————
if __name__ == "__main__":
    # 1) Warm‑start DDPG
    env = ExecutionEnv()
    action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.2*np.ones(1))

    vec_env = DummyVecEnv([lambda: ExecutionEnv()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    ddpg = DDPG("MlpPolicy", vec_env, verbose=1)
    ddpg.learn(total_timesteps=15000)

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

    # exit()

    # 2) Collect data & train dynamics model
    data_buffer = collect_data(env, ddpg, steps=20000)
    dyn_model = DynamicsModel(obs_dim=3, act_dim=1)
    train_dynamics(dyn_model, data_buffer, epochs=100)

    # 3) Evaluate MPC‑planner
    obs = env.reset()
    positions, port_vals = [], []
    done = False
    while not done:
        a = mpc_action(dyn_model, obs, plan_horizon=10, n_candidates=512)
        obs, rew, done, info = env.step(a)
        positions.append(env.position)
        port_vals.append(env.portfolio_value)

    # Plot results
    plt.subplot(3,1,1)
    plt.plot(positions); plt.title("MPC Planner: Position over Time")
    plt.subplot(3,1,2)
    plt.plot(port_vals); plt.title("MPC Planner: Portfolio Value")
    plt.subplot(3,1,3)
    plt.plot(env.price_series); plt.title("MPC Planner: Price")
    plt.tight_layout(); plt.show()
