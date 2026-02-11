import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# 1. SYNTHETIC PRICE DATA
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=200, freq="D")
# random walk for close price
close = 100 + np.cumsum(np.random.randn(len(dates)) * 2)
# highs and lows around close
high  = close + np.random.rand(len(dates)) * 2
low   = close - np.random.rand(len(dates)) * 2

data = pd.DataFrame({
    "date":  dates,
    "high":  high,
    "low":   low,
    "close": close
})

print("Sample data:\n", data.head())



# 2. CUSTOM GYM ENVIRONMENT
class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.max_steps = len(df) - 1
        self.initial_balance = 10_000
        
        # Actions: 0 = hold, 1 = buy-all, 2 = sell-all
        self.action_space = spaces.Discrete(3)
        # Observations: [current_price, balance, shares_held]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.step_idx    = 0
        self.balance     = self.initial_balance
        self.shares      = 0.0
        self.total_asset = self.initial_balance
        return self._get_obs()

    def _get_obs(self):
        price = self.df.loc[self.step_idx, "close"]
        return np.array([price, self.balance, self.shares], dtype=np.float32)

    def step(self, action):
        price = self.df.loc[self.step_idx, "close"]
        # Execute action
        if action == 1:   # BUY
            self.shares = self.balance / price
            self.balance = 0.0
        elif action == 2: # SELL
            self.balance = self.shares * price
            self.shares = 0.0

        self.step_idx += 1
        done = self.step_idx >= self.max_steps

        # Recompute total asset
        new_price = self.df.loc[self.step_idx, "close"]
        self.total_asset = self.balance + self.shares * new_price
        reward = self.total_asset - self.initial_balance

        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        print(f"Step {self.step_idx}: Balance={self.balance:.2f}, "
              f"Shares={self.shares:.4f}, Asset={self.total_asset:.2f}")

# Instantiate environment
env = StockTradingEnv(data)


# 3. TRAIN PPO AGENT
model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=1e-3,
        n_steps=16,       # tiny rollouts
        batch_size=8,     # small batches
        n_epochs=10       # more epochs per update
    )
model.learn(total_timesteps=10000)


# 4. EVALUATION
obs = env.reset()
asset_values = []
for _ in range(len(data)-1):
    action, _ = model.predict(obs)

    print(obs, action)
    obs, reward, done, info = env.step(action)
    asset_values.append(env.total_asset)
    if done:
        break

# 5. PLOT PERFORMANCE
plt.figure(figsize=(10, 4))
plt.plot(data["date"][1:], asset_values)
plt.title("PPO Trading Agent: Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Total Asset Value (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
