import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt

class StockTradingEnv(gym.Env):
    """
    Trading environment with limited trades, time-weighted rewards to encourage later trades,
    and intrinsic reward shaping based on future price movement.
    Actions: 0=Hold, 1=Buy, 2=Sell
    Obs: [price, high, low, cash, holdings, trades_left, steps_remaining]
    Reward: time-weighted P&L + intrinsic shaping - flat trade cost + terminal bonus
    """
    def __init__(self, df, initial_cash=10000, max_trades=5, trade_cost=0.1):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.max_steps = len(df) - 1
        self.max_trades = max_trades
        self.trade_cost = trade_cost

        self.action_space = spaces.Discrete(3)
        obs_low = np.zeros(7, dtype=np.float32)
        obs_high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            float(max_trades),
            float(self.max_steps)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def reset(self):
        self.step_idx = 0
        self.cash = self.initial_cash
        self.holdings = 0
        self.trades_made = 0
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.step_idx]
        return np.array([
            row['price'], row['high'], row['low'],
            self.cash, self.holdings,
            self.max_trades - self.trades_made,
            self.max_steps - self.step_idx
        ], dtype=np.float32)

    def step(self, action):
        done = False
        row = self.df.iloc[self.step_idx]
        prev_val = self.cash + self.holdings * row['price']
        reward = 0.0

        # Flat trade cost
        cost = self.trade_cost

        # Intrinsic shaping and trade execution
        if self.trades_made < self.max_trades:
            if action == 1 and self.cash >= row['low']:
                future_price = self.df.iloc[min(self.step_idx + 1, self.max_steps)]['price']
                intrinsic = future_price - row['price']
                self.cash -= row['low'] + cost
                self.holdings += 1
                self.trades_made += 1
                reward += intrinsic - cost
            elif action == 2 and self.holdings > 0:
                future_price = self.df.iloc[min(self.step_idx + 1, self.max_steps)]['price']
                intrinsic = row['price'] - future_price
                self.cash += row['high'] - cost
                self.holdings -= 1
                self.trades_made += 1
                reward += intrinsic - cost
        # Advance
        self.step_idx += 1
        if self.step_idx >= self.max_steps:
            done = True
            final = self.cash + self.holdings * row['price']
            terminal_bonus = final - self.initial_cash
        else:
            terminal_bonus = 0.0

        # Intermediate profit/loss
        current_val = self.cash + self.holdings * row['price']
        pnl = current_val - prev_val
        reward += pnl

        # Time-weighted reward: scale by (step_idx / max_steps)
        time_weight = self.step_idx / self.max_steps
        reward *= time_weight

        reward += terminal_bonus
        obs = self._get_obs()
        info = {'portfolio_value': current_val}
        return obs, reward, done, info

if __name__ == '__main__':
    # Synthetic varying trend data
    np.random.seed(0)
    n_steps = 200
    sinusoid = np.sin(np.linspace(0, 4 * np.pi, n_steps)) * 10
    drift = np.linspace(-10, 10, n_steps)
    noise = np.random.normal(0, 0.5, n_steps)
    prices = 100 + sinusoid + drift + noise
    highs = prices + np.random.uniform(0.5, 1.5, n_steps)
    lows = prices - np.random.uniform(0.5, 1.5, n_steps)
    df = pd.DataFrame({'price': prices, 'high': highs, 'low': lows})

    max_trades = 5
    base_env = DummyVecEnv([lambda: StockTradingEnv(df, max_trades=max_trades, trade_cost=0.1)])
    env = VecNormalize(base_env, norm_obs=True, norm_reward=False)

    model = PPO(
        'MlpPolicy', env,
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.05,
        vf_coef=0.5,
        clip_range=0.2,
        n_steps=32,
        batch_size=16,
        n_epochs=5
    )
    model.learn(total_timesteps=100000)

    # Evaluate and plot
    test_env = StockTradingEnv(df, max_trades=max_trades, trade_cost=0.1)
    obs = test_env.reset()
    buys_x, buys_y, sells_x, sells_y = [], [], [], []
    prices_list, port_vals = [], []
    for t in range(n_steps - 1):
        prices_list.append(test_env.df.iloc[test_env.step_idx]['price'])
        action, _ = model.predict(obs, deterministic=False)
        price = test_env.df.iloc[test_env.step_idx]['price']
        if action == 1 and test_env.trades_made < test_env.max_trades:
            buys_x.append(t); buys_y.append(price)
        if action == 2 and test_env.trades_made < test_env.max_trades:
            sells_x.append(t); sells_y.append(price)
        obs, _, done, info = test_env.step(action)
        port_vals.append(info['portfolio_value'])
        if done: break

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(prices_list, label='Price')
    ax1.scatter(buys_x, buys_y, marker='^', s=50, label='Buy')
    ax1.scatter(sells_x, sells_y, marker='v', s=50, label='Sell')
    ax1.legend(); ax1.set_title('Price & Trades')
    ax2.plot(port_vals, label='Portfolio Value')
    ax2.legend(); ax2.set_title('Portfolio Value Over Time')
    plt.tight_layout(); plt.show()
