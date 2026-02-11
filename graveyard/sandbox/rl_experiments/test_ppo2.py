import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

def get_price_df(n_steps):
    prices = 100 + np.cumsum(np.random.randn(n_steps) * 2)

    # Generate realistic high/low spreads
    spreads_high = np.random.uniform(0.5, 1.5, n_steps)
    spreads_low = np.random.uniform(0.5, 1.5, n_steps)
    highs = prices + spreads_high
    lows = prices - spreads_low

    return pd.DataFrame({'price': prices, 'high': highs, 'low': lows})

class StockTradingEnv(gym.Env):
    """
    Deterministic trading environment to encourage overfitting.
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    Observation: [current_price, high, low, cash, stock_holdings]
    Reward: change in total portfolio value.
    """

    def __init__(self, df, initial_cash=10000):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.max_steps = len(df) - 1

        self.action_space = spaces.Discrete(3)
        # Observations: price, high, low, cash, holdings
        obs_low = np.array([0, 0, 0], dtype=np.float32)
        obs_high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = 0
        self._update_portfolio()
        return self._get_observation()

    def _update_portfolio(self):
        row = self.df.iloc[self.current_step]
        self.current_price = row['price']
        self.current_high = row['high']
        self.current_low = row['low']
        self.portfolio_value = self.cash + self.holdings * self.current_price

    def _get_observation(self):
        return np.array([
            self.current_price,
            self.cash,
            self.holdings
        ], dtype=np.float32)

    def step(self, action):
        prev_value = self.portfolio_value
        # Deterministic transaction: buy at low, sell at high
        if action == 1 and self.holdings == 0 and self.cash >= self.current_low:
            self.cash -= self.current_low
            self.holdings += 1
        elif action == 2 and self.holdings > 0:
            self.cash += self.current_high
            self.holdings -= 1

        self.current_step += 1
        done = self.current_step >= self.max_steps
        self._update_portfolio()

        # Reward scaled to encourage larger portfolios
        reward = (self.portfolio_value - prev_value) * 10
        return self._get_observation(), reward, done, {'portfolio_value': self.portfolio_value}

if __name__ == '__main__':
    # Generate a varying trend with variable highs and lows
    # np.random.seed(0)
    n_steps = 200
    # # Combine sinusoidal cycles and linear drift for up/down movements
    # sinusoid = np.sin(np.linspace(0, 4 * np.pi, n_steps)) * 10
    # drift = np.linspace(-10, 10, n_steps)
    # noise = np.random.normal(0, 0.5, n_steps)
    # prices = 100 + sinusoid + drift + noise

    df = get_price_df(n_steps)

    # Create env
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    # Overfit: small dataset, lots of timesteps, small batch size
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=1e-3,
        n_steps=16,       # tiny rollouts
        batch_size=8,     # small batches
        n_epochs=10       # more epochs per update
    )
    # Train on same data to overfit
    model.learn(total_timesteps=10000)
    
    test_df = get_price_df(n_steps)
    # test_df = df

    test_prices = test_df['price']

    # Evaluate on training set, record actions
    test_env = StockTradingEnv(test_df)
    obs = test_env.reset()
    portfolio_values = []
    buy_steps, buy_prices = [], []
    sell_steps, sell_prices = [], []

    for t in range(n_steps - 1):
        action, _ = model.predict(obs)
        print(obs, action)
        # record entry/exit points
        price = test_env.current_price
        if action == 1:
            buy_steps.append(t)
            buy_prices.append(price)
        elif action == 2:
            sell_steps.append(t)
            sell_prices.append(price)

        obs, _, done, info = test_env.step(action)
        portfolio_values.append(info['portfolio_value'])
        if done:
            break

    # Plot price with entry/exit markers and portfolio value
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(test_prices, label='Price')
    ax1.scatter(buy_steps, buy_prices, marker='^', label='Buy', s=50)
    ax1.scatter(sell_steps, sell_prices, marker='v', label='Sell', s=50)
    ax1.set_title('Price with Entry/Exit Points')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(portfolio_values, label='Portfolio Value')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Value')
    ax2.legend()

    plt.tight_layout()

    plt.figure()
    plt.plot(df['price'], label='Price')


    plt.legend()
    plt.show()

