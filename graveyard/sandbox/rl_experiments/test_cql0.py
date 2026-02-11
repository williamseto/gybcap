import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt
import d3rlpy

from scipy.signal import find_peaks
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# 1. Generate sample stock data (synthetic sine wave + noise)
def generate_synthetic_data(length=1000, seed=42):
    np.random.seed(seed)
    t = np.arange(length)
    prices = 50 + 5 * np.sin(0.02 * t) + np.random.normal(0, 0.5, size=length)

    df = pd.DataFrame({'price': prices})
    peaks, _ = find_peaks(prices, prominence=8.0)
    troughs, _ = find_peaks(-prices, prominence=8.0)

    df['true_label'] = 0
    df.loc[peaks, 'true_label'] = -1
    df.loc[troughs, 'true_label'] = 1

    # lags = 3
    # X = []
    # for i in range(len(df)):
    #     feats = []
    #     for lag in range(1, lags + 1):
    #         j = max(0, i - lag)
    #         feats.append(prices[i] - prices[j])
    #     future = prices[i+1:] if i+1 < len(prices) else np.array([prices[i]])
    #     feats.append(future.max() - prices[i])
    #     feats.append(prices[i] - future.min())
    #     X.append(feats)
    # X = np.array(X)
    # y = df['true_label'].values

    # # Train/test split & predictor training
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # predictor = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0)
    # predictor.fit(X_train, y_train)

    # # Predict on full dataset
    # y_pred = predictor.predict(X)
    # df['pred_label'] = y_pred


    # # Plot price with true and predicted extrema
    # plt.figure(figsize=(12, 6))
    # plt.plot(prices, label='Price')
    # # True extrema
    # plt.scatter(peaks, prices[peaks], marker='^', color='green', s=30, label='True Peaks')
    # plt.scatter(troughs, prices[troughs], marker='v', color='red', s=30, label='True Troughs')
    # # Predicted extrema
    # pred_peaks = df.index[df['pred_label'] == 2].to_numpy()
    # pred_troughs = df.index[df['pred_label'] == 1].to_numpy()
    # plt.scatter(pred_peaks, prices[pred_peaks], marker='x', color='blue', s=60, label='Predicted Peaks')
    # plt.scatter(pred_troughs, prices[pred_troughs], marker='o', color='orange', s=60, label='Predicted Troughs')

    # plt.title('Global Extrema: True vs Predicted')
    # plt.xlabel('Timestep')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # exit()

    return df

# 2. Define a simple trading environment
class TradingEnv(gym.Env):
    def __init__(self, data, window_size=10):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.current_step = window_size
        self.position = 0  # 0 = flat, 1 = long
        self.entry_price = None
        # Observation: window of past prices
        low_range = np.array([0] * (window_size-1) + [-1.0], dtype=np.float32)
        high_range = np.array([1.0] * (window_size-1) + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_range, high=high_range, shape=(window_size,), dtype=np.float32
        )
        # Actions: 0 = sell/hold, 1 = buy/hold
        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = None
        return self._get_obs()

    def _get_obs(self):
        # prices = self.data['price'].values[self.current_step-self.window_size:self.current_step].astype(np.float32)
        prices = self.data['price'].values[self.current_step-self.window_size:self.current_step].astype(np.float32)
        prices_diff_feat = sigmoid(np.diff(prices))
        return np.append(prices_diff_feat, self.data['true_label'].iloc[self.current_step])

    def step(self, action):
        reward = 0.0
        price = self.data['price'].iloc[self.current_step]
        prev_price = self.data['price'].iloc[self.current_step-1]
        info = {}
        # Execute trade
        if action == 1 and self.position == 0:
            self.entry_price = prev_price
            self.position = 1
            info['trade'] = ('buy', self.current_step)
        elif action == 2 and self.position == 1:
            reward = price - self.entry_price
            info['trade'] = ('sell', self.current_step)
            self.position = 0


        mapped_action = action
        if action == 2:
            mapped_action = -1
        if mapped_action == self.data['true_label'].iloc[self.current_step]:
            if action != 0:
                reward += 100.0
            else:
                reward += 0.5
        else:
            reward -= 0.1


        # Advance
        self.current_step += 1
        done = self.current_step >= len(self.data)
        obs = self._get_obs() if not done else None
        return obs, reward, done, info

# 3. Seed policy: simple one-step predictor
# If next price > current price: buy (1), else sell/hold (0)
def generate_offline_dataset(env):

    observations, actions, rewards, next_obs, terminals = [], [], [], [], []
    obs = env.reset()
    done = False
    while not done:
        curr_price = env.data['price'].iloc[env.current_step-1]
        next_price = env.data['price'].iloc[env.current_step]

        # action = 1 if next_price > curr_price else 0


        action = env.data['true_label'].iloc[env.current_step]

        if action == -1:
            action = 2

        o2, r, done, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(r)
        # next_obs.append(o2 if o2 is not None else np.zeros_like(obs))
        terminals.append(done)
        obs = o2
    return d3rlpy.dataset.MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=bool)
    )

# 4. Main training & evaluation with visualization
if __name__ == '__main__':
    # Generate data & env
    df = generate_synthetic_data(length=2000)

    env = TradingEnv(df)

    # Build offline dataset
    dataset = generate_offline_dataset(env)

    # Train CQL agent offline
    # cql = d3rlpy.algos.DiscreteCQL(d3rlpy.algos.DiscreteCQLConfig(
    #     learning_rate=3e-4,
    #     batch_size=256),
    #     device='cpu',
    #     enable_ddp=False
    # )
    # cql.fit(dataset, n_steps=100000, save_interval=10)

    cql = d3rlpy.load_learnable("d3rlpy_logs/DiscreteCQL_20250518142904/model_100000.d3")

    # Evaluate learned policy and record trades
    eval_env = TradingEnv(generate_synthetic_data(length=2000, seed=8))
    obs = eval_env.reset()
    done = False
    buy_steps, sell_steps = [], []
    while not done:
        action = cql.predict(np.array([obs]))[0]
        obs, reward, done, info = eval_env.step(action)
        if info.get('trade'):
            typ, step = info['trade']
            if typ == 'buy': buy_steps.append(step)
            else: sell_steps.append(step)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df['price'].values, label='Price', zorder=-1)
    plt.scatter(buy_steps, df['price'].iloc[buy_steps], marker='^', label='Buys', color='green', zorder=1)
    plt.scatter(sell_steps, df['price'].iloc[sell_steps], marker='v', label='Sells', color='red', zorder=1)
    plt.title('Trades Executed During Evaluation')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
