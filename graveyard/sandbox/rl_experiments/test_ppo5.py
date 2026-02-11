import numpy as np
import pandas as pd
import gym
from gym import spaces
from scipy.signal import find_peaks
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt

# === 1) Generate synthetic price series ===
np.random.seed(42)
n_steps = 500
sinusoid = np.sin(np.linspace(0, 6 * np.pi, n_steps)) * 10
drift = np.linspace(-20, 20, n_steps)
noise = np.random.normal(0, 1, n_steps)
prices = 100 + sinusoid + drift + noise
highs = prices + np.random.uniform(0.5, 2.0, n_steps)
lows = prices - np.random.uniform(0.5, 2.0, n_steps)
df = pd.DataFrame({'price': prices, 'high': highs, 'low': lows})

# === 2) Label global extrema with prominence ===
prominence_val = 8.0
peaks, _ = find_peaks(prices, prominence=prominence_val)
troughs, _ = find_peaks(-prices, prominence=prominence_val)
df['label'] = 0
df.loc[peaks, 'label']   = -1   # global peaks
df.loc[troughs, 'label'] = +1   # global troughs

# === 3) Feature engineering: recent returns + future swings ===
lags = 3
X = []
for i in range(n_steps):
    feats = []
    for lag in range(1, lags+1):
        j = max(0, i-lag)
        feats.append(prices[i] - prices[j])
    future = prices[i+1:] if i+1 < n_steps else np.array([prices[i]])
    feats.append(future.max() - prices[i])
    feats.append(prices[i] - future.min())
    X.append(feats)
X = np.array(X)
y = df['label'].values

# === 4) Train/test split & predictor training ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
predictor = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=0)
predictor.fit(X_train, y_train)
enum = predictor.classes_
proba = predictor.predict_proba(X)
idx_peak   = list(enum).index(-1)
idx_trough = list(enum).index(1)
df['conf_peak']   = proba[:, idx_peak]
df['conf_trough'] = proba[:, idx_trough]

# === 5) Define Trading Environment with label & confidence reward ===
class TwoStageTradingEnv(gym.Env):
    def __init__(self, df, initial_cash=10000, max_trades=5, trade_penalty=0.1,
                 label_bonus=100.0, conf_bonus=0.2):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.max_steps = len(df) - 1
        self.max_trades = max_trades
        self.trade_penalty = trade_penalty
        self.label_bonus = label_bonus
        self.conf_bonus = conf_bonus

        self.action_space = spaces.Discrete(3)
        obs_low = np.zeros(9, dtype=np.float32)
        obs_high = np.array([
            np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
            np.finfo(np.float32).max, np.finfo(np.float32).max,
            float(max_trades), 1.0, 1.0, float(self.max_steps)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def reset(self):
        self.step_idx = 0
        self.cash = self.initial_cash
        self.holdings = 0
        self.trades_made = 0
        return self._get_obs()

    def _get_obs(self):
        r = self.df.iloc[self.step_idx]
        return np.array([r.price, r.high, r.low,
                         self.cash, self.holdings,
                         self.max_trades - self.trades_made,
                         r.conf_trough, r.conf_peak,
                         self.max_steps - self.step_idx], dtype=np.float32)

    def step(self, action):
        r = self.df.iloc[self.step_idx]
        prev_val = self.cash + self.holdings * r.price
        reward = 0.0
        # next price
        next_price = self.df.iloc[min(self.step_idx+1, self.max_steps)].price
        # LABEL and CONFIDENCE reward on action
        if action == 1 and self.trades_made < self.max_trades and self.cash >= r.low:
            # label-based bonus: only if label==+1 (trough)
            if r.label == 1:
                reward += self.label_bonus
            # confidence-based small bonus
            reward += self.conf_bonus * r.conf_trough
            # execute buy
            self.cash -= r.low + self.trade_penalty
            self.holdings += 1
            self.trades_made += 1
            reward -= self.trade_penalty
        elif action == 2 and self.trades_made < self.max_trades and self.holdings > 0:
            if r.label == -1:
                reward += self.label_bonus
            reward += self.conf_bonus * r.conf_peak
            self.cash += r.high - self.trade_penalty
            self.holdings -= 1
            self.trades_made += 1
            reward -= self.trade_penalty
        # advance
        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        # continuous P&L
        curr_val = self.cash + self.holdings * r.price
        reward += curr_val - prev_val
        if done:
            reward += (curr_val - self.initial_cash)
        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        return obs, reward, done, {'portfolio_value': curr_val}

# === 6) Train PPO ===
base_env = DummyVecEnv([lambda: TwoStageTradingEnv(df)])
env = VecNormalize(base_env, norm_obs=True, norm_reward=False)
model = PPO('MlpPolicy', env, verbose=1,
            learning_rate=3e-4, ent_coef=0.01,
            batch_size=32, n_steps=64, n_epochs=5)
model.learn(total_timesteps=100000)

# === 7) Evaluate and plot ===
test_env = TwoStageTradingEnv(df)
obs = test_env.reset()
prices_list, port_vals = [], []
buys_x, buys_y, sells_x, sells_y = [], [], [], []
for t in range(n_steps - 1):
    prices_list.append(test_env.df.iloc[test_env.step_idx].price)
    action, _ = model.predict(obs, deterministic=False)
    p = test_env.df.iloc[test_env.step_idx].price
    if action == 1 and test_env.trades_made < test_env.max_trades:
        buys_x.append(t); buys_y.append(p)
    if action == 2 and test_env.trades_made < test_env.max_trades:
        sells_x.append(t); sells_y.append(p)
    obs, _, done, info = test_env.step(action)
    port_vals.append(info['portfolio_value'])
    if done: break
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8), sharex=True)
ax1.plot(prices_list)
ax1.scatter(peaks, prices[peaks], marker='^', color='green', s=100, label='True Peaks')
ax1.scatter(troughs, prices[troughs], marker='v', color='red', s=100, label='True Troughs')
ax1.scatter(buys_x, buys_y, marker='x', color='blue', s=60, label='Agent Buys')
ax1.scatter(sells_x, sells_y, marker='o', color='orange', s=60, label='Agent Sells')
ax1.legend(); ax1.set_title('Price, Extrema & Trades')
ax2.plot(port_vals); ax2.set_title('Portfolio Value')
plt.tight_layout(); plt.show()
