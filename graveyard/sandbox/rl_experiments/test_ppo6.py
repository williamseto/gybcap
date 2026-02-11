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

# === 2) Label global extrema and train predictor ===
prominence_val = 8.0
peaks, _ = find_peaks(prices, prominence=prominence_val)
troughs, _ = find_peaks(-prices, prominence=prominence_val)
df['label'] = 0
df.loc[peaks, 'label'] = -1
df.loc[troughs, 'label'] = 1
# feature engineering for predictor
def make_features(prices, highs, lows):
    lags = 3
    X = []
    for i in range(len(prices)):
        feats = []
        for lag in range(1, lags+1):
            j = max(0, i-lag)
            feats.append(prices[i] - prices[j])
        future = prices[i+1:] if i+1 < len(prices) else np.array([prices[i]])
        feats.append(future.max() - prices[i])
        feats.append(prices[i] - future.min())
        # feats.append(highs[i] - lows[i])
        X.append(feats)
    return np.array(X)
X = make_features(prices, highs, lows)
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
predictor = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=0)
predictor.fit(X_train, y_train)
proba = predictor.predict_proba(X)
cls = predictor.classes_
idx_peak = list(cls).index(-1)
idx_trough = list(cls).index(1)
df['conf_peak'] = proba[:, idx_peak]
df['conf_trough'] = proba[:, idx_trough]


y_pred = predictor.predict(X)


# # plt.figure(figsize=(12, 6))
# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), sharex=True)
# ax1.plot(prices, label='Price')
# # True extrema
# ax1.scatter(peaks, prices[peaks], marker='^', color='green', s=100, label='True Peaks')
# ax1.scatter(troughs, prices[troughs], marker='v', color='red', s=100, label='True Troughs')
# # Predicted extrema
# pred_peaks = df.index[y_pred == -1].to_numpy()
# pred_troughs = df.index[y_pred == 1].to_numpy()
# ax1.scatter(pred_peaks, prices[pred_peaks], marker='x', color='blue', s=60, label='Predicted Peaks')
# ax1.scatter(pred_troughs, prices[pred_troughs], marker='o', color='orange', s=60, label='Predicted Troughs')

# ax1.set_title('Global Extrema: True vs Predicted')
# # ax1.xlabel('Timestep')
# # ax1.ylabel('Price')
# ax1.legend()

# ax2.plot(df['conf_trough'], label='Confidence (Trough)')
# ax2.plot(df['conf_peak'], label='Confidence (Peak)')
# ax2.set_ylabel('Confidence')
# ax2.legend()

# plt.show()
# exit()

# === 3) High-level Timing Environment with confidences ===
class TimingEnv(gym.Env):
    def __init__(self, df, max_steps, max_trades):
        super().__init__()
        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0=wait,1=enter
        self.df = df.reset_index(drop=True)
        self.max_steps = max_steps
        self.max_trades = max_trades

    def reset(self):
        self.step_idx = 0
        self.trades_made = 0
        return self._obs()

    def _obs(self):
        row = self.df.iloc[self.step_idx]

        adj_label = row['label']
        if adj_label == 1:
            adj_label = 0.333
        elif adj_label == -1:
            adj_label = 0.666

        return np.array([
            row['conf_trough'],
            row['conf_peak'],
            adj_label
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.step_idx]
        reward = 0.0
        # if action == 1 and self.trades_made < self.max_trades:
        #     # reward proportional to confidence when label present
        #     if row['label'] != 0:
        #         reward += row['conf_trough'] + row['conf_peak']
        #     else:
        #         # heavy penalty for false trade
        #         reward -= 1.0

        if action == abs(row['label']):
            if action != 0:
                reward += 100.0
            else:
                reward += 0.5
        else:
            reward -= 5.0



        self.step_idx += 1
        done = (self.step_idx >= self.max_steps)
        return self._obs(), reward, done, {}

# === 4) Low-level Trading Environment ===
class TradingEnv(gym.Env):
    def __init__(self, df, initial_cash=10000, max_trades=1):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.max_steps = len(df) - 1
        self.max_trades = max_trades
        self.action_space = spaces.Discrete(3)  # hold, buy, sell
        low = np.zeros(5, dtype=np.float32)
        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            float(max_trades),
            float(self.max_steps)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        self.step_idx = 0
        self.cash = self.initial_cash
        self.holdings = 0
        self.trades_made = 0
        return self._obs()

    def _obs(self):
        r = self.df.iloc[self.step_idx]
        return np.array([
            r.price, r.high, r.low,
            self.max_trades - self.trades_made,
            self.max_steps - self.step_idx
        ], dtype=np.float32)

    def step(self, action):
        r = self.df.iloc[self.step_idx]
        reward = 0.0
        if action == 1 and self.trades_made < self.max_trades and self.cash >= r.low:
            self.cash -= r.low
            self.holdings += 1
            self.trades_made += 1
        elif action == 2 and self.trades_made < self.max_trades and self.holdings > 0:
            self.cash += r.high
            self.holdings -= 1
            self.trades_made += 1
        self.step_idx += 1
        done = self.step_idx >= self.max_steps
        current_val = self.cash + self.holdings * r.price
        reward += current_val - self.initial_cash if done else current_val - (self.cash + self.holdings * r.price)
        return (self._obs() if not done else np.zeros_like(self._obs()), reward, done, {'portfolio_value': current_val})

# === 5) Train high-level timing agent ===
timing_env = DummyVecEnv([lambda: TimingEnv(df, n_steps-1, max_trades=5)])
timing_env = VecNormalize(timing_env, norm_obs=True, norm_reward=False)
timing_model = PPO('MlpPolicy', timing_env, verbose=1, n_steps=500)
# timing_model = PPO(
#         'MlpPolicy',
#         timing_env,
#         verbose=1,
#         learning_rate=1e-4,
#         n_steps=64,       # tiny rollouts
#         batch_size=64,     # small batches
#         n_epochs=10       # more epochs per update
#     )
timing_model.learn(total_timesteps=100000)


# Intermediate visualization of timing agent vs confidence
viz_env = TimingEnv(df, n_steps-1, max_trades=5)
obs = viz_env.reset()
conf_tr = []
conf_pr = []
enter_actions = []
for _ in range(n_steps-1):
    row = df.iloc[viz_env.step_idx]
    conf_tr.append(row['conf_trough'])
    conf_pr.append(row['conf_peak'])

    action, _ = timing_model.predict(obs)
    enter_actions.append(action)
    obs, _, done, _ = viz_env.step(action)
    if done: break

plt.figure(figsize=(10,4))
plt.plot(conf_tr, label='Conf Trough')
plt.plot(conf_pr, label='Conf Peak')
plt.plot(np.array(enter_actions)*0.5, label='Enter Action', linestyle='--')
plt.legend(); plt.title('Timing Agent: Confidences vs Enter Decisions')

# plt.figure(figsize=(10,4))
# plt.plot(df['price'], label='Price')
plt.show()

exit()

# === 6) Train low-level execution agent ===
env_exec = DummyVecEnv([lambda: TradingEnv(df, initial_cash=10000, max_trades=1)])
env_exec = VecNormalize(env_exec, norm_obs=True, norm_reward=False)
exec_model = PPO('MlpPolicy', env_exec, verbose=1)
exec_model.learn(total_timesteps=50000)

# === 7) Orchestrate and evaluate ===
timing_test = TimingEnv(df, n_steps-1, max_trades=5)
trading_test = TradingEnv(df, initial_cash=10000, max_trades=1)
obs_t = timing_test.reset()
obs_e = trading_test.reset()
prices_list, port_vals = [], []
buys, sells = [], []
for t in range(n_steps-1):
    prices_list.append(prices[t])
    enter, _ = timing_model.predict(obs_t)
    if enter:
        action, _ = exec_model.predict(obs_e, deterministic=False)
    else:
        action = 0
    if action == 1: buys.append(t)
    elif action == 2: sells.append(t)
    obs_t, _, done_t, _ = timing_test.step(enter)
    obs_e, _, done_e, info_e = trading_test.step(action)
    port_vals.append(info_e.get('portfolio_value', np.nan))
    if done_t or done_e: break

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), sharex=True)
ax1.plot(prices_list, label='Price')
ax1.scatter(peaks, prices[peaks], marker='^', c='g', label='Peaks')
ax1.scatter(troughs, prices[troughs], marker='v', c='r', label='Troughs')
ax1.scatter(buys, np.array(prices)[buys], marker='x', c='b', label='Buys')
ax1.scatter(sells, np.array(prices)[sells], marker='o', c='orange', label='Sells')
ax1.legend(); ax1.set_title('Hierarchical RL Trades')
ax2.plot(port_vals, label='Portfolio Value')
ax2.set_title('Portfolio Value Over Time'); ax2.legend(); plt.tight_layout(); plt.show()
