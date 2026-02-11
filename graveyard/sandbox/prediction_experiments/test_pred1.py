import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

# Recreate synthetic data and predictor as in the example
np.random.seed(42)
n_steps = 500
sinusoid = np.sin(np.linspace(0, 6 * np.pi, n_steps)) * 10
drift = np.linspace(-20, 20, n_steps)
noise = np.random.normal(0, 1, n_steps)
prices = 100 + sinusoid + drift + noise
highs = prices + np.random.uniform(0.5, 2.0, n_steps)
lows = prices - np.random.uniform(0.5, 2.0, n_steps)
df = pd.DataFrame({'price': prices, 'high': highs, 'low': lows})

# Label global extrema
prominence_val = 8.0
peaks, _ = find_peaks(prices, prominence=prominence_val)
troughs, _ = find_peaks(-prices, prominence=prominence_val)
df['label'] = 0
df.loc[peaks, 'label'] = -1
df.loc[troughs, 'label'] = 1

# Feature engineering
lags = 3
X = []
for i in range(n_steps):
    feats = []
    for lag in range(1, lags+1):
        j = max(0, i - lag)
        feats.append(prices[i] - prices[j])
    future = prices[i+1:] if i+1 < n_steps else np.array([prices[i]])
    feats.append(future.max() - prices[i])
    feats.append(prices[i] - future.min())
    X.append(feats)
X = np.array(X)
y = df['label'].values

# Train predictor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
predictor = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=0)
predictor.fit(X_train, y_train)

# Compute confidences
proba = predictor.predict_proba(X)
classes = predictor.classes_
idx_peak = list(classes).index(-1)
idx_trough = list(classes).index(1)
df['conf_peak'] = proba[:, idx_peak]
df['conf_trough'] = proba[:, idx_trough]

# Compute next-step returns
returns = np.empty(n_steps)
returns[:-1] = prices[1:] - prices[:-1]
returns[-1] = 0

# Plot confidences and returns
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
ax1.plot(df['conf_trough'], label='Confidence (Trough)')
ax1.plot(df['conf_peak'], label='Confidence (Peak)')
ax1.set_ylabel('Confidence')
ax1.legend()
ax1.set_title('Predictor Confidence Trajectories')

ax2.plot(returns, color='gray', label='Next-step Return')
ax2.set_ylabel('Return')
ax2.set_xlabel('Timestep')
ax2.legend()

ax3.plot(df['price'], label='Price')
ax3.set_ylabel('Price')
ax3.legend()

plt.tight_layout()
plt.show()

# Compute correlations
corr_trough = np.corrcoef(df['conf_trough'][:-1], returns[:-1])[0, 1]
corr_peak = np.corrcoef(df['conf_peak'][:-1], returns[:-1])[0, 1]

print(f"Correlation between conf_trough and returns: {corr_trough:.2f}")
print(f"Correlation between conf_peak and returns: {corr_peak:.2f}")
