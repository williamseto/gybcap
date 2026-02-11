import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# 1) Generate synthetic price series with highs and lows
np.random.seed(42)
n_steps = 500
sinusoid = np.sin(np.linspace(0, 6 * np.pi, n_steps)) * 10
drift = np.linspace(-20, 20, n_steps)
noise = np.random.normal(0, 1, n_steps)
prices = 100 + sinusoid + drift + noise
highs = prices + np.random.uniform(0.5, 2.0, n_steps)
lows = prices - np.random.uniform(0.5, 2.0, n_steps)
df = pd.DataFrame({'price': prices, 'high': highs, 'low': lows})

# 2) Label only global extrema using prominence
peaks, _ = find_peaks(prices, prominence=8.0)
troughs, _ = find_peaks(-prices, prominence=8.0)
df['true_label'] = 0
df.loc[peaks, 'true_label'] = -1
df.loc[troughs, 'true_label'] = 1


# 3) Feature engineering: recent returns + future swing magnitudes
lags = 3
X = []
for i in range(len(df)):
    feats = []
    for lag in range(1, lags + 1):
        j = max(0, i - lag)
        feats.append(prices[i] - prices[j])
    future = prices[i+1:] if i+1 < len(prices) else np.array([prices[i]])
    feats.append(future.max() - prices[i])
    feats.append(prices[i] - future.min())
    X.append(feats)
X = np.array(X)
y = df['true_label'].values

# 4) Train/test split & predictor training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
predictor = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0)
predictor.fit(X_train, y_train)

# 5) Predict on full dataset
y_pred = predictor.predict(X)
df['pred_label'] = y_pred


# 6) Plot price with true and predicted extrema
plt.figure(figsize=(12, 6))
plt.plot(prices, label='Price')
# True extrema
plt.scatter(peaks, prices[peaks], marker='^', color='green', s=100, label='True Peaks')
plt.scatter(troughs, prices[troughs], marker='v', color='red', s=100, label='True Troughs')
# Predicted extrema
pred_peaks = df.index[df['pred_label'] == -1].to_numpy()
pred_troughs = df.index[df['pred_label'] == 1].to_numpy()
plt.scatter(pred_peaks, prices[pred_peaks], marker='x', color='blue', s=60, label='Predicted Peaks')
plt.scatter(pred_troughs, prices[pred_troughs], marker='o', color='orange', s=60, label='Predicted Troughs')

plt.title('Global Extrema: True vs Predicted')
plt.xlabel('Timestep')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()
