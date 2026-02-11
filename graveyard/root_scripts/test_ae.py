
import pandas as pd

from datetime import datetime, timedelta


price_data_df = pd.read_csv('test_seconds.csv')
price_data_df = price_data_df.drop(columns=['Unnamed'])


# dt_format_str = "%Y/%m/%d %H:%M:%S.%f"

# new_price_df['dt'] = new_price_df.apply(lambda row: datetime.strptime(f"{row['Date']}{row['Time']}", dt_format_str), axis=1)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


# df = pd.DataFrame({
#     "Best_Bid": np.random.uniform(100, 200, 1000),
#     "Best_Ask": np.random.uniform(100, 200, 1000) + np.random.uniform(0, 1, 1000),  # Ask > Bid
#     "Bid_Volume": np.random.randint(100, 5000, 1000),
#     "Ask_Volume": np.random.randint(100, 5000, 1000),
#     "Mid_Price": (np.random.uniform(100, 200, 1000) + np.random.uniform(100, 200, 1000)) / 2,
#     "Trade_Volume": np.random.randint(100, 10000, 1000)
# })

# Compute features
price_data_df["OFI"] = price_data_df["Bid_Volume"] - price_data_df["Ask_Volume"]  # Order Flow Imbalance
price_data_df["Depth_Ratio"] = price_data_df["Bid_Volume"] / (price_data_df["Bid_Volume"] + price_data_df["Ask_Volume"])
price_data_df["Price_Change_Rate"] = price_data_df["Close"].pct_change().rolling(window=60).std()
# price_data_df["Spread"] = price_data_df["Best_Ask"] - price_data_df["Best_Bid"]  # Bid-Ask Spread


price_data_df.dropna(inplace=True)

# Normalize features
scaler = MinMaxScaler()
features = ["OFI", "Depth_Ratio", "Price_Change_Rate"]
df_scaled = scaler.fit_transform(price_data_df[features])

# df_scaled = price_data_df


# Convert to tensors
X_train = torch.tensor(df_scaled, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True)

# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)  # Bottleneck (compressed latent space)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)  # Reconstruct original input
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model
input_dim = len(features)
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train autoencoder
num_epochs = 50
for epoch in range(num_epochs):
    for batch in train_loader:
        x_batch = batch[0]
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, x_batch)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Compute reconstruction error on entire dataset
X_reconstructed = model(X_train).detach().numpy()
reconstruction_error = np.mean((df_scaled - X_reconstructed) ** 2, axis=1)

# Identify high-error points as absorption zones
threshold = np.percentile(reconstruction_error, 95)  # Top 5% anomalies
price_data_df["Absorption_Predicted"] = reconstruction_error > threshold

result_df = price_data_df.loc[price_data_df['Absorption_Predicted'] == True]
print(result_df[["OFI", "Depth_Ratio", "Price_Change_Rate", "Absorption_Predicted"]].tail(10))
