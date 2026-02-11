import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Download data for S&P 500 and VIX
# tickers = ['^GSPC', '^VIX']
# data = yf.download(tickers, start=pd.Timestamp.today() - pd.DateOffset(years=20), end=pd.Timestamp.today())

# data.to_csv('sp_vix_daily.csv', index=True)

data = pd.read_csv('sp_vix_daily.csv', header=[0, 1], index_col=0)
data.index = pd.to_datetime(data.index)

# 2. Compute S&P 500 daily range percentage
gspc = data['High']['^GSPC'].to_frame('High')
gspc['Low'] = data['Low']['^GSPC']
gspc['Close'] = data['Close']['^GSPC']
gspc['RangePct'] = (gspc['High'] - gspc['Low']) / gspc['Close'] * 100
# 30-day moving average of S&P 500 range
gspc['RangePct_30d_MA'] = gspc['RangePct'].rolling(window=30).mean()

# 3. Compute VIX implied daily move (annualized VIX in % -> daily %) 
vix = data['Close']['^VIX'].to_frame('VIX_AnnualPct')
vix['VIX_DailyImpliedPct'] = vix['VIX_AnnualPct'] / np.sqrt(252)
# 30-day moving average of VIX implied daily move
vix['VIX_30d_MA'] = vix['VIX_DailyImpliedPct'].rolling(window=30).mean()

# 4. Merge datasets
df = gspc[['RangePct', 'RangePct_30d_MA']].merge(
    vix[['VIX_DailyImpliedPct', 'VIX_30d_MA']],
    left_index=True, right_index=True
)
# Add lagged range (previous day)
df['RangePct_Lag1'] = df['RangePct'].shift(1)


# 5. Add seasonality features
# Day of week (0=Monday) as cyclical features
df['DayOfWeek'] = df.index.dayofweek
df['DOW_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DOW_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
# Day of year as cyclical features
df['DayOfYear'] = df.index.dayofyear
df['DOY_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['DOY_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

# Drop any rows with NaN (from rolling and lag)
df.dropna(inplace=True)


# 5. Define features and target
y = df['RangePct']
X = df[['VIX_DailyImpliedPct', 'VIX_30d_MA', 'RangePct_Lag1', 'RangePct_30d_MA', 'DOW_sin', 'DOW_cos', 'DOY_sin', 'DOY_cos']]


# 6. Train-test split (no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 7. Fit a simple linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Evaluate performance on training set
y_train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
print("Training set:")
print(f"  MAE: {train_mae:.4f}%")
print(f"  MSE: {train_mse:.4f}")
print(f"  R^2: {train_r2:.4f}\n")

# 9. Evaluate performance on test set
y_test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print("Test set:")
print(f"  MAE: {test_mae:.4f}%")
print(f"  MSE: {test_mse:.4f}")
print(f"  R^2: {test_r2:.4f}\n")

exit()

# 10. Estimate next-day range using latest values
latest_features = np.array([
    vix['VIX_DailyImpliedPct'].iloc[-1],
    vix['VIX_30d_MA'].iloc[-1],
    df['RangePct'].iloc[-1],
    df['RangePct_30d_MA'].iloc[-1],
    df['DOW_sin'].iloc[-1],
    df['DOW_cos'].iloc[-1],
    df['DOY_sin'].iloc[-1],
    df['DOY_cos'].iloc[-1]
]).reshape(1, -1)
estimate_pct = model.predict(latest_features)[0]
print(f"Estimated next-day range (pct): {estimate_pct:.4f}%")

joblib.dump(model, 'linear_range_model.joblib')

# Generate and save predictions for all dates
# Use the trained model to predict for the entire dataset
df['Predicted_RangePct'] = model.predict(X)
# Save predictions with actuals and features for lookup
output_cols = ['Predicted_RangePct']
df[output_cols].to_csv('range_predictions.csv')
print("All predictions saved to range_predictions.csv")