import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# 1. Download data
tickers = ['^GSPC', '^VIX']
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(years=20)
data = yf.download(tickers, start=start_date, end=end_date)

# data = pd.read_csv('sp_vix_daily.csv', header=[0, 1], index_col=0)
data.index = pd.to_datetime(data.index)

# 2. Feature engineering
# S&P500 daily range
spc = data['High']['^GSPC'].to_frame('High')
spc['Low'] = data['Low']['^GSPC']
spc['Close'] = data['Close']['^GSPC']
spc['RangePct'] = (spc['High'] - spc['Low']) / spc['Close'] * 100
spc['RangePct_30d_MA'] = spc['RangePct'].rolling(30).mean()
# VIX implied move
vix = data['Close']['^VIX'].to_frame('VIX_AnnualPct')
vix['VIX_DailyImpliedPct'] = vix['VIX_AnnualPct'] / np.sqrt(252)
vix['VIX_30d_MA'] = vix['VIX_DailyImpliedPct'].rolling(30).mean()
# Merge
df = spc[['RangePct', 'RangePct_30d_MA']].merge(vix[['VIX_DailyImpliedPct','VIX_30d_MA']], left_index=True, right_index=True)
# Lag and seasonality
df['RangePct_Lag1'] = df['RangePct'].shift(1)
df['DOW_sin'] = np.sin(2*np.pi*df.index.dayofweek/7)
df['DOW_cos'] = np.cos(2*np.pi*df.index.dayofweek/7)
df['DOY_sin'] = np.sin(2*np.pi*df.index.dayofyear/365)
df['DOY_cos'] = np.cos(2*np.pi*df.index.dayofyear/365)
df.dropna(inplace=True)

# 3. Prepare matrices
target = 'RangePct'
features = ['VIX_DailyImpliedPct','VIX_30d_MA','RangePct_Lag1','RangePct_30d_MA','DOW_sin','DOW_cos','DOY_sin','DOY_cos']
X = df[features]
y = df[target]

# 4. Train-test split (last 20% as test)
test_size = int(0.2*len(df))
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# 5. XGBoost with time-series CV
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1]
}
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
gs = GridSearchCV(xgb, param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
gs.fit(X_train, y_train)
best_xgb = gs.best_estimator_
print(f"Best XGBoost params: {gs.best_params_}")

# 6. Metrics function
def report(name, y_true, y_pred):
    print(f"{name} set:")
    print(f"  MAE: {mean_absolute_error(y_true,y_pred):.4f}%")
    print(f"  MSE: {mean_squared_error(y_true,y_pred):.4f}")
    print(f"  R2: {r2_score(y_true,y_pred):.4f}\n")

# 7. Evaluate
report('Training', y_train, best_xgb.predict(X_train))
report('Test', y_test, best_xgb.predict(X_test))

# 8. Full-dataset predictions & save
df['Predicted_RangePct_XGB'] = best_xgb.predict(X)
df[[target,'Predicted_RangePct_XGB',*features]].to_csv('xgb_range_predictions.csv')
print("Saved predictions to xgb_range_predictions.csv")

# 9. Save model
# joblib.dump(best_xgb,'xgb_range_model.pkl')
# print("Saved XGBoost model to xgb_range_model.pkl")
