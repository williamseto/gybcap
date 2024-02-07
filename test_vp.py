
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, signal
from sklearn.cluster import KMeans

def hack_euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None, squared=False):
    print("hi")
    return 0

# monkey patch
from sklearn.metrics.pairwise import _euclidean_distances 
old_euclidean_distances = _euclidean_distances


def test_calc(df):

    print(df['High'].max(), df['Low'].min())

    # px.histogram(df, x='Volume', y='Close', nbins=100, orientation='h').show()

    kde = stats.gaussian_kde(df['Close'], weights=df['Volume'], bw_method=0.1)

    df.iloc[0, df.columns.get_loc('tst')] = kde
    return df
    exit()

    num_samples = 500
    xr = np.linspace(df['Close'].min(), df['Close'].max(), num_samples)
    kdy = kde(xr)
    ticks_per_sample = (xr.max() - xr.min()) / num_samples

    fig = go.Figure()
    fig.add_trace(go.Histogram(name='Vol Profile', x=df['Close'], y=df['Volume'], nbinsx=100, 
                               histfunc='sum', histnorm='probability density',
                               marker_color='#B0C4DE'))
    fig.add_trace(go.Scatter(name='KDE', x=xr, y=kdy, mode='lines', marker_color='#D2691E'))
    fig.show()

    exit()



price_df = pd.read_csv('test_time.csv')
price_df = price_df.assign(tst=np.nan)

price_df = price_df.groupby(['trading_day']).apply(lambda x: test_calc(x))

# print(price_df)

vp_df = price_df[~price_df['tst'].isna()]


kmeans = KMeans(n_clusters = 3, n_init='auto')
kmeans.fit(vp_df['tst'])