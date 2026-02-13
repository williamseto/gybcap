
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, signal
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.fftpack import fft, fftfreq
import pywt
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import *

def hack_euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None, squared=False):
    print("hi")
    return 0

# monkey patch
from sklearn.metrics.pairwise import _euclidean_distances 
old_euclidean_distances = _euclidean_distances

# pd.set_option('display.max_rows', None)

def get_vp_features(ptv_df):
        
    if (len(ptv_df) < 30):
        return False, 0, 0

    volume_profile = ptv_df.groupby("Close").agg({"Volume": "sum"}).reset_index()

    kde = stats.gaussian_kde(volume_profile['Close'], weights=volume_profile['Volume'], bw_method=0.1)
    xr = np.linspace(volume_profile['Close'].min(), volume_profile['Close'].max(), max(1, int(4 * (volume_profile['Close'].max() - volume_profile['Close'].min()))))
    kdy = kde(xr)

    high_volume_threshold = np.percentile(kdy, 75)
    low_volume_threshold = np.percentile(kdy, 25)

    high_peak_candidates_idx, _ = signal.find_peaks(kdy)
    high_peak_candidates_y = kdy[high_peak_candidates_idx]

    high_peak_idx = high_peak_candidates_y >= high_volume_threshold
    high_peak_x = xr[high_peak_candidates_idx][high_peak_idx]

    low_peak_candidates_idx, _ = signal.find_peaks(-kdy)
    low_peak_candidates_y = kdy[low_peak_candidates_idx]

    low_peak_idx = low_peak_candidates_y <= low_volume_threshold
    low_peak_x = xr[low_peak_candidates_idx][low_peak_idx]

    if len(high_peak_x) == 0 or len(low_peak_x) == 0:
        return False, 0, 0


    dist_lvn = min(abs(low_peak_x - ptv_df["Close"].iloc[-1]))
    dist_hvn = min(abs(high_peak_x - ptv_df["Close"].iloc[-1]))

    return True, dist_lvn, dist_hvn

def test_calc(df):

    # num_samples = 500
    # xr = np.linspace(df['Close'].min(), df['Close'].max(), num_samples)
    # kdy = kde(xr)

    # fig = go.Figure()
    # fig.add_trace(go.Histogram(name='Vol Profile', x=df['Close'], y=df['Volume'], nbinsx=100, 
    #                            histfunc='sum', histnorm='probability density',
    #                            marker_color='#B0C4DE'))
    # fig.add_trace(go.Scatter(name='KDE', x=xr, y=kdy, mode='lines', marker_color='#D2691E'))
    # fig.show()

    

    # volume_profile_ba = df.groupby("Close").agg({
    #                     " Bid Volume": "sum",
    #                     " Ask Volume": "sum"
    #                     }).reset_index()
    
    volume_profile = df.groupby("Close").agg({
                        "Volume": "sum"
                        }).reset_index()
    
    normalized_volume = volume_profile["Volume"] / volume_profile["Volume"].sum()
    # entropy = -np.sum(normalized_volume * np.log(normalized_volume + 1e-8))

    # cumulative_vp = volume_profile["Volume"].cumsum()

    # # Normalize to create a percentile scale (0 to 1)
    # cumulative_vp_percentile = cumulative_vp / cumulative_vp.iloc[-1]
    # price_percentiles = np.interp(df["Close"], volume_profile["Close"], cumulative_vp_percentile)

 
    kde = stats.gaussian_kde(volume_profile['Close'], weights=volume_profile['Volume'], bw_method=0.1)
    xr = np.linspace(volume_profile['Close'].min(), volume_profile['Close'].max(), len(volume_profile['Close']))
    kdy = kde(xr)

    high_volume_threshold = np.percentile(kdy, 75)
    low_volume_threshold = np.percentile(kdy, 25)
 
    # min_prom = kdy.max() * 0.1
    # peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom)
    # pkx = xr[peaks]
    # pky = kdy[peaks]

    # high_peak_candidates_idx, _ = signal.find_peaks(kdy)
    # high_peak_candidates_y = kdy[high_peak_candidates_idx]

    # high_peak_idx = high_peak_candidates_y > high_volume_threshold
    # high_peak_x = xr[high_peak_candidates_idx][high_peak_idx]
    # high_peak_y = high_peak_candidates_y[high_peak_idx]

    # low_peak_candidates_idx, _ = signal.find_peaks(-kdy)
    # low_peak_candidates_y = kdy[low_peak_candidates_idx]

    # low_peak_idx = low_peak_candidates_y < low_volume_threshold
    # low_peak_x = xr[low_peak_candidates_idx][low_peak_idx]
    # low_peak_y = low_peak_candidates_y[low_peak_idx]


    low_peak_idxs = kdy <= low_volume_threshold
    low_peak_x = xr[low_peak_idxs]
    low_peak_y = kdy[low_peak_idxs]

    fig = plt.figure()

    plt.bar(volume_profile["Close"], normalized_volume * 4, width=0.25, alpha=0.6, edgecolor='k', label='Volume')
    
    plt.plot(xr, kdy)

    # plt.scatter(high_peak_x, high_peak_y, c='green')

    plt.scatter(low_peak_x, low_peak_y, c='red')

    plt.show()

    # feat_distance_to_lvn = np.min(abs(np.subtract.outer(low_peak_x, df["Close"].to_numpy()).T), axis=1)
    # feat_distance_to_hvn = np.min(abs(np.subtract.outer(high_peak_x, df["Close"].to_numpy()).T), axis=1)


    exit()

    def get_per_min_vp_features(ptv_df):
        
        # is_day_feat_valid, day_dist_lvn, day_dist_hvn = get_vp_features(ptv_df)

        # is_rolling_feat_valid, rolling_dist_lvn, rolling_dist_hvn = get_vp_features(ptv_df.tail(30))

        # return is_day_feat_valid, day_dist_lvn, day_dist_hvn

        return 0, 0, 0




    # vp_results = pd.DataFrame([get_per_min_vp_features(df_) for df_ in df.expanding()])
    # vp_results.columns = ['is_day_feat_valid', 'day_dist_lvn', 'day_dist_hvn']

    vp_results = pd.DataFrame()

    return vp_results


def get_volume_features(df):

    if len(df) < 30:
        return pd.DataFrame(
            {
                'dist_to_lvn' : [0] * len(df),
                'dist_to_hvn' : [0] * len(df),
                'price_percentile' : [0] * len(df)
            })

    volume_profile = df.groupby("Close").agg({"Volume": "sum"}).reset_index()

    kde = stats.gaussian_kde(volume_profile['Close'], weights=volume_profile['Volume'], bw_method=0.1)
    xr = np.linspace(volume_profile['Close'].min(), volume_profile['Close'].max(), max(1, int(4 * (volume_profile['Close'].max() - volume_profile['Close'].min()))))
    kdy = kde(xr)

    high_volume_threshold = np.percentile(kdy, 75)
    low_volume_threshold = np.percentile(kdy, 25)

    high_peak_candidates_idx, _ = signal.find_peaks(kdy)
    high_peak_candidates_y = kdy[high_peak_candidates_idx]

    high_peak_idx = high_peak_candidates_y >= high_volume_threshold
    high_peak_x = xr[high_peak_candidates_idx][high_peak_idx]

    # low_peak_candidates_idx, _ = signal.find_peaks(-kdy)
    # low_peak_candidates_y = kdy[low_peak_candidates_idx]

    # low_peak_idx = low_peak_candidates_y <= low_volume_threshold
    # low_peak_x = xr[low_peak_candidates_idx][low_peak_idx]

    low_peak_x = xr[kdy <= low_volume_threshold]

    dists_to_lvn = np.min(abs(np.subtract.outer(low_peak_x, df["Close"].to_numpy()).T), axis=1)
    dists_to_hvn = np.min(abs(np.subtract.outer(high_peak_x, df["Close"].to_numpy()).T), axis=1)

    cumulative_vp = volume_profile["Volume"].cumsum()
    # Normalize to create a percentile scale (0 to 1)
    cumulative_vp_percentile = cumulative_vp / cumulative_vp.iloc[-1]
    price_percentiles = np.interp(df["Close"], volume_profile["Close"], cumulative_vp_percentile)

    pd.set_option('display.max_rows', None)

    print(volume_profile)


    close_idx = volume_profile[volume_profile['Close'] == 556.0].index

    window_min_idx = max(close_idx[0]-10, 0)
    window_max_idx = min(close_idx[0]+10, len(volume_profile))

    vp_window = volume_profile.iloc[window_min_idx:window_max_idx]

    print(vp_window)


    print(vp_window["Volume"].sum() / volume_profile["Volume"].sum())
    exit()

    return pd.DataFrame(
        {
            'dist_to_lvn' : dists_to_lvn,
            'dist_to_hvn' : dists_to_hvn,
            'price_percentile' : price_percentiles
        })



price_df = pd.read_csv('test_time_trend.csv')
# price_df = price_df.assign(tst=np.nan)

# vp_feat_df = price_df.groupby(['trading_day']).apply(lambda x: test_calc(x))

vol_feat_df = price_df.groupby(['trading_day']).apply(lambda x: get_volume_features(x)).reset_index(drop=True)

price_df_vp_feat = pd.concat([price_df, vol_feat_df], axis=1)
price_df_vp_feat.to_csv('test_time_vp.csv', index=False)

# vp_df = price_df[~price_df['tst'].isna()]


# kmeans = KMeans(n_clusters = 3, n_init='auto')
# kmeans.fit(vp_df['tst'])