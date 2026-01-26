import socket
import threading
import queue
import time
import random
import argparse
import sqlite3
import datetime
import zoneinfo
import pandas as pd
import torch
import torch.nn.functional as F
from collections import deque
from sqlalchemy import create_engine

from strategy_util import *

import sys
sys.path.append("../sandbox")
from pred_util import *

sys.path.append("../gex")
from gex_provider import RealtimeGEXProvider

HOST = '172.30.144.1'
PORT = 65432

# Shared queue for data between threads
stock_data_queue = queue.Queue()


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS price_data (
    symbol TEXT,
    timestamp INTEGER,
    price   REAL,
    volume  INTEGER DEFAULT 0,
    buys INTEGER DEFAULT 0,
    sells INTEGER DEFAULT 0,
    PRIMARY KEY(symbol, timestamp)
)
"""

DB_PATH = 'price_data.db'

def init_db(path: str):
    """
    Initialize the SQLite database: enable WAL and create the ticks table.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    # Enable Write-Ahead Logging for safe concurrent readers/writers
    conn.execute("PRAGMA journal_mode=WAL;")
    # Create schema
    conn.execute(DB_SCHEMA)
    conn.commit()
    conn.close()
    print(f"Initialized database at '{path}' (WAL mode, schema created).")

def insert_price_data(conn, price_data):
    symbol = 'ES'
    timestamp = price_data[0]
    price = price_data[2]
    volume = price_data[3]
    buys = price_data[4]
    sells = price_data[5]

    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO price_data (symbol, timestamp, price, volume, buys, sells) VALUES (?, ?, ?, ?, ?, ?)",
            (symbol, timestamp, price, volume, buys, sells)
        )

# Thread: read from TCP socket
def socket_reader(sock):

    buffer = ""
    while True:
        try:
            data = sock.recv(1024).decode('utf-8')
            if not data:
                break
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                # timestamp, price = map(float, line.strip().split(','))
                # stock_data_queue.put({'timestamp': timestamp, 'price': float(price)})

                sock_data = list(map(float, line.strip().split(',')))
                print(f"[Socket Reader] Received: {sock_data}")

        except Exception as e:
            print(f"[Socket Reader] Error: {e}")
            break

# Thread: make trading decisions
def trading_bot():
    while True:
        try:
            data = stock_data_queue.get(timeout=1000)
            price = data['price']
            decision = "BUY" if price < 150 else "HOLD"
            print(f"[Trading Bot] Price: {price} -> Decision: {decision}")
            # time.sleep(random.uniform(1.0, 3.0))  # simulate variable decision time
        except queue.Empty:
            print("[Trading Bot] No new data. Exiting.")
            break

def get_trading_day_start_ts(now_ts: int) -> int:
    """
    Returns the UTC timestamp of the current trading day's start,
    defined as 15:00 America/Los_Angeles on the day containing now_ts
    (or the previous calendar day if now < 15:00 PST).
    """
    la = zoneinfo.ZoneInfo("America/Los_Angeles")
    # localize now
    dt_local = datetime.datetime.fromtimestamp(now_ts, la)
    # decide which calendar date's 15:00 marks the start
    if dt_local.hour >= 15:
        day_start_local = dt_local.replace(hour=15, minute=0, second=0, microsecond=0)
    else:
        prev = dt_local.date() - datetime.timedelta(days=1)
        day_start_local = datetime.datetime(prev.year, prev.month, prev.day,
                                    15, 0, 0, tzinfo=la)
        
    print(day_start_local)
    # convert back to UTC timestamp
    return int(day_start_local.astimezone(datetime.timezone.utc).timestamp())


def get_pred_range_pct(day: datetime.date) -> float:
    range_filename = '../sandbox/range_predictions.csv'
    range_df = pd.read_csv(range_filename)
    range_df['Date'] = pd.to_datetime(range_df['Date'])

    try:
        pred_range_pct = range_df[range_df['Date'].dt.date==day]['Predicted_RangePct'].iloc[0] / 100.0
    except:
        # If no range data available for this day, use a default value
        pred_range_pct = 0.005

    return pred_range_pct


def compute_features(data):

    agg_dict = {
        'price': 'mean',
        'buys': 'sum',
        'sells': 'sum',
        'volume': 'sum',
    }

    data['dt'] = pd.to_datetime(data['timestamp'], unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    min_df = data.set_index('dt').resample('1Min').agg(agg_dict).bfill().reset_index()


    min_df['ret_1m'] = min_df['price'].pct_change(1).fillna(0)
    min_df['ret_5m'] = min_df['price'].pct_change(5).fillna(0)
    min_df['vol_10m'] = (min_df['ret_1m']
                            .rolling(10).std()
                            .reset_index(level=0, drop=True)
                            .fillna(0))
    min_df['min_to_close'] = min_df['dt'].transform(lambda x: (x.max() - x).dt.total_seconds()/60)
        
    min_df['dist_to_max'] = min_df['price'].transform(lambda x: x.max() - x)
    min_df['dist_to_min'] = min_df['price'].transform(lambda x: x - x.min())

    min_df['delta_m'] = (min_df['buys'] - min_df['sells'])
    min_df['cum_delta_m'] = min_df['delta_m'].cumsum()

    min_df['vol_20m_avg'] = min_df['volume'].transform(lambda x: x.rolling(20).mean()).fillna(1)

    min_df['buy_vol_pct'] = min_df['buys'] / min_df['vol_20m_avg']
    min_df['sell_vol_pct'] = min_df['sells'] / min_df['vol_20m_avg']

    min_df['curr_lod'] = min_df['price'].cummin()
    min_df['curr_hod'] = min_df['price'].cummax()

    min_df['dist_to_lod'] = min_df['price'] - min_df['curr_lod']
    min_df['dist_to_hod'] = min_df['curr_hod'] - min_df['price']

    pred_range_pct = get_pred_range_pct(min_df['dt'].dt.date.iloc[-1])
    pred_range_high = min_df['price'].iloc[0] * (1 + pred_range_pct * 0.5)
    pred_range_low = min_df['price'].iloc[0] * (1 - pred_range_pct * 0.5)

    min_df['dist_to_max'] = (pred_range_high - min_df['price']) / min_df['price']
    min_df['dist_to_min'] = (min_df['price'] - pred_range_low) / min_df['price']

    min_df['ml_pred_rev'] = 0  # Placeholder for ML model predictions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ml_rev_model = TwoStreamReversalModel(device)

    ml_rev_model.load_model('../sandbox/twostream_mlp_model_fast.pth')

    ml_rev_model.mlp_model.eval()

    norm_feature_cols = ['price_z', 'vol_z', 'buys_z', 'sells_z']
    N_SEC = TwoStreamReversalDatasetVec.L_sec
    N_MIN = TwoStreamReversalDatasetVec.L_min

    sec_feat_window = TwoStreamReversalDatasetVec.precompute_norms(data.tail(N_SEC).copy(), N_SEC)[norm_feature_cols].to_numpy().transpose()
    min_feat_window = TwoStreamReversalDatasetVec.precompute_norms(min_df.tail(N_MIN).copy(), N_MIN)[norm_feature_cols].to_numpy().transpose()


    # Disable gradient tracking during inference
    with torch.no_grad():

        x_loc = torch.tensor(sec_feat_window, dtype=torch.float32).unsqueeze(0)  # shape: (1, num_features, N_SEC)
        x_glo = torch.tensor(min_feat_window, dtype=torch.float32).unsqueeze(0)  # shape: (1, num_features, N_MIN)

        x_loc_dev, x_glo_dev = x_loc.to(device), x_glo.to(device)
        logits  = ml_rev_model.mlp_model(x_loc_dev, x_glo_dev)
        probs  = F.softmax(logits, dim=1)[:,1].cpu()
        print(f"[ML Model] Predicted probabilities: {probs}")

    exit()

    return min_df

def test_func():
    print("Running test function...")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # get current day price data
    now_ts = int(time.time()) - (3600 * 72)

    trading_day_start_ts = get_trading_day_start_ts(now_ts)

    curr_day_price_data = pd.read_sql_query(
        "SELECT * FROM price_data "
        "WHERE symbol = ? AND timestamp >= ? "
        "ORDER BY timestamp ASC",
        conn,
        params=('ES', trading_day_start_ts)
    )

    # first_dt = datetime.datetime.fromtimestamp(curr_day_price_data['timestamp'].iloc[0], zoneinfo.ZoneInfo("America/Los_Angeles"))
    # print(first_dt)

    curr_day_price_data = compute_features(curr_day_price_data)

    curr_row = curr_day_price_data.tail(1)
    curr_row.rename(columns={'volume': 'Volume'}, inplace=True)
    curr_row['y_rev'] = 0  # Placeholder for reversal labels

    exit()

    rev_model = ReversalModel(curr_day_price_data)
    rev_model.bst = xgb.Booster()
    rev_model.bst.load_model('../sandbox/rev_model.json')

    prob_rev, contribs = rev_model.predict(curr_row)
    pred_rev = (prob_rev>=0.3).astype(int)

    print(prob_rev, pred_rev)

    conn.close()


import requests
def send_discord_message(content: str):
    DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1379637698029162577/gQ5pLAy111DYY6bM4Uce6w_FaloZo2cwK6qh89NrT1b1-JiO1QkVeIjbvumvC_JLUkgT"
    payload = {"content": content}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, headers=headers)
    if not resp.ok:
        print(f"Failed to send webhook: {resp.status_code} {resp.text}")

class RealtimeEngine:
    def __init__(self, db_path, max_window_sec=120):

        self.conn = create_engine(
            f"mysql+mysqlconnector://kibblesoup:kibblesoup@{HOST}:3306/sys"
        )

        # Store only last max_window_sec seconds for rolling
        self.raw_deque = deque(maxlen=max_window_sec)
        self.min_df = pd.DataFrame()
        self.last_ts = None

        self.last_trigger_dt = None

        self.bo_strat = BreakoutRetestStrategy(level_cols=['prev_high','prev_low','vwap', 'ovn_lo', 'ovn_hi', 'ib_lo', 'ib_hi'],
                            threshold_pct=0.0012, lookahead_bars=12)
        
        self.revert_strat = ReversionStrategy(level_cols=['prev_high','prev_low','vwap', 'ovn_lo', 'ovn_hi', 'ib_lo', 'ib_hi'],
                    threshold_pct=0.0012, lookahead_bars=12)

        self.gex_provider = RealtimeGEXProvider()

        return

        self.rev_model = ReversalModel(self.min_df)
        self.rev_model.bst = xgb.Booster()
        self.rev_model.bst.load_model('../sandbox/rev_model.json')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ml_rev_model = TwoStreamReversalModel(self.device)

        self.ml_rev_model.load_model('../sandbox/twostream_mlp_model_fast.pth')

        self.ml_rev_model.mlp_model.eval()



    def initialize(self, now_ts: int = None):

        if now_ts is None:
            now_ts = int(time.time())

        trading_day_start_ts = get_trading_day_start_ts(now_ts)

        self.trading_day_end_ts = trading_day_start_ts + 22 * 3600

        sec_df = pd.read_sql_query(
            "SELECT * FROM price_data "
            "WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s "
            "ORDER BY timestamp ASC",
            self.conn,
            params=('ES', trading_day_start_ts, now_ts)
        )

        min_df = self._build_minute_df(sec_df)

        pred_range_pct = get_pred_range_pct(min_df.index.date[-1])
        self.pred_range_high = min_df['price'].iloc[0] * (1 + pred_range_pct * 0.5)
        self.pred_range_low = min_df['price'].iloc[0] * (1 - pred_range_pct * 0.5)
        
        self.raw_deque.extend(sec_df.to_dict('records'))
        self.last_ts = int(sec_df['timestamp'].max()) if not sec_df.empty else trading_day_start_ts

        self.min_df = min_df

        # Initialize GEX provider
        try:
            self.gex_provider.fetch_chain()

            rth_df = self.min_df.between_time('6:30', '12:59')

            self.gex_provider.spx_es_spread = self.gex_provider.spot_price - rth_df['open'].iloc[0]

            self.min_df['nearby_gamma_score'] = 0.0
            self.min_df.loc[rth_df.index, 'nearby_gamma_score'] = self.gex_provider.compute_gex_score_batch_mapped(rth_df)


        except Exception as e:
            print(f"[RealtimeEngine] GEX initialization warning: {e}")

    def _build_minute_df(self, sec_df):        
        sec_df['dt'] = pd.to_datetime(sec_df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
        
        sec_df.set_index('dt', inplace=True)
        ohlc = sec_df["price"].resample("1Min").ohlc().ffill()


        pd.set_option('display.max_rows', None)


        agg_dict = {
            'price': 'mean',
            'buys': 'sum',
            'sells': 'sum',
            'volume': 'sum',
        }
        min_df = sec_df.resample('1Min').agg(agg_dict).bfill()

        min_df = min_df.join(ohlc)

        return min_df


    def _compute_features(self, min_df):
        # m['vwap'] = (m['last']*m['volume']).cumsum() / m['volume'].cumsum()


        min_df['ret_1m'] = min_df['price'].pct_change(1).fillna(0)
        min_df['ret_5m'] = min_df['price'].pct_change(5).fillna(0)
        min_df['vol_10m'] = (min_df['ret_1m']
                                .rolling(10).std()
                                .fillna(0))

        end_of_day = datetime.datetime.fromtimestamp(self.trading_day_end_ts, zoneinfo.ZoneInfo('America/Los_Angeles'))
        # min_df['min_to_close'] = min_df.index.transform(lambda x: (x.max() - x).total_seconds()/60)
        min_df['min_to_close'] = (end_of_day - min_df.index).total_seconds() / 60
            
        min_df['dist_to_max'] = min_df['price'].transform(lambda x: x.max() - x)
        min_df['dist_to_min'] = min_df['price'].transform(lambda x: x - x.min())

        min_df['delta_m'] = (min_df['buys'] - min_df['sells'])
        min_df['cum_delta_m'] = min_df['delta_m'].cumsum()

        min_df['vol_20m_avg'] = min_df['volume'].transform(lambda x: x.rolling(20).mean()).fillna(1)

        min_df['buy_vol_pct'] = min_df['buys'] / min_df['vol_20m_avg']
        min_df['sell_vol_pct'] = min_df['sells'] / min_df['vol_20m_avg']

        min_df['curr_lod'] = min_df['price'].cummin()
        min_df['curr_hod'] = min_df['price'].cummax()

        min_df['dist_to_lod'] = min_df['price'] - min_df['curr_lod']
        min_df['dist_to_hod'] = min_df['curr_hod'] - min_df['price']

        min_df['dist_to_max'] = (self.pred_range_high - min_df['price']) / min_df['price']
        min_df['dist_to_min'] = (min_df['price'] - self.pred_range_low) / min_df['price']

        min_df['ml_pred_rev'] = 0  # Placeholder for ML model predictions

        norm_feature_cols = ['price_z', 'vol_z', 'buys_z', 'sells_z']
        N_SEC = TwoStreamReversalDatasetVec.L_sec
        N_MIN = TwoStreamReversalDatasetVec.L_min

        sec_window = pd.DataFrame(list(self.raw_deque)[N_SEC:])
        
        sec_feat_window = TwoStreamReversalDatasetVec.precompute_norms(sec_window, N_SEC)[norm_feature_cols].to_numpy().transpose()
        min_feat_window = TwoStreamReversalDatasetVec.precompute_norms(min_df.tail(N_MIN).copy(), N_MIN)[norm_feature_cols].to_numpy().transpose()


        # Disable gradient tracking during inference
        with torch.no_grad():

            x_loc = torch.tensor(sec_feat_window, dtype=torch.float32).unsqueeze(0)  # shape: (1, num_features, N_SEC)
            x_glo = torch.tensor(min_feat_window, dtype=torch.float32).unsqueeze(0)  # shape: (1, num_features, N_MIN)

            x_loc_dev, x_glo_dev = x_loc.to(self.device), x_glo.to(self.device)
            logits  = self.ml_rev_model.mlp_model(x_loc_dev, x_glo_dev)
            probs  = F.softmax(logits, dim=1)[:,1].cpu()

            min_df['ml_pred_rev'] = probs.item()

    def update(self, now_ts: int):
    
        # 1) fetch only the *new* rows
        df_new = pd.read_sql_query(
            "SELECT * FROM price_data "
            "WHERE symbol = %s AND timestamp > %s AND timestamp <= %s "
            "ORDER BY timestamp ASC",
            self.conn,
            params=('ES', self.last_ts, now_ts)
        )

        if df_new.empty:
            # print(f"[RealtimeEngine] No new data since last timestamp {self.last_ts}. {now_ts}")
            return

        # 2) append to deque (automatically drops old if beyond window)
        for rec in df_new.to_dict('records'):
            self.raw_deque.append(rec)
        self.last_ts = int(df_new['timestamp'].max())


        deque_df = pd.DataFrame(self.raw_deque)
        deque_min = self._build_minute_df(deque_df)

    
        # 3) aggregate *only* the newly completed minutes
        tmp = pd.DataFrame(df_new)
        tmp['dt'] = pd.to_datetime(tmp['timestamp'], unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
        tmp.set_index('dt', inplace=True)
        new_min = tmp.resample('1Min').agg({
            'price': 'last',
            'buys': 'sum',
            'sells': 'sum',
            'volume': 'sum'
        })

        new_min_df = deque_min.loc[new_min.index].copy()

        # 4) merge into minute_df (overwrite or append)
        self.min_df.update(new_min_df)

        new_idx = new_min.index.difference(self.min_df.index)

        if len(new_idx):
            new_min_df['nearby_gamma_score'] = self.gex_provider.compute_gex_score_batch_mapped(new_min_df)
            self.min_df = pd.concat([self.min_df, new_min_df])
    
        bo_signals = self.bo_strat.find_retest_and_build_trades(self.min_df)
        bo_ts = bo_signals[0]['entry_ts'] if len(bo_signals) else None

        triggered = False
        trigger_ts = None
        content_str = ""
        if len(bo_signals) > 0 and (self.last_trigger_dt is None or bo_ts > self.last_trigger_dt):
            content_str += f"BO triggered at"
            for signal in bo_signals:
                content_str += f" {signal}"
                # print(signal)

            content_str += f"\n"
            # send_discord_message(content_str)

            triggered = True
            trigger_ts = bo_ts

            # print(f"[RealtimeEngine] Breakout-retest triggered at {bo_ts}")


        rev_signals = self.revert_strat.find_retest_and_build_trades(self.min_df)
        rev_ts = rev_signals[0]['entry_ts'] if len(rev_signals) else None

        if len(rev_signals) > 0 and (self.last_trigger_dt is None or rev_ts > self.last_trigger_dt):
            content_str += f"Rev triggered at"
            for signal in rev_signals:
                content_str += f" {signal}"
                # print(signal)

            content_str += f"\n"
            # send_discord_message(content_str)

            triggered = True
            trigger_ts = rev_ts

            # print(f"[RealtimeEngine] Reversion triggered at {rev_ts}")

        if triggered:
            self.last_trigger_dt = trigger_ts

            if len(content_str) > 2000:
                content_str = content_str[:2000]
            send_discord_message(content_str)
            # print(content_str)
            # exit()

        return
    
        # run any/all strategies

        # recompute features on affected window
        self._compute_features(self.min_df)


        curr_feature_row = self.min_df.tail(1).copy()
        curr_feature_row.rename(columns={'volume': 'Volume'}, inplace=True)
        curr_feature_row['y_rev'] = 0  # Placeholder for reversal labels


        prob_rev, contribs = self.rev_model.predict(curr_feature_row)
        pred_rev = (prob_rev>=0.3).astype(int)

        # print(f"[RealtimeEngine] Predicted reversal: {pred_rev}, Probability: {prob_rev}")

        if pred_rev[0] or prob_rev[0] >= 0.5 or curr_feature_row['ml_pred_rev'].values[0] >= 0.7:
            if self.last_trigger_dt is None or curr_feature_row.index > self.last_trigger_dt:
                content_str = f"Reversal triggered at {curr_feature_row.to_string()}\n"
                send_discord_message(content_str)
                self.last_trigger_dt = curr_feature_row.index

                print(content_str)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SQlite-backed Client: ingest timestamped price data over a socket"
    )
    parser.add_argument(
        "--init-db",
        dest="init_db",
        action="store_true",
        help="Initialize the SQLite database",
    )
    parser.add_argument(
        "--db-path", default="/mnt/c/Users/william/Downloads/price_data.db",
        help="Path to the SQLite .db file (default: price_data.db)"
    )

    args = parser.parse_args()

    DB_PATH = args.db_path

    if args.init_db:
        init_db(args.db_path)


    trading_engine = RealtimeEngine(DB_PATH)

    now_ts = int(time.time())
    # now_ts = 1769699083
    
    trading_engine.initialize(now_ts)

    while True:
        # now_ts += 60
        # print(f"Updating to {datetime.datetime.fromtimestamp(now_ts, zoneinfo.ZoneInfo("America/Los_Angeles"))}")
        
        time.sleep(5.0)
        now_ts = int(time.time())

        trading_engine.update(now_ts)

        if now_ts >= trading_engine.trading_day_end_ts:
            print("Trading day ended.")
            break

