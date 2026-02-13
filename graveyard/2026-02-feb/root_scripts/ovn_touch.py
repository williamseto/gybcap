from backtesting import Strategy
from backtesting import Backtest
import pandas as pd
import numpy as np
import pdb

def test_calc(df):
    return df.assign(ovn=df[df['ovn'] == 1]['Low'].min())['ovn']

class OvnTouch(Strategy):
    ovn_hi = -9999
    ovn_lo = 9999
    curr_trading_day = 0
    
    def init(self):
        self.data.df['ovn_lo'] = self.data.df.groupby(['trading_day']).apply(lambda x: test_calc(x)).droplevel(0)

        # self.data.df.to_csv('test_ovn_lo.csv', index=False)
        # exit()
    
    def next(self):

        curr_row_num = self.data.df.shape[0]

        curr_time = self.data.df.index[-1]

        try:
            is_early_close = self.data._Data__df.iloc[curr_row_num + 1]['trading_day'] > self.data.trading_day[-1]
        except:
            is_early_close = True

        if self.position:
            # todo: setup trailing stop
            if self.position.pl > 10:
                self.position.close()

            # exit position before close
            if (curr_time.hour == 16 and curr_time.minute > 10) or is_early_close:
                self.position.close()

            return

        if self.data.ovn[-1] == 1 or curr_time.hour >= 16 or is_early_close:
            return
        

        if self.data.Low[-2] <= self.data.ovn_lo[-1] and \
           self.data.Close[-1] > self.data.ovn_lo[-1]:
            self.buy(size=1.0, sl=(self.data.Close[-1] - 5))


price_df = pd.read_csv('test_time_full.csv')

price_df = price_df.assign(DT='NAN')
price_df["DT"] = price_df["Date"] + price_df["Time"]
price_df['DT'] = pd.to_datetime(price_df['DT'])
price_df = price_df.set_index("DT")


bt = Backtest(price_df, OvnTouch, cash=10_000, commission=0.0)
stats = bt.run()
print(stats)
