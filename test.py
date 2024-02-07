
import pandas as pd

price_data = pd.read_csv('es_historical_public222.txt')

test_data = price_data.tail(4124)

test_data.to_csv('test.csv')

