import pandas as pd
import quandl
import math



df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#forecast will be based on this column
forecast_col = 'Adj. Close'

df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.1 * len(df))) #finding 10% of the days

df['label'] = df[forecast_col].shift(-forecast_out) #shift changes the position of rows down, negative value will take the rows up. Basically we want to see what was the value of adj. close after 10% of the days for the given features.



