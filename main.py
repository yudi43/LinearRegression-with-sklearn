import pandas as pd
import quandl


df = quandl.get('WIKI/GOOGL')


df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100



print(df.head(10))





