import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm, cross_validation
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#forecast will be based on this column
forecast_col = 'Adj. Close'

df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.01 * len(df))) #finding 1% of the days

df['label'] = df[forecast_col].shift(-forecast_out) #shift changes the position of rows down, negative value will take the rows up. Basically we want to see what was the value of adj. close after 1% of the days for the given features.


df.dropna(inplace = True)

X = np.array(df.drop(['label'], 1)) #this is the feature column and it consists of everything except the 'label' column.
y = np.array(df['label']) #labels column
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)


clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test) 

print(accuracy)  # when tested it gave 94.5% for me.


