import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

ds=r'FB.csv'
df=pd.read_csv(ds, index_col='Date', parse_dates=True)
df=df.rename(columns={'Adj Close' : 'close'})
df=df.loc[:,['Open','High','Low','close']]
print(df.tail(30))

forecast_col='close'
forecast_out= int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
X=np.array(df.drop(['label'], 1))
y=np.array(df['label'])
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)
clf=LinearRegression()
clf.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
accuracy=clf.score(X_test, y_test)

X=X[:-forecast_out]
X_lately=X[-forecast_out:]
Forecast_set=clf.predict(X_lately)


def suggestion(inp):
    print("\n    **Predicted Stocks**")
    for x in range(0,len(inp)-1):
        if((inp[x+1]/inp[x])>1.01):
            print(inp[x+1],'\tBuy')
        elif((inp[x+1]/inp[x])<1):
            print(inp[x+1],'\tSell')
        elif(1<(inp[x+1]/inp[x])<1.01):
            print(inp[x+1],'\tHOLD')
suggestion(Forecast_set)
plt.plot(Forecast_set)

