#codes used, please test and improve：
#install these in terminal
pip install pandas
pip install prophet 
pip install yfinance==0.1.83 #this version is good for prophet. Newer versions give you an unwanted Ticker row.
stock='3988.HK'
#prepare a dataset
import yfinance as yf
data = yf.download(stock, start = '2022-01-01' )
data.reset_index(inplace=True) 
data.info()
data.dropna()
data.rename(columns={data.columns[0]:'ds'},inplace=True)
data.rename(columns={data.columns[4]:'y'},inplace=True)
df=data[['ds','y']].copy()
df.info()

#Prophet
import pandas as pd
from prophet import Prophet
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(365)

#make a plot
fig1 = m.plot(forecast)
ax=fig1.gca()
ax.set_xlabel(stock,size=14)
fig1.show()

#plot the components
fig2=m.plot_components(forecast)
ax=fig2.gca()
ax.set_xlabel(stock,size=14)
fig2.show()
