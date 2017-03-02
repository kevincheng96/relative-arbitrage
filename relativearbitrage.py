# Beta-neutral Statistical Arbitrage / Divergence

import pandas as pd
from pandas import DataFrame
from datetime import datetime
import numpy as np
import scipy.odr as odr
import pandas_datareader.data as web
import matplotlib.pyplot as plt # matplotlib is version 1.3.1, need to update to 1.4 or higher...

# print nvda.head()
# print np.array(nvda[['Adj Close', 'Open']])
# print pd.DataFrame(np.array(nvda[['Adj Close', 'Open']]))
# print nvda.describe()

# nvda['Adj Close'].plot()
# plt.show()
stock1 = 'HD'
stock2 = 'LOW'
days_moving_avg = 20
start = datetime(2016, 10, 1)
end = datetime(2017, 3, 1)
# need to dynamically calculate hedge ratio based on historical data
# hedge_ratio = 0.67153805 # ratio of stock2 to stock1

def statisticalArb(stock1, stock2, start, end, days_moving_avg):
	df = getStocks(stock1, stock2, start, end, days_moving_avg)
	hedge_ratio = calculateHedgeRatio(df, days_moving_avg)
	print "Hedge Ratio Used: " + str(hedge_ratio)
	df = calculateSpread(df, stock1, stock2, days_moving_avg, hedge_ratio)
	# print df
	# print df[[stock1,stock2]].corr()
	graph(df)

# retrieving and sanitizing data
def getStocks(stock1, stock2, start, end, days_moving_avg):
	s1 = web.DataReader(stock1, 'yahoo', start, end)
	s2 = web.DataReader(stock2, 'yahoo', start, end)	
	s1 = s1.drop(['High','Low','Open','Volume','Adj Close'], axis=1) # drop unnecessary columns
	s1.columns = [stock1] # closing price of NVDA
	s2 = s2.drop(['High','Low','Open','Volume','Adj Close'], axis=1)
	s2.columns = [stock2] # closing price of AMD
	df = pd.concat([s1, s2], axis=1) # concatenates the dataframes
	return df # returns a single dataframe containing the closing prices, moving avg, spread, and lower and upper stdev of stocks

# running Total Least Errors regression to find hedge ratio (using days_moving_avg as the # of historical days to use)
# spread = p1 - b0 * p2 + b1(where p1=price of stock 1, p2=price of stock 2, b=hedge ratio, b1=constant)
# spread - p1 = -b * p2 + b1
# run augmented Dickey-Fuller (ADF) test to see if spreads are stationary
def calculateHedgeRatio(df, days_moving_avg):
	s1list = df[stock1].tolist()[-days_moving_avg:]
	s2list = df[stock2].tolist()[-days_moving_avg:]
	def f(B, x):
		return B[0] * x + B[1]
	print len(s1list)
	print np.std(s1list)
	print np.std(s2list)
	linear = odr.Model(f)
	mydata = odr.RealData(s1list, s2list, sx=np.std(s1list), sy=np.std(s2list))
	myodr = odr.ODR(mydata, linear, beta0=[2, 0])
	myoutput = myodr.run()
	# myoutput.pprint()
	return myoutput.beta[0] # returns the hedge ratio

def calculateSpread(df, stock1, stock2, days_moving_avg, hedge_ratio):
	df['Spread'] = df[stock1] - df[stock2] * hedge_ratio
	df['MovingAvg'] = df['Spread'].rolling(window=days_moving_avg).mean()
	df['Stdev'] = df['Spread'].rolling(window=days_moving_avg).std()
	df['UpperStdev'] = df['MovingAvg'] + 2 * df['Stdev']
	df['LowerStdev'] = df['MovingAvg'] - 2 * df['Stdev']
	return df

# now want to find the spread and the past 60 day standard deviation for spreads
def graph(df):
	df['Spread'].plot(label='Spread', color='g')
	df['MovingAvg'].plot(label='MovingAvg', color='b')
	df['UpperStdev'].plot(label='UpperStdev', color='r', linestyle='dashed')
	df['LowerStdev'].plot(label='LowerStdev', color='r', linestyle='dashed')
	plt.legend(loc='lower right')
	plt.show()

def calculateProfit(df):
	return

statisticalArb(stock1, stock2, start, end, days_moving_avg)

# HOW TO READ THE CHART:
# when spread crosses higher trigger level, sell stock1 and buy stock2.
# when spread crosses lower trigger level, buy stock1 and sell stock2.
# when spread returns to moving avg, get rid of both positions
# want the spread to be at a higher level than when initially bought
# REMEMBER:
# if bought at higher trigger level, want a downward slope between buy spread and sell spread
# if bought at lower trigger level, want a upward slope between buy spread and sell spread 

# Sources
# Total Least Squares for Hedge Ratio: http://quantdevel.com/public/betterHedgeRatios.pdf
# Intro to Statistical Arbitrage: https://www.youtube.com/watch?v=LLgV2Dse2Tc
