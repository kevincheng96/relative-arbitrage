# Beta-neutral Statistical Arbitrage / Pair Trading
# Calculating spread using Moving Average and Rolling Standard Deviation

import pandas as pd
from pandas import DataFrame
from datetime import datetime
import numpy as np
import scipy.odr as odr
import pandas_datareader.data as web
import statsmodels
import statsmodels.tsa.stattools as tsa # test for cointegration between stocks (do the)
import matplotlib.pyplot as plt # matplotlib is version 1.3.1, need to update to 1.4 or higher...
from pandas.stats.api import ols
import arbstrategy # importing the arbitrage strategy I wrote in another file

# Good stock pairs (ranked):
# HD and LOW
# AREX and WLL
# MCD and SBUX

stock1 = 'AMX'
stock2 = 'NEE'
days_moving_avg = 60
start = datetime(2016, 4, 1)
end = datetime(2017, 6, 1)
capital_limit = 1000
hedge_method = 'Rolling'
tickers = ['CHL','VZ','T','VOD','NTT','AMX','CHA','BT','CHU','ORAN','BCE','S','NEE','SKM']
# use polyfit or total least squares?

############
## To Do: ##
############
# Calculate hedge-ratio for each day (ratio dynamically changes) (use both Kalman Filter and rolling)
# Recalculate spread based on current hedge-ratio? ***
# Implement findCointegratedStocks()
# Find endpoints for cryptocurrency historical data


# given a list of stocks, find the cointegrated pairs
def findCointegratedPairs(tickers, start, end, days_moving_avg):
	n = len(tickers)
	cointegrated_pairs = []
	for i in range(n):
		for j in range(i+1, n):
			stock1 = tickers[i]
			stock2 = tickers[j]
			df = getStocks(stock1, stock2, start, end, days_moving_avg)
			df = calculateHedgeRatio(df, stock1, stock2, days_moving_avg, 'Rolling')
			df = calculateSpread(df, stock1, stock2, days_moving_avg)
			coint_pvalue = round(tsa.coint(df[stock1], df[stock2])[1], 5)
			cadf_pvalue = round(tsa.adfuller(df["Spread"][days_moving_avg - 1:])[1], 5)
			if coint_pvalue <= 0.05 and cadf_pvalue <= 0.05:
				cointegrated_pairs.append((stock1, stock2, coint_pvalue, cadf_pvalue))
	print cointegrated_pairs
	return cointegrated_pairs

def statisticalArb(cash, stock1, stock2, start, end, days_moving_avg, hedge_method):
	df = getStocks(stock1, stock2, start, end, days_moving_avg)
	df = calculateHedgeRatio(df, stock1, stock2, days_moving_avg, hedge_method)
	# print "Hedge Ratio Used: " + str(hedge_ratio)
	df = calculateSpread(df, stock1, stock2, days_moving_avg)
	getStatistics(df, hedge_method)
	Strategy = arbstrategy.ArbStrategy(capital_limit, df, stock1, stock2, days_moving_avg)
	Strategy.run()
	graph(df)
	# print df['HedgeRatio'].tolist()
	# print df['2016-09-01':'2016-11-01']
	staticHedge(df[:'2016-09-19 00:00:00'], days_moving_avg) # DEBUGGING
	plt.plot(df.index, df['HedgeRatio'], label='Hedge Ratio') 
	plt.show()

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
def calculateHedgeRatio(df, stock1, stock2, days_moving_avg, method):
	if method == 'Static': # initial (but flawed) method of keeping hedge ratio the same (uses future data)
		return staticHedge(df, stock1, stock2, days_moving_avg)
	elif method == 'Rolling': # calculates hedge ratio using a rolling method based on days_moving_avg
		return rollingHedge(df, stock1, stock2, days_moving_avg)
	elif method == 'Kalman':
		return kalmanHedge(df, stock1, stock2, days_moving_avg)
	else:
		raise Exception('Unknown hedge ratio calculation method thrown!')

	# y = np.asarray(df[stock1].tolist()[-days_moving_avg:]) # stock 1 data
	# x = np.asarray(df[stock2].tolist()[-days_moving_avg:]) # stock 2 data
	# # Fit the data using scipy.odr
	# def f(B, x):
	# 	return B[0] * x + B[1]
	# linear = odr.Model(f)
	# mydata = odr.RealData(x, y, sx=np.std(y), sy=np.std(x))
	# myodr = odr.ODR(mydata, linear, beta0=[2, 0])
	# myoutput = myodr.run()
	# # fit the data using numpy.polyfit
	# fit_np = np.polyfit(x, y, 1)
	# # graph to compare the fit
	# print 'polyfit beta', fit_np[0]
	# print 'least errors beta', myoutput.beta[0]
	# plt.plot(x, y, label='Actual Data', linestyle='dotted') # actual data points
	# plt.plot(x, np.polyval(fit_np, x), "r--", lw = 2, label='Polyfit') # polyfit regression line
	# plt.plot(x, f(myoutput.beta, x), "g--", lw = 2, label='Least Errors') # least errors regression line
	# plt.legend(loc='lower right')
	# plt.show()
	# # myoutput.pprint()
	# # df["HedgeRatio"] = calculateHedgeRatio(df[])
	# return myoutput.beta[0] # returns the hedge ratio

# use np.polyfit to find the hedge ratio
def polyfitHedgeRatio(df, stock1, stock2, days_moving_avg):
	y = np.asarray(df[stock1].tolist()[-days_moving_avg:]) # stock 1 data
	x = np.asarray(df[stock2].tolist()[-days_moving_avg:]) # stock 2 data
	fit_np = np.polyfit(x, y, 1)
	return fit_np[0]

# use total least errors to find the hedge ratio
def leastErrorsHedgeRatio(df, stock1, stock2, days_moving_avg):
	y = np.asarray(df[stock1].tolist()[-days_moving_avg:]) # stock 1 data
	x = np.asarray(df[stock2].tolist()[-days_moving_avg:]) # stock 2 data
	# First use polyfit to generate rough estimate of betas
	fit_np = np.polyfit(x, y, 1)
	# Fit the data using scipy.odr
	def f(B, x):
		return B[0] * x + B[1]
	linear = odr.Model(f)
	mydata = odr.RealData(x, y, sx=np.std(x), sy=np.std(y))
	myodr = odr.ODR(mydata, linear, beta0=[fit_np[0], fit_np[1]])
	myoutput = myodr.run()
	return myoutput.beta[0]

def hedge(df, days_moving_avg): # USED DEBUGGING
	y = np.asarray(df[stock1].tolist()[-days_moving_avg:]) # stock 1 data
	x = np.asarray(df[stock2].tolist()[-days_moving_avg:]) # stock 2 data
	# Fit the data using scipy.odr
	def f(B, x):
		return B[0] * x + B[1]
	linear = odr.Model(f)
	mydata = odr.RealData(x, y, sx=np.std(x), sy=np.std(y))
	fit_np = np.polyfit(x, y, 1)
	print fit_np
	myodr = odr.ODR(mydata, linear, beta0=[fit_np[0], fit_np[1]]) # THINK ABOUT CHANGING BETA0
	myoutput = myodr.run()
	# fit the data using numpy.polyfit
	# graph to compare the fit
	print 'polyfit beta', fit_np[0]
	print 'least errors beta', myoutput.beta[0]
	plt.plot(x, y, label='Actual Data', linestyle='dotted') # actual data points
	plt.plot(x, np.polyval(fit_np, x), "r--", lw = 2, label='Polyfit') # polyfit regression line
	plt.plot(x, f(myoutput.beta, x), "g--", lw = 2, label='Least Errors') # least errors regression line
	plt.legend(loc='lower right')
	plt.show()
	# myoutput.pprint()
	# df["HedgeRatio"] = calculateHedgeRatio(df[])
	return myoutput.beta[0] # returns the hedge ratio

def staticHedge(df, days_moving_avg):
	hedge_ratio = hedge(df, days_moving_avg)
	df['HedgeRatio'] = hedge_ratio
	return df

def rollingHedge(df, stock1, stock2, days_moving_avg):
	df['HedgeRatio'] = np.nan
	# calculate the hedge ratios in a rolling manner
	for i in range(days_moving_avg - 1, len(df['HedgeRatio'])):
		df['HedgeRatio'][i] = leastErrorsHedgeRatio(df[:i], stock1, stock2, days_moving_avg)
	return df # last one should be  0.117262 !!!!!!

def calculateSpread(df, stock1, stock2, days_moving_avg):
	df['Spread'] = df[stock1] - df[stock2] * df['HedgeRatio']
	df['MovingAvg'] = df['Spread'].rolling(window=days_moving_avg).mean()
	df['Stdev'] = df['Spread'].rolling(window=days_moving_avg).std()
	df['UpperTrigger'] = df['MovingAvg'] + 2 * df['Stdev']
	df['LowerTrigger'] = df['MovingAvg'] - 2 * df['Stdev']
	print df
	return df

# now want to find the spread and the past 60 day standard deviation for spreads
def graph(df):
	df['Spread'].plot(label='Spread', color='g')
	df['MovingAvg'].plot(label='MovingAvg', color='b')
	df['UpperTrigger'].plot(label='UpperTrigger', color='r', linestyle='dashed')
	df['LowerTrigger'].plot(label='LowerTrigger', color='r', linestyle='dashed')
	plt.legend(loc='upper left')
	plt.show()

# prints correlations, cointegrations test results, etc.
def getStatistics(df, hedge_method):
	print 'Correlation Table\n', df[[stock1,stock2]].corr()
	coint_tstat, coint_pvalue, _ = tsa.coint(df[stock1], df[stock2])
	print 'Cointegration P-value', coint_pvalue
	# NEED TO RUN adfuller ON RESIDUALS OF LINEAR REGRESSION 
	if hedge_method == 'Static':
		cadf_pvalue = tsa.adfuller(df["Spread"])[1]
	else:
		cadf_pvalue = tsa.adfuller(df["Spread"][days_moving_avg - 1:])[1]
	print 'Augmented Dickey-Fuller P-value', cadf_pvalue

statisticalArb(capital_limit, stock1, stock2, start, end, days_moving_avg, hedge_method)
# findCointegratedPairs(tickers, start, end, days_moving_avg)

### Cointegrated Pairs ###
# Telecommunications
# [('AMX', 'ORAN', 0.03343, 0.00443), ('AMX', 'BCE', 0.00847, 0.00061), ('AMX', 'NEE', 0.04631, 0.00929)]

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
# CADF Cointegrated Testing: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing-Part-II
# Dynamic Hedge Ratios: https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter
# Finding cointegrated pairs: https://www.quantopian.com/posts/how-to-build-a-pairs-trading-strategy-on-quantopian

# Possible Useful Sources:
