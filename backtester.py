import numpy as np
import matplotlib.pyplot as plt
import copy

# calculates the total profit earned from this arb-stat strategy

class Backtester:
	'''
	A class that implements a relative-value arbitrage strategy back-tester.
	When the residual crosses higher trigger level, sell stock1 and buy stock2.
	When the residual crosses lower trigger level, buy stock1 and sell stock2.
	When the residual returns to moving avg, get rid of both positions.
	We want the residual to be at a higher level than when initially bought.
	REMEMBER:
	If bought at higher trigger level, want a downward slope between buy residual and sell residual
	If bought at lower trigger level, want a upward slope between buy residual and sell residual 
	'''
	def __init__(self, capital_limit, df, stock1, stock2, days_moving_avg):
		self.capital_limit = capital_limit
		self.df = df
		self.x = stock1
		self.y = stock2
		self.days_moving_avg = days_moving_avg

	def run(self):
		'''
		Runs backtesting for the statistical arbitrage strategy.
		'''
		df = self.df
		# make a copy for reference later
		df_original = copy.deepcopy(df)
		x_price = df[self.x] # an array of daily prices for stock x
		y_price = df[self.y] # an array of daily prices for stock y
		triggered_upper = False # whether or not an upper signal has been triggered
		triggered_lower = False # whether or not a lower signal has been triggered
		x_shares = 0 # number of shares of stock x currently held
		y_shares = 0 # number of shares of stock y currently held
		x_buyprice = 0 # price at which x was bought
		y_buyprice = 0 # price at which y was bought
		trades = [] # a list containing the history of trades
		profit = 0 # total profit from transactions

		##### Since hedge ratio changes every day, let hedge ratio stay fixed after entering a position.
		##### Then, recalculate all residuals and trigger data based on this constant hedge ratio. Once position is closed,
		##### reset residuals and triggers to the old data.  
		# TODO: How to update Moving Avg? Current method is actually making the triggers look weird.
		for i in range(len(df)):
			if df['MovingAvg'][i] == np.nan: # pass if no data is available
				pass
			if triggered_upper:
				if df['Residual'][i] <= df['MovingAvg'][i]:
					# get rid of positions (short x, long y) by buying x and selling y
					profit += x_shares * (x_price[i] - x_buyprice) + y_shares * (y_price[i] - y_buyprice) 
					print x_shares, x_price[i], y_shares, y_price[i]
					x_shares = y_shares = 0
					x_buyprice = y_buyprice = 0
					df[i:] = df_original[i:] # reset residual, moving avg, and triggers to the originally values
					triggered_upper = False
					trades.append((df.index[i], "SOLD UPPER"))
			elif triggered_lower:
				if df['Residual'][i] >= df['MovingAvg'][i]: 
					# get rid of positions (long x, short y) by selling x and buying y
					profit += x_shares * (x_price[i] - x_buyprice) + y_shares * (y_price[i] - y_buyprice) 
					print 'x profit', x_shares * (x_price[i] - x_buyprice)
					print 'y profit', y_shares * (y_price[i] - y_buyprice)
					x_shares = y_shares = 0
					x_buyprice = y_buyprice = 0
					df[i:] = df_original[i:] # reset residual, moving avg, and triggers to the originally values
					triggered_lower = False
					trades.append((df.index[i], "SOLD LOWER"))
			else:
				if df['HedgeRatio'][i] <= 0: # ignore if they are negatively correlated (too risky)
					continue
				if df['Residual'][i] >= df['UpperTrigger'][i]:
					# short stock x, buy stock y
					triggered_upper = True
					x_buyprice = x_price[i]
					y_buyprice = y_price[i]
					x_shares = -1 * self.capital_limit/(x_price[i] + df['HedgeRatio'][i] * y_price[i]) # amount of stock x to short (negative)
					y_shares = (-1 * x_shares) * df['HedgeRatio'][i] # amount of stock y to buy
					df['Residual'][i:] = x_price[i:] - df['HedgeRatio'][i] * y_price[i:] - df['Intercept'][i] # recalculate the residuals using the current hedge ratio
					df[i:] = self.calculateTriggers(df, self.days_moving_avg, i)
					print 'hedge ratio', df['HedgeRatio'][i] 
					print 'x shares', x_shares
					print 'y shares', y_shares
					print 'net exposure', x_shares*x_price[i] + y_shares*y_price[i] # need net position to be 0
					trades.append((df.index[i], "UPPER TRIGGER"))
				elif df['Residual'][i] <= df['LowerTrigger'][i]:
					# buy stock x, short stock y
					triggered_lower = True
					x_buyprice = x_price[i]
					y_buyprice = y_price[i] 
					x_shares = self.capital_limit/(x_price[i] + df['HedgeRatio'][i] * y_price[i]) # amount of stock x to buy
					y_shares = (-1 * x_shares) * df['HedgeRatio'][i] # amount of stock y to short (negative)
					df['Residual'][i:] = x_price[i:] - df['HedgeRatio'][i] * y_price[i:] - df['Intercept'][i] # recalculate the residuals using the current hedge ratio
					df[i:] = self.calculateTriggers(df, self.days_moving_avg, i) 
					print 'hedge ratio', df['HedgeRatio'][i] 
					print 'x shares', x_shares
					print 'y shares', y_shares
					print 'net exposure', x_shares*x_price[i] + y_shares*y_price[i]
					trades.append((df.index[i], "LOWER TRIGGER"))
				else:
					pass # do nothing if no levels are triggered
		print "Total profit is", profit
		self.graphResidualsAndTriggers(df)
		# Graph trades
		self.graphTrades(trades)
		print trades
		return profit

	# Calculating moving average and trigger data starting from the index. This is called 
	# only when a new position has been entered and the hedge ratio remains constant while 
	# the position is being held.
	def calculateTriggers(self, df, days_moving_avg, index):
		i = index
		df['MovingAvg'][i:] = (df['Residual'].rolling(window=days_moving_avg).mean())[i:]
		df['Stdev'][i:] = (df['Residual'].rolling(window=days_moving_avg).std())[i:]
		df['UpperTrigger'][i:] = (df['MovingAvg'] + 2 * df['Stdev'])[i:]
		df['LowerTrigger'][i:] = (df['MovingAvg'] - 2 * df['Stdev'])[i:]
		return df

	# Graphs the historical residuals, moving average, and triggers.
	def graphResidualsAndTriggers(self, df):
		df['Residual'].plot(label='Residual', color='g')
		df['MovingAvg'].plot(label='MovingAvg', color='b')
		df['UpperTrigger'].plot(label='UpperTrigger', color='r', linestyle='dashed')
		df['LowerTrigger'].plot(label='LowerTrigger', color='r', linestyle='dashed')
		plt.title('Residuals and Triggers')
		plt.ylabel('Spread')
		plt.xlabel('Date')
		plt.legend(loc='upper left')
		plt.show()

	# Graphs each trade (entering position and exiting position).
	# TODO: Also graph PnL below.
	def graphTrades(self, trades):
		'''
		Graphs each buy and sell point.
		'''
		df = self.df
		for trade in trades:
			date, trade_type = trade
			x_price = df[self.x][date]
			y_price = df[self.y][date]
			if trade_type == "UPPER TRIGGER":
				# short x, buy y
				plt.plot([date], [x_price], marker='o', markersize=3, color="red")
				plt.plot([date], [y_price], marker='o', markersize=3, color="green")
			elif trade_type == "LOWER TRIGGER":
				# buy x, short y
				plt.plot([date], [x_price], marker='o', markersize=3, color="green")
				plt.plot([date], [y_price], marker='o', markersize=3, color="red")
			elif trade_type == "SOLD UPPER":
				# buy x, sell y
				plt.plot([date], [x_price], marker='x', markersize=3, color="black")
				plt.plot([date], [y_price], marker='x', markersize=3, color="black")
			elif trade_type == "SOLD LOWER":
				# sell x, buy y
				plt.plot([date], [x_price], marker='x', markersize=3, color="black")
				plt.plot([date], [y_price], marker='x', markersize=3, color="black")
		plt.title('Buy and Sell Points')
		plt.show()
