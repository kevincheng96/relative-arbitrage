import numpy as np

# calculates the total profit earned from this strategy

class ArbStrategy:
	'''
	A class that implements a relative-value arbitrage strategy.
	when residual crosses higher trigger level, sell stock1 and buy stock2.
	when residual crosses lower trigger level, buy stock1 and sell stock2.
	when residual returns to moving avg, get rid of both positions
	want the residual to be at a higher level than when initially bought
	REMEMBER:
	if bought at higher trigger level, want a downward slope between buy residual and sell residual
	if bought at lower trigger level, want a upward slope between buy residual and sell residual 
	'''
	def __init__(self, capital_limit, df, stock1, stock2, days_moving_avg):
		self.capital_limit = capital_limit
		self.df = df
		self.x = stock1
		self.y = stock2
		self.days_moving_avg = days_moving_avg

	def run(self):
		'''
		Runs the arbitrage strategy.
		'''
		df_trimmed = self.df[self.days_moving_avg - 1:] # start from the first day that a moving avg can be calculated
		x_price = df_trimmed[self.x] # an array of daily prices for stock x
		y_price = df_trimmed[self.y] # an array of daily prices for stock y
		residual = df_trimmed['Residual']
		moving_avg = df_trimmed['MovingAvg']
		hedge_ratio = df_trimmed['HedgeRatio']
		intercept = df_trimmed['Intercept']
		upper_trigger = df_trimmed['UpperTrigger']
		lower_trigger = df_trimmed['LowerTrigger']
		triggered_upper = False # whether or not an upper signal has been triggered
		triggered_lower = False # whether or not a lower signal has been triggered
		x_shares = 0 # number of shares of stock x currently held
		y_shares = 0 # number of shares of stock y currently held
		x_buyprice = 0 # price at which x was bought
		y_buyprice = 0 # price at which y was bought
		trades = [] # a list containing the history of trades
		profit = 0 # total profit from transactions

		##### Since hedge ratio changes every day, let hedge stay fixed after entering a position
		##### Need to recalculate all residuals based on this hedge ratio. Once position is closed,
		##### calculate the residuals again with a new hedge ratio.   
		for i in range(len(x_price)):
			# if residual[i] == np.nan: # pass if no data is available
			# 	pass
			if triggered_upper:
				if residual[i] <= moving_avg[i]:
					# get rid of positions (short x, long y) by buying x and selling y
					profit += x_shares * (x_price[i] - x_buyprice) + y_shares * (y_price[i] - y_buyprice) 
					print x_shares, x_price[i], y_shares, y_price[i]
					x_shares = y_shares = 0
					x_buyprice = y_buyprice = 0
					residual = df_trimmed['Residual'] # reset the residual to the originally calculated residual
					triggered_upper = False
					trades.append((df_trimmed.index[i], "SOLD UPPER"))
			elif triggered_lower:
				if residual[i] >= moving_avg[i]: 
					# get rid of positions (long x, short y) by selling x and buying y
					profit += x_shares * (x_price[i] - x_buyprice) + y_shares * (y_price[i] - y_buyprice) 
					print 'x profit', x_shares * (x_price[i] - x_buyprice)
					print 'y profit', y_shares * (y_price[i] - y_buyprice)
					x_shares = y_shares = 0
					x_buyprice = y_buyprice = 0
					residual = df_trimmed['Residual'] # reset the residual to the originally calculated residual
					triggered_lower = False
					trades.append((df_trimmed.index[i], "SOLD LOWER"))
			else:
				if hedge_ratio[i] <= 0: # ignore if they are negatively correlated (too risky)
					continue
				if residual[i] >= upper_trigger[i]:
					# short stock x, buy stock y
					triggered_upper = True
					x_buyprice = x_price[i]
					y_buyprice = y_price[i]
					x_shares = -1 * self.capital_limit/(x_price[i] + hedge_ratio[i] * y_price[i]) # amount of stock x to short (negative)
					y_shares = (-1 * x_shares) * hedge_ratio[i] # amount of stock y to buy
					residual = x_price - hedge_ratio[i] * y_price - intercept[i] # recalculate the residuals using the current hedge ratio
					print 'hedge ratio', hedge_ratio[i] 
					print 'x shares', x_shares
					print 'y shares', y_shares
					print 'net exposure', x_shares*x_price[i] + y_shares*y_price[i] # need net position to be 0
					trades.append((df_trimmed.index[i], "UPPER TRIGGER"))
				elif residual[i] <= lower_trigger[i]:
					# buy stock x, short stock y
					triggered_lower = True
					x_buyprice = x_price[i]
					y_buyprice = y_price[i] 
					x_shares = self.capital_limit/(x_price[i] + hedge_ratio[i] * y_price[i]) # amount of stock x to buy
					y_shares = (-1 * x_shares) * hedge_ratio[i] # amount of stock y to short (negative)
					residual = x_price - hedge_ratio[i] * y_price - intercept[i] # recalculate the residuals using the current hedge ratio
					print 'hedge ratio', hedge_ratio[i] 
					print 'x shares', x_shares
					print 'y shares', y_shares
					print 'net exposure', x_shares*x_price[i] + y_shares*y_price[i]
					trades.append((df_trimmed.index[i], "LOWER TRIGGER"))
				else:
					pass # do nothing if no levels are triggered
		print "Total profit is", profit
		print trades
		return profit

