# calculates the total profit earned from this strategy

class ArbStrategy:
	'''
	A class that implements a relative-value arbitrage strategy.
	when spread crosses higher trigger level, sell stock1 and buy stock2.
	when spread crosses lower trigger level, buy stock1 and sell stock2.
	when spread returns to moving avg, get rid of both positions
	want the spread to be at a higher level than when initially bought
	REMEMBER:
	if bought at higher trigger level, want a downward slope between buy spread and sell spread
	if bought at lower trigger level, want a upward slope between buy spread and sell spread 
	'''
	def __init__(self, capital_limit, df, stock1, stock2, days_moving_avg):
		self.capital_limit = capital_limit
		self.df = df
		self.x = stock1
		self.y = stock2
		self.days_moving_avg = days_moving_avg

	def run():
		'''
		Runs the arbitrage strategy.
		'''
		df_trimmed = self.df[days_moving_avg - 1:] # start from the first day that a moving avg can be calculated
		x_price = df_trimmed[self.x] # an array of daily prices for stock x
		y_price = df_trimmed[self.y] # an array of daily prices for stock y
		spread = df_trimmed['Spread']
		moving_avg = df_trimmed['MovingAvg']
		hedge_ratio = df_trimmed['HedgeRatio']
		upper_trigger = df_trimmed['UpperTrigger']
		lower_trigger = df_trimmed['LowerTrigger']
		triggered_upper = False # whether or not an upper signal has been triggered
		triggered_lower = False # whether or not a lower signal has been triggered
		x_shares = 0 # number of shares of stock x currently held
		y_shares = 0 # number of shares of stock y currently held
		trades = [] # a list containing the history of trades
		profit = 0 # total profit from transactions

		##### Since hedge ratio changes every day, let hedge stay fixed after entering a position
		##### Need to recalculate all spreads based on this hedge ratio. Once position is closed,
		##### calculate the spreads again with a new hedge ratio.   
		for i in range(len(x_price)):
			if triggered_upper:
				if spread[i] <= moving_avg[i]:
					# get rid of positions (short x, long y) by buying x and selling y
					profit += x_shares * x_price[i] + y_shares * y_price[i] 
					x_shares = y_shares = 0
					triggered_upper = False
			elif triggered_lower:
				if spread[i] >= moving_avg[i]: 
					# get rid of positions (long x, short y) by selling x and buying y
					profit += x_shares * x_price[i] + y_shares * y_price[i]
					x_shares = y_shares = 0
					triggered_lower = False
			else:
				if spread[i] >= upper_trigger[i]:
					# short stock x, buy stock y
					triggered_upper = True
					x_buy_price = -x_price[i]
					y_buy_price = y_price[i]
					x_shares = -self.capital_limit/(x_price[i] + hedge_ratio[i] * y_price[i]) # amount of stock x to short
					y_shares = -x_shares * hedge_ratio[i] # amount of stock y to buy
				elif spread[i] <= lower_trigger[i]:
					# buy stock x, short stock y
					triggered_lower = True
					x_buy_price = x_price[i]
					y_buy_price = -y_price[i]
					x_shares = self.capital_limit/(x_price[i] + hedge_ratio[i] * y_price[i]) # amount of stock x to buy
					y_shares = -x_shares * hedge_ratio[i] # amount of stock y to short
				else:
					pass # do nothing if no levels are triggered
		return "Total profit is", profit

