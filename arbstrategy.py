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
	def __init__(self, cash, df, stock1, stock2, days_moving_avg):
		self.cash = cash
		self.df = df
		self.stock1 = stock1
		self.stock2 = stock2
		self.days_moving_avg = days_moving_avg

	def run():
		'''
		Runs the arbitrage strategy.
		'''
		df_trimmed = self.df[days_moving_avg - 1:] # start from the first day that a moving avg can be calculated
		s1_price = df_trimmed[self.stock1] 
		s2_price = df_trimmed[self.stock2]
		spread = df_trimmed['Spread']
		moving_avg = df_trimmed['MovingAvg']
		upper_trigger = df_trimmed['UpperTrigger']
		lower_trigger = df_trimmed['LowerTrigger']
		profit = 0 # total profit from transactions
		triggered_upper = False # whether or not an upper signal has been triggered
		triggered_lower = False # whether or not a lowwer signal has been triggered
		shares_s1 = 0 # number of shares of stock 1 currently held
		shares_s2 = 0
		for i in range(len(s1_price)):
			if triggered_upper:
				if spread[i] <= moving_avg[i]:
					# get rid of positions
					cash_left += shares_s1[i] * s1_price[i] + shares_s2[i] * s2_price[i] 
					shares_s1 = shares_s2 = 0
					triggered_upper = False
			elif triggered_lower:
				if spread[i] >= moving_avg[i]: 
					# get rid of positions
					cash_left += shares_s1[i] * s1_price[i] + shares_s2[i] * s2_price[i]
					shares_s1 = shares_s2 = 0
					triggered_lower = False
			else:
				if spread[i] >= upper_trigger[i]:
					# short stock1, buy stock2
					triggered_upper = True
					shares_s1 += cash_left / s1_price
					shares_s2 += 
				elif spread[i] <= lower_trigger[i]:
					# buy stock1, short stock2
					triggered_lower = True
		return