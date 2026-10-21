import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random_number_generator = np.random.default_rng(42)

"""here are some assumptions"""

# number of benchmark companies
n_peers = 5
# number of total companies in this simulation
n_companies = n_peers + 1
# number of simulation
n_simulations = 100000
# number of performance years
performance_years = 3
# number of trading days
trading_days = 252

# ARBITRARY stock price at time t = 0 
initial_price = np.array([50, 45, 60, 35, 70, 55])

# some names for the company
company_names = ["client", "A", "B", "C", "D", "E"]

#test
# print(pd.Series(initial_price, index = company_names))

"""this following section generates historical stock return data"""

n_historical_years = 3
n_historical_days = n_historical_years * trading_days

# arbitrary annual volatility data generation
annual_volatility = np.array([0.30, 0.25, 0.35, 0.28, 0.40, 0.32])
# generate arbitrary annual returns into a matrix
historical_returns = random_number_generator.normal(loc = 0, scale = annual_volatility / np.sqrt(trading_days), size = (n_historical_days, n_companies))
# structure the generated returns per company into a dataframe
historical_returns = pd.DataFrame(historical_returns, columns = company_names)

# test, shows the generated historical return data
#print(historical_returns.info)

"""this section calculates historical volitility and correlation"""

# annual volatility, ie stdev
estimated_volatility = historical_returns.std() * np.sqrt(trading_days)

# correlation
correlation_matrix = historical_returns.corr()

# test, now we can see the matrix is ready for cholesky
# print(estimated_volatility)
# print(correlation_matrix)

"""cholesky factorization"""
L = np.linalg.cholesky(correlation_matrix.to_numpy())

# generate independent shocks that is standard normal Z
independent_shocks = random_number_generator.standard_normal(size = (n_simulations, n_companies))

# turn independent shocks into correlated ones which is the vector of all independent shocks times L transpose.
correlated_shocks = independent_shocks @ L.T

# tests
# this one should be lower triangular matrix
# print(L)
# checking independent shocks, good size
# print(independent_shocks, independent_shocks.shape)
# print(correlation_shocks)
# final sanity check
# print (np.allclose(L @ L.T, correlation_matrix.to_numpy()))

"""geometric brownian motion"""

# assume a RFR, in real life, we go find the US treasury zero coupon spot rate that matches our duration and adjust based on convention
rfr = 0.04
# this is also an assumption that we are not paying dividend
# dividend_yield = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
dividend_yield = np.zeros(n_companies)

volatility = estimated_volatility.to_numpy()
# this one below is the part that goes (r-delta-1/2sigma**2)T
drift = (rfr - dividend_yield - 0.5 * volatility ** 2) * performance_years
# this one below is basically the sigma sqrt(T) * Z
random_component = (volatility * np.sqrt(performance_years) * correlated_shocks)
# this function below calculates the S at timet = T
terminal_prices = initial_price * np.exp(drift + random_component)

# test. ended up finding I did the correlated shocks and independent shocks linalg wrong.
# print(terminal_prices.shape)
# print(pd.DataFrame(terminal_prices, columns = company_names).head())

"""TSR Total Shareholder Return"""

tsr = terminal_prices / initial_price - 1

tsr_df = pd.DataFrame(tsr, columns = company_names)

#test
# print(tsr_df.shape)
# print(tsr_df.head())

"""rank the tsr"""
# we are doing ascending order this time
tsr_ranks = tsr_df.rank(axis = 1, ascending = True, method = "min")
# our rank
client_rank = tsr_ranks["client"]

# test, this one was hard to write correctly
# print(pd.DataFrame({"Client TSR ": tsr_df["client"], "Client Rank ": client_rank}).head())

"""PSU payout schedule, assumed arbitrary"""
payout_schedule = np.array([0.0, 0.5, 0.75, 1.0, 1.5, 2.0])
payout_multiplier = payout_schedule[client_rank.to_numpy().astype(int) - 1]

# sanity check, I should've made the order descending, now if the client is ranked 5th technically he's second best. 
# print(pd.DataFrame(client_rank))
# print(pd.DataFrame(payout_multiplier))

"""Grant-Date Fair Value"""
# assumptions again
target_shares = 1000
# shares earned in each simulation
shares_awarded = target_shares * payout_multiplier
# value of these shares at year 3
simulated_payoffs = shares_awarded * terminal_prices[:, 0]
# apply discount
pv = simulated_payoffs * np.exp(-rfr * performance_years)

# avg across the 10,000 simulations to find the sum of 1000x fair value
grant_date_fair_value = np.mean(pv)
# result
fair_value_per_target_psu = grant_date_fair_value / target_shares

# Finally we used monte-carlo to get this grant-date fair price
print(f"Total Grant-Date Fair Value = ${grant_date_fair_value:,.2f}")
print(f"Fair Value per Target PSU: ${fair_value_per_target_psu:,.2f}")

# I'd also like to present a 95% CI for the client
mcse = (np.std(pv, ddof = 1) / np.sqrt(n_simulations) / target_shares)

print(f"We are 95% confident from this Monte-Carlo simulation that the simulated grant-date fair value per target PSU is between : ${fair_value_per_target_psu - 1.96 * mcse:.2f} and ${fair_value_per_target_psu + 1.96 * mcse:.2f}")

