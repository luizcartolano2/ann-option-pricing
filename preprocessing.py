import pandas as pd
import numpy as np
from impliedVolatility.volimp import volimp

# read DataFrames with available info about options
df_contracts = pd.read_csv('data-source/contracts.csv', usecols=['commodity', 'contract', 'expiry_date'])
df_fut = pd.read_csv('data-source/und-fut.csv', usecols=['commodity', 'contract', 'date', 'future_price'])
df_opt = pd.read_csv('data-source/option-price.csv', usecols=['commodity', 'contract', 'date', 'put_or_call', 'strike_value', 'settle_price'])

# here we merge the DataFrame with informations about contracts
# into the one with informations about the options so we are able to
# get the expiration date to every option we are looking for
df_opt = pd.merge(df_opt, df_contracts, how='left', left_on=['commodity', 'contract'], right_on=['commodity', 'contract'])

# here we merge the DataFrame with informations about future
# into the one with informations about the options so we are able to
# get the underlying price to every option we are looking for
df_opt = pd.merge(df_opt, df_fut, how='left', left_on=['commodity', 'contract', 'date'], right_on=['commodity', 'contract', 'date'])

# in order to calculate the option tenor we firts need to know
# the number of days between the "operation date" eg, the day data
# was collected and the expiration date of the contract
days_to_mat = (pd.to_datetime(df_opt['expiry_date']) - pd.to_datetime(df_opt['date'])).dt.days.values
# the tenor is a number representing
# the number of days to maturiry divide
# by 365.25 (since we consider) running days
tenor = (days_to_mat + 1) / 365.25

# add the days to maturiry and the tenor
# to the options info DataFrame
df_opt['days_to_mat'] = days_to_mat
df_opt['tenor'] = tenor

# drop if any NaN number exists on DataFrame
df_opt = df_opt.dropna(axis=0, how='any')

# in order to calculate the implied volatility
# we are going to use the Black Scholes model
# together with a bissection function to find equation roots
prices = df_opt['settle_price'].values.reshape((-1, 1))
underlying = df_opt['future_price'].values.reshape((-1, 1))
strike = df_opt['strike_value'].values.reshape((-1, 1))
call_put = df_opt['put_or_call'].values.reshape((-1, 1)).astype('float64')
r = np.log(1.00501)
tenor = df_opt['tenor'].values.reshape((-1, 1))
i, temp, vols = volimp(prices, underlying, strike, call_put, r, tenor, 0.0001, 2000)

# add the implied volatility
# to the options DataFrame
df_opt['volimp'] = vols

### save the options DataFrame that will be
# used in the future to training the machine
# learning model in order to predict options price
df_opt.to_csv('data-source/options-data.csv')
