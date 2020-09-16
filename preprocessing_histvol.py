import pandas as pd
import numpy as np

# read DataFrames with available info about options
df_contracts = pd.read_csv('data-source/contracts.csv', usecols=['commodity', 'contract', 'expiry_date'])
df_fut = pd.read_excel('data-source/und-fut.xlsx', usecols=['commodity', 'contract', 'date', 'future_price'])
df_opt = pd.read_csv('data-source/option-price.csv', usecols=['commodity', 'contract', 'date', 'put_or_call', 'strike_value', 'settle_price'])
df_fut['date'] = df_fut['date'].dt.strftime('%Y-%m-%d')

# here we merge the DataFrame with information's about contracts
# into the one with information's about the options so we are able to
# get the expiration date to every option we are looking for
df_opt = pd.merge(df_opt, df_contracts, how='left', left_on=['commodity', 'contract'], right_on=['commodity', 'contract'])

# here we merge the DataFrame with information's about future
# into the one with information's about the options so we are able to
# get the underlying price to every option we are looking for
df_opt = pd.merge(df_opt, df_fut, how='left', left_on=['commodity', 'contract', 'date'], right_on=['commodity', 'contract', 'date'])

# in order to calculate the option tenor we first need to know
# the number of days between the "operation date" eg, the day data
# was collected and the expiration date of the contract
days_to_mat = (pd.to_datetime(df_opt['expiry_date']) - pd.to_datetime(df_opt['date'])).dt.days.values
# the tenor is a number representing
# the number of days to maturity divide
# by 365.25 (since we consider) running days
tenor = (days_to_mat + 1) / 365.25

# add the days to maturity and the tenor
# to the options info DataFrame
df_opt['days_to_mat'] = days_to_mat
df_opt['tenor'] = tenor

# drop if any NaN number exists on DataFrame
df_opt = df_opt[df_opt['days_to_mat'] > 0]
df_opt = df_opt.dropna(axis=0, how='any')
df_opt['date'] = pd.to_datetime(df_opt['date'])

# in order to calculate the historical volatility
# we first need to drop all duplicate values for a same
# (date, commodity, contract) combination
temp_df = df_opt.drop_duplicates(['commodity', 'contract', 'date'])
temp_df = temp_df[['commodity', 'contract', 'date', 'future_price']]

# get a list of all unique commodities
commodities = list(temp_df.commodity.unique())

# create a df to store the volatility's
vol_df = pd.DataFrame(data={'commodity': [], 'contract': [],
                            'vol': [], 'date': []})

# iterate over the unique commodities
for comm in commodities:
    # get unique contracts for the commodity
    temp = temp_df[temp_df['commodity'] == comm]
    maturities = list(temp.contract.unique())
    # iterate over all mats
    for mat in maturities:
        # calculate the annualized historical vol
        temp2 = temp[temp['contract'] == mat].sort_values('date')
        vol = temp2['future_price'].pct_change().rolling(20).std()*(252**0.5)
        new_df = pd.DataFrame(data={'commodity': temp2['commodity'].values,
                                    'contract': temp2['contract'].values,
                                    'date': temp2['date'].values,
                                    'vol': vol})
        if not new_df.empty:
            vol_df = vol_df.append(new_df)

# merge the vols with the rest of the information's
df_opt = pd.merge(df_opt, vol_df, how='left', left_on=['commodity', 'contract', 'date'], right_on=['commodity', 'contract', 'date'])
df_opt = df_opt.dropna(axis=0, how='any')

# save the options DataFrame that will be
# used in the future to training the machine
# learning model in order to predict options price
df_opt.to_csv('data-source/options-data-hist-vol.csv')
