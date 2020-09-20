import pandas as pd
import numpy as np
from impliedVolatility.ImpliedVolatility import ImpliedVolatility

modes = ['impvol', 'histvol']

for mode in modes:
    if mode == 'impvol':
        test_df = pd.read_csv('data-source/test_df.csv')
    else:
        test_df = pd.read_csv('data-source/test_df_histvol.csv')

    # read x_values and y_values
    y_values = test_df[['result']].values

    bs_obj = ImpliedVolatility(np.log(1.00501), 1, 1)
    underlying = test_df['3'].values
    call_put = test_df['0'].values
    strike = test_df['1'].values
    tenor = test_df['4'].values
    sigma = test_df['5'].values

    results = bs_obj.calc_opt_price(underlying, call_put, strike, tenor, sigma)

    data = {'results': results.reshape(-1,),
            'expected': y_values.reshape(-1,)}

    final_df = pd.DataFrame(data=data)
    final_df['diff'] = final_df['expected'] - final_df['results']
    final_df['mse'] = np.mean(np.square(final_df['diff']))
    final_df['rel'] = final_df['diff'] / final_df['expected']
    final_df['bias'] = 100 * np.median(final_df['rel'])
    final_df['aape'] = 100 * np.mean(np.abs(final_df['rel']))
    final_df['mape'] = 100 * np.median(np.abs(final_df['rel']))
    final_df['pe5'] = 100 * sum(np.abs(final_df['rel']) < 0.05) / len(final_df['rel'])
    final_df['pe10'] = 100 * sum(np.abs(final_df['rel']) < 0.10) / len(final_df['rel'])
    final_df['pe20'] = 100 * sum(np.abs(final_df['rel']) < 0.20) / len(final_df['rel'])

    final_df.to_csv(f'data-source/bs-results-{mode}.csv', index=False)

    statistics = {
        'max': np.max(final_df['diff']),
        'mean': np.mean(final_df['diff']),
        'median': np.median(final_df['diff']),
        'min': np.min(final_df['diff']),
        'rmse': np.sqrt(np.mean(np.power(final_df['diff'], 2))),
        'sse': np.sum(np.power(final_df['diff'], 2)),
        'std': np.std(final_df['diff']),
        'mse': final_df['mse'].mean(),
        'aape': final_df['aape'].mean(),
        'mape': final_df['mape'].mean(),
        'pe5': final_df['pe5'].mean(),
        'pe10': final_df['pe10'].mean(),
        'pe20': final_df['pe20'].mean(),
    }

    # write response to a .txt file
    with open(f'data-source/bs-statistics-{mode}.txt', 'w') as f:
        for key, value in statistics.items():
            f.write(f'{key}: {value} \n\n')

    print()

