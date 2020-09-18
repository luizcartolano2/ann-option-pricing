import pandas as pd
import numpy as np
from src.LSTM import LSTM
from src.predict import predict_lstm
import torch

modes = ['impvol', 'histvol']
# modes = ['impvol']

for mode in modes:
    if mode == 'impvol':
        test_df = pd.read_csv('data-source/test_df.csv')
        model_path = 'models/train_model_lstm_at_20-09-15.model.train'
    else:
        test_df = pd.read_csv('data-source/test_df_histvol.csv')
        model_path = 'models/train_model_lstm_histvol_at_20-09-16.model.train'

    # read x_values and y_values
    y_values = test_df[['result']].values
    x_values = test_df.drop(['result'], axis=1).values

    M_lstm = LSTM(x_values.shape[1])

    M_lstm.load_state_dict(torch.load(model_path))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDA used.')
        M_lstm = M_lstm.cuda()

    M_lstm.eval()

    results = predict_lstm(x_values, M_lstm)

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

    final_df.to_csv(f'data-source/lstm-results-{mode}.csv', index=False)

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
    with open(f'data-source/lstm-statistics-{mode}.txt', 'w') as f:
        for key, value in statistics.items():
            f.write(f'{key}: {value} \n\n')

    print()

