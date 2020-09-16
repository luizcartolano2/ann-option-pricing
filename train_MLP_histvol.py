import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Function
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from src.MultilayerPerceptron import MultilayerPerceptron
import datetime

today_datetime = datetime.date.today().strftime('%y-%m-%d')

# Prepare Data
# columns to use
usecols = ['put_or_call', 'strike_value', 'settle_price',
           'future_price', 'tenor', 'vol']
# read dataframe
df = pd.read_csv('data-source/options-data-hist-vol.csv', usecols=usecols)
# set a risk free column
df['risk_free'] = np.ones((len(df),)) * np.log(1.00501)

# read x_values and y_values
x_values = df[['put_or_call', 'strike_value', 'risk_free', 'future_price', 'tenor', 'vol']].values
y_values = df[['settle_price']].values

# split df into train : 70% / validation : 15% / test : 15%
x_train, x_valid, y_train, y_valid = train_test_split(x_values, y_values, test_size=0.3)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5)

# save the test values to use later
test_df = pd.DataFrame(x_test)
test_df['result'] = y_test
test_df.to_csv('data-source/test_df_histvol.csv')

# convert numpy to tensors
features_train = torch.from_numpy(x_train)
targets_train = torch.from_numpy(y_train)
features_valid = torch.from_numpy(x_valid)
targets_valid = torch.from_numpy(y_valid)

# create tensor dataset
train_dataset = TensorDataset(features_train, targets_train)
valid_dataset = TensorDataset(features_valid, targets_valid)

# create dataloader object
train_dl = DataLoader(train_dataset, batch_size=8)
valid_dl = DataLoader(valid_dataset, batch_size=8)

print('='*50)
print(f'Number of samples in training dataset: {x_train.shape[0]}')
print(f'Number of samples in validation dataset: {x_valid.shape[0]}')
print(f'Number of samples in test dataset: {x_test.shape[0]}')
print('='*50)

# Model
print(f"Number of features to use: {x_train.shape[1]}")

M_mlp = MultilayerPerceptron(x_train.shape[1])
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('CUDA used.')
    M_mlp = M_mlp.cuda()
print(M_mlp)
print('='*50)

# Create optimizer
optimizer = torch.optim.Adam(M_mlp.parameters(), lr=0.0001)

# Train
save_model_path = f'models/train_model_histvol_at_{today_datetime}.model'
record_path = f'records/loss_histvol_{today_datetime}.txt'
save_optimizer_path = f'optimizer/optimizer_model_histvol_at_{today_datetime}.optimizer'

print('Record loss in: ', record_path)
min_loss_t = 1e10
min_loss_v = 1e10
epochs = 200

M_mlp.train()
for ep in range(epochs):
    st_t = time.time()
    print('='*50)

    # Train
    M_mlp.train()
    loss_mean = 0
    t_loss_list = []
    for t_x, t_y in train_dl:
        if use_cuda:
            t_x = t_x.cuda(non_blocking=True)
            t_y = t_y.cuda(non_blocking=True)

        ls = M_mlp.step(t_x, t_y, optimizer).data.cpu().numpy()
        t_loss_list.append(float(ls))
        loss_mean += float(ls)

    print('Train take {:.1f} sec'.format(time.time()-st_t))
    loss_mean /= len(train_dl)

    # Validation
    st_t = time.time()
    M_mlp.eval()
    loss_mean_valid = 0
    v_loss_list = []
    for v_x, v_y in valid_dl:
        if use_cuda:
            v_x = v_x.cuda(non_blocking=True)
            v_y = v_y.cuda(non_blocking=True)

        v_ls = M_mlp.get_loss(v_x, v_y).data.cpu().numpy()
        v_loss_list.append(float(v_ls))
        loss_mean_valid += float(v_ls)

    print('Valid take {:.1f} sec'.format(time.time()-st_t))
    loss_mean_valid /= len(valid_dl)

    f = open(record_path, 'a')
    f.write('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))
    print('Epoch {}\ntrain loss mean: {}, std: {:.2f}\nvalid loss mean: {}, std: {:.2f}\n'.format(ep+1, loss_mean, np.std(t_loss_list), loss_mean_valid, np.std(v_loss_list)))

    # Save model
    # save if the valid loss decrease
    check_interval = 1
    if loss_mean_valid < min_loss_v and ep % check_interval == 0:
        min_loss_v = loss_mean_valid
        print('Save model at ep {}, mean of valid loss: {}'.format(ep+1, loss_mean_valid))
        torch.save(M_mlp.state_dict(), save_model_path+'.valid')
        torch.save(optimizer.state_dict(), save_optimizer_path + '.valid')

    # save if the training loss decrease
    check_interval = 1
    if loss_mean < min_loss_t and ep % check_interval == 0:
        min_loss_t = loss_mean
        print('Save model at ep {}, mean of train loss: {}'.format(ep+1, loss_mean))
        torch.save(M_mlp.state_dict(), save_model_path+'.train')
        torch.save(optimizer.state_dict(), save_optimizer_path + '.train')
    f.close()
