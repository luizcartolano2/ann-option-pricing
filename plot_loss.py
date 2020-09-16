import matplotlib.pyplot as plt
import re

filename = 'records/loss_histvol_20-09-16.txt'

with open(filename) as f:
    lines = f.readlines()

epochs = []
train_loss = []
valid_loss = []

print()
epoch = 0
for i, line in enumerate(lines):
    if 'Epoch' in line:
        epoch += 1
        epochs.append(epoch)
        print(f'Epoch: {epoch}')
    elif 'train' in line:
        temp1 = line.find(':')
        temp2 = line[temp1:].find(',')
        loss = float(line[temp1+1: temp1+temp2])
        train_loss.append(loss)

        print(f'\tTrain Loss: {loss}')
    else:
        temp1 = line.find(':')
        temp2 = line[temp1:].find(',')
        loss = float(line[temp1 + 1: temp1 + temp2])
        valid_loss.append(loss)

        print(f'\tValid Loss: {loss}')


plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, valid_loss, label='Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.savefig('figures/train-mlp-histvol.png')
