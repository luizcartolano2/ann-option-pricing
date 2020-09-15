import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, lstm_size=100, lstm_layers=4, output_size=1, dropout=0.1):
        """

            :param input_size:
            :param lstm_size:
            :param lstm_layers:
            :param output_size:
            :param dropout:
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.dropout = dropout

        # Setup LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.lstm_size, self.lstm_layers,
                            dropout=self.dropout, batch_first=False)

        # Setup additional layers
        self.fc = nn.Linear(self.lstm_size, self.output_size)

    def forward(self, x, hidden_state):
        """

            :param x:
            :param hidden_state:
            :return:
        """
        nn_input = x.float()
        lstm_out, hidden_state = self.lstm(nn_input, hidden_state)

        lstm_out = lstm_out[-1, :, :]
        # Dropout and fc layer
        out = self.fc(lstm_out)

        return out, hidden_state

    def step(self, x, y, optimizer, hidden_state):
        """

            :param x:
            :param y:
            :param optimizer:
            :param hidden_state:
            :return:
        """
        optimizer.zero_grad()

        loss, new_hidden_state = self.get_loss(x, y, hidden_state)
        loss.backward()
        optimizer.step()

        return loss, new_hidden_state

    def init_hidden(self, batch_size):
        """

            :param batch_size:
            :return:
        """
        """
            Initializes hidden state.

            :param batch_size:
            :return:
        """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden_state = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                        weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())

        return hidden_state

    def get_loss(self, x, y, hidden_state):
        """

            :param x:
            :param y:
            :param hidden_state:
            :return:
        """
        predicted, new_hidden_state = self.forward(x, hidden_state)

        # Weighted MSE Loss
        loss = nn.functional.mse_loss(predicted, y.float())

        return loss.float(), new_hidden_state
