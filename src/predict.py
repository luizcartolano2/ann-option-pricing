import numpy as np
import torch


def predict_mlp(input, model):
    """

        :param input:
        :param model:
        :return:
    """
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)

    pred = model.forward(input.float())

    return pred.detach().numpy()


def predict_lstm(input, model):
    """

        :param input:
        :param model:

        :return:
    """
    hidden = model.init_hidden(batch_size=input.shape[0])

    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)

    input = input.view(1, input.shape[0], -1)

    predicted, _ = model.forward(input, hidden)

    return predicted.detach().numpy()
