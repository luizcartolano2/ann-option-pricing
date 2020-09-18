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
