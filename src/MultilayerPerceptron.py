import torch
import torch.nn as nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, layer1_size=400, layer2_size=400,
                 layer3_size=400, layer4_size=400, output_size=1):
        """

            :param input_size:
            :param layer1_size:
            :param layer2_size:
            :param layer3_size:
            :param layer4_size:
            :param output_size:
        """
        super(MultilayerPerceptron, self).__init__()

        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.layer3_size = layer3_size
        self.layer4_size = layer4_size
        self.output_size = output_size

        # hidden layers
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size, self.layer1_size),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(self.layer1_size, self.layer2_size),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(self.layer2_size, self.layer3_size),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(self.layer3_size, self.layer4_size),
            nn.LeakyReLU()
        )

        # output layer
        self.output = nn.Linear(self.layer4_size, self.output_size)

    def forward(self, x):
        """

            :param x:
            :return:
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.output(x)

        return output

    def step(self, x, y, optimizer):
        optimizer.zero_grad()

        loss = self.get_loss(x, y)
        loss.backward()
        optimizer.step()

        return loss

    def get_loss(self, x, y):
        predicted = self.forward(x.float())

        # Weighted MSE Loss
        loss = nn.functional.mse_loss(predicted, y.float())

        return loss.float()
