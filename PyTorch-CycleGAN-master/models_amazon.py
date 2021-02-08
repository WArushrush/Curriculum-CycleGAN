import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.hiddens = nn.ModuleList([nn.Linear(5000, 1000), nn.LeakyReLU(),
                                      nn.Linear(1000, 500), nn.LeakyReLU(),
                                      nn.Linear(500, 100), nn.LeakyReLU(),
                                      nn.Linear(100, 500), nn.LeakyReLU(),
                                      nn.Linear(500, 1000), nn.LeakyReLU(),
                                      nn.Linear(1000, 5000)])

    def forward(self, x):
        for hidden in self.hiddens:
            x = hidden(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hiddens = nn.ModuleList([nn.Linear(5000, 1000), nn.LeakyReLU(),
                                      nn.Linear(1000, 500), nn.LeakyReLU(),
                                      nn.Linear(500, 100), nn.LeakyReLU(),
                                      nn.Linear(100, 2)])

    def forward(self, x):
        for hidden in self.hiddens:
            x = hidden(x)
        return F.log_softmax(x, dim=1)


class CMSS(nn.Module):
    def __init__(self):
        super(CMSS, self).__init__()
        self.hiddens = nn.ModuleList([nn.Linear(5000, 1000), nn.LeakyReLU(),
                                      nn.Linear(1000, 500), nn.LeakyReLU(),
                                      nn.Linear(500, 100), nn.LeakyReLU(),
                                      nn.Linear(100, 1)])

    def forward(self, x):
        for hidden in self.hiddens:
            x = hidden(x)
        return x
