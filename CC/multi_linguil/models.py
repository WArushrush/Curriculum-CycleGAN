import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.hiddens = nn.ModuleList([nn.Linear(1200, 600), nn.LeakyReLU(),
                                      nn.Linear(600, 300), nn.LeakyReLU(),
                                      nn.Linear(300, 150), nn.LeakyReLU(),
                                      nn.Linear(150, 300), nn.LeakyReLU(),
                                      nn.Linear(300, 600), nn.LeakyReLU(),
                                      nn.Linear(600, 1200)])

    def forward(self, x):
        for hidden in self.hiddens:
            x = hidden(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hiddens = nn.ModuleList([nn.Linear(1200, 600), nn.LeakyReLU(),
                                      nn.Linear(600, 300), nn.LeakyReLU(),
                                      nn.Linear(300, 150), nn.LeakyReLU(),
                                      nn.Linear(150, 2)])

    def forward(self, x):
        for hidden in self.hiddens:
            x = hidden(x)
        return F.log_softmax(x, dim=1)


class CMSS(nn.Module):
    def __init__(self):
        super(CMSS, self).__init__()
        self.hiddens = nn.ModuleList([nn.Linear(1200, 600), nn.LeakyReLU(),
                                      nn.Linear(600, 300), nn.LeakyReLU(),
                                      nn.Linear(300, 150), nn.LeakyReLU(),
                                      nn.Linear(150, 1)])

    def forward(self, x):
        for hidden in self.hiddens:
            x = hidden(x)
        return x
