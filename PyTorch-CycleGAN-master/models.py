import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.hiddens=nn.ModuleList([nn.Linear(1024,512),nn.LeakyReLU(),
                                    nn.Linear(512,256),nn.LeakyReLU(),
                                    nn.Linear(256,128),nn.LeakyReLU(),
                                    nn.Linear(128,256),nn.LeakyReLU(),
                                    nn.Linear(256,512),nn.LeakyReLU(),
                                    nn.Linear(512,1024)])

    def forward(self, x):
        for hidden in self.hiddens:
            x=hidden(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hiddens=nn.ModuleList([nn.Linear(1024,512),nn.LeakyReLU(),
                                    nn.Linear(512,256),nn.LeakyReLU(),
                                    nn.Linear(256,128),nn.LeakyReLU(),
                                    nn.Linear(128,2)])

    def forward(self, x):
        for hidden in self.hiddens:
            x=hidden(x)
        return F.log_softmax(x,dim=1)
    

class CMSS(nn.Module):
    def __init__(self):
        super(CMSS, self).__init__()
        self.hiddens = nn.ModuleList([nn.Linear(1024,512),nn.LeakyReLU(),
                                    nn.Linear(512,256),nn.LeakyReLU(),
                                    nn.Linear(256,128),nn.LeakyReLU(),
                                    nn.Linear(128,1)])
    def forward(self, x):
        for hidden in self.hiddens:
            x=hidden(x)
        return x
