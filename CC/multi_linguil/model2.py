import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

device = torch.device("cuda")


# MAX_LENGTH=106
# SOS_token=55583
# EOS_token=55584

def init_weight(emb, lan, dom):
    f = open("/root/data/" + lan + ".pkl", 'rb')
    language_dict = pickle.load(f)
    f.close()
    f = open("/root/data/xiaoyang/PyTorch-CycleGAN-master/cls-acl10-unprocessed/" + lan + "/" + dom + "/word_dict.pkl",
             'rb')
    word_dict = pickle.load(f)
    f.close()
    for key in language_dict:
        if key in word_dict:
            emb.weight[word_dict[key]] = torch.tensor(language_dict[key])
            # emb.weight[word_dict[key]] = emb.weight[word_dict[key]].detach()


class Emb(nn.Module):
    def __init__(self, language, domain, input_size=55585, hidden_size=300):
        super(Emb, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=input_size-1)
        init_weight(self.embedding, language, domain)
        self.embedding.weight = torch.nn.Parameter(self.embedding.weight.detach())
        self.embedding.weight.requires_grad = False

    def forward(self, input):
        return self.embedding(input)


class Encoder(nn.Module):
    # 参数：input_size为输入语言包含的词个数
    def __init__(self, language, domain, input_size=55585, hidden_size=300):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_size, hidden_size) #每词 hidden_size个属性
        # init_weight(self.embedding, language, domain)
        # self.embedding = self.embedding.to("cuda")
        # self.embedding.weight.requires_grad=False
        # print("grad: ", self.embedding.weight.requires_grad)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.hidden = (torch.rand(4, 16, hidden_size), torch.rand(4, 16, hidden_size))

    def forward(self, input):
        self.lstm.flatten_parameters()
        # embedded = self.embedding(input)
        # print(embedded.shape)
        output, hidden = self.lstm(input)
        return output, hidden


class Decoder(nn.Module):
    # 参数：output_size为输出语言包含的所有单词数
    def __init__(self, language, domain, hidden_size=300, output_size=55585, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        init_weight(self.embedding, language, domain)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, bidirectional=False, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)  # 把256个特征转换成输出语言的词汇个数

    # 参数：input每步输入，hidden上一步结果，encoder_outputs编码的状态矩阵
    # 计算的值是各词出现的概率
    def forward(self, input, hidden):
        self.lstm.flatten_parameters()
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hiddens = nn.ModuleList([nn.Linear(1200, 600), nn.LeakyReLU(),
                                      nn.Linear(600, 300), nn.LeakyReLU(),
                                      nn.Linear(300, 150), nn.LeakyReLU(),
                                      nn.Linear(150, 2)])

    def forward(self, input):
        for hidden in self.hiddens:
            input = hidden(input)
        return F.log_softmax(input, dim=1)
