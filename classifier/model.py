import torch
from torch import autograd, nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers,
                     num_dir, batch_size, emb_dim,
                     dropout, net_type, prevecs=None, embfix=False):
        super().__init__()
        self.num_layers = num_layers
        self.num_dir = num_dir
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.emd_dim = emb_dim
        self.emb = nn.Embedding(input_size, emb_dim)
        self.net_type = net_type
        self.num_class = num_classes

        if embfix:
            self.emb.weight.requires_grad=False
        if prevecs is not None:
            self.emb.weight = nn.Parameter(prevecs)
        if net_type == 'lstm':
            self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers,
                                    batch_first=True,bidirectional=(num_dir==2),
                                    dropout=dropout)
        elif net_type == 'gru':
            self.gru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers,
                                    batch_first=True,bidirectional=(num_dir==2),
                                    dropout=dropout)
        #self.TanH = nn.TanH(hidden_size*num_dir*num_layers, num_classes)
        self.softmax = None
        self.Lin = nn.Linear(hidden_size*num_dir*num_layers, num_classes)


    def get_ch(self,size):
        hx = autograd.Variable(torch.FloatTensor(self.num_layers*self.num_dir,
                                                size, self.hidden_size).zero_())
        cx = autograd.Variable(torch.FloatTensor(self.num_layers*self.num_dir,
                                                size, self.hidden_size).zero_())
        if int(torch.cuda.is_available()) == 1:
            hx.data = hx.data.cuda()
            cx.data = cx.data.cuda()
        return (hx,cx)

    def forward(self, inp):
        hc = self.get_ch(inp.size(0))
        e = self.emb(inp)
        #e = inp
        if self.net_type == 'lstm':
            _, (y,_) = self.lstm(e, hc)
        elif self.net_type == 'gru':
            _, y = self.gru(e, hc[0])
        if self.num_dir==2:
            y = torch.cat([y[0:y.size(0):2], y[1:y.size(0):2]], 2)
        if self.num_layers>1:
            y = torch.cat([y[i].unsqueeze(0) for i in range(self.num_layers)],2)
        y = torch.squeeze(y,0)
        #z = self.TanH(y)
        self.softmax = nn.Softmax(y)
        return self.Lin(y)
