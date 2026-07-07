import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=128,num_layers=1):
        super().__init__() #gia to nn.module
        self.gru = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)

    def forward(self,x):
        output, h_n = self.gru(x)
        return h_n[-1]