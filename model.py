import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GRUEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=128,num_layers=1):
        super().__init__() #gia to nn.module
        self.gru = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)

    def forward(self,x):
        output, h_n = self.gru(x)
        return h_n[-1]
    

class GATInteractor(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, heads=4):
        super().__init__()
        self.gat = GATConv(in_channels,out_channels,heads=heads,concat=False)

    def forward(self,x,edge_index):
        # x = [N,128] ta embeddings & edge index [2,E] grafos me geitonika oximata
        return self.gat(x,edge_index)