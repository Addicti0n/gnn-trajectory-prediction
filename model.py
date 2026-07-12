import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

################### GRU ENCODER ################
class GRUEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=128,num_layers=1):
        super().__init__() #gia to nn.module
        self.gru = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)

    def forward(self,x):
        output, h_n = self.gru(x)
        return h_n[-1]
    
################### GAT Interactor ################

class GATInteractor(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, heads=4):
        super().__init__()
        self.gat = GATConv(in_channels,out_channels,heads=heads,concat=False)

    def forward(self,x,edge_index):
        # x = [N,128] ta embeddings & edge index [2,E] grafos me geitonika oximata
        return self.gat(x,edge_index)

################### LAPLACIAN MDN DECODER ################
class LaplacianMDNDecoder(nn.Module):
    def __init__(self,in_channels=128,num_modes=6,future_len=12,out_dim=2):
        super().__init__()
        self.num_modes = num_modes
        self.future_len = future_len
        self.out_dim = out_dim
        
        # 3 heads. means & scales βγαζουν K*T*2 αριθμους, το weight βγαζει Κ.
        self.fc_mean = nn.Linear(in_channels, num_modes * future_len * out_dim)
        self.fc_scale = nn.Linear(in_channels, num_modes * future_len * out_dim)
        self.fc_weight = nn.Linear(in_channels,num_modes)

    def forward(self,x):
        # x: [N,128]
        N = x.shape[0]
        K,T,D = self.num_modes, self.future_len, self.out_dim

        mean = self.fc_mean(x).view(N,K,T,D)
        scales = F.softplus(self.fc_scale(x)).view(N,K,T,D)
        weights = F.softmax(self.fc_weight(x),dim=-1) #dim = -1 in order to normalize last dim k modes
        return mean,scales,weights
