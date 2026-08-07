import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch 
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
        self.goal_head = nn.Linear(in_channels,num_modes * out_dim) #predicts k goals

        # means/scales παιρνουν (embedding + goal) και βγαζουν 1 τροχια ανα mode
        self.fc_mean = nn.Linear(in_channels + out_dim, future_len * out_dim)
        self.fc_scale = nn.Linear(in_channels + out_dim, future_len * out_dim) 
        self.fc_weight = nn.Linear(in_channels ,num_modes)

    def forward(self,x):
        # x: [N,128]
        N = x.shape[0]
        K,T,D = self.num_modes, self.future_len, self.out_dim


        goals = self.goal_head(x).view(N,K,D) #δλδ [N,K,2] οι στοχοι
        x_exp = x.unsqueeze(1).expand(-1,K,-1) #[N,K,128] αντιγραφο ανα mode
        cond = torch.cat([x_exp,goals], dim=-1) #[N,K,130] EMBEDDING +GOAL
        mean = self.fc_mean(cond).view(N,K,T,D)
        scales = F.softplus(self.fc_scale(cond)).view(N,K,T,D)
        weights = F.softmax(self.fc_weight(x),dim=-1) #dim = -1 in order to normalize last dim k modes
        return mean,scales,weights


################### FULL MODEL ################
class TrajectoryPredictor(nn.Module):
    def __init__(self,hidden_size=128,num_modes=6,future_len=12):
        super().__init__()
        self.encoder = GRUEncoder(input_size=2, hidden_size=hidden_size)
        self.interactor = GATInteractor(in_channels=hidden_size, out_channels=hidden_size)
        self.decoder = LaplacianMDNDecoder(in_channels=hidden_size, num_modes=num_modes, future_len=future_len)

    def forward(self,x,edge_index):
        #x:[N,5,2], edge_index: [2,E]
        emb = self.encoder(x)
        enriched = self.interactor(emb,edge_index)
        means,scales,weights = self.decoder(enriched)
        return means,scales,weights