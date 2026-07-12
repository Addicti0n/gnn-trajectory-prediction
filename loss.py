import torch

def laplacian_mdn_loss(means,scales,weights,target,mask):
    #means:[N,K,T,2], scales:[N,K,T,2] weights:[N,K], target: [N,T,2] mask:[N,T]

    #1. Broadcasting θελουμε target & mask να ταιριαζουν με τα K mods
    target = target.unsqueeze(1) #target -> [N,1,T,2]
    mask = mask.unsqueeze(1).unsqueeze(-1) #mask - > [N,1,T,1]

    #2. laplacian log density ανα coordinates

    log_prob = -torch.abs(target - means) / scales - torch.log(2*scales)

    #3. apply mask & αθροισμα σε χρονο T=dim2 & coordinates D=dim3
    log_prob_per_mode = (log_prob * mask).sum(dim=(2,3))
     
    #4. sum log(π) & logsumexp στα modes -> log p(Y)
    log_weights = torch.log(weights + 1e-9)  # [N,K]  (+1e-9 = ασφάλεια στο log(0))
    log_p = torch.logsumexp(log_weights + log_prob_per_mode,dim=1)

    #5. negative log-likelihood mean on vehicles
    loss = -log_p.mean()
    return loss