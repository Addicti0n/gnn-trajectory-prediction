import torch
from data_builder import POS_SCALE,MAP_SIZE_M
from map_utils import offroad_flags

@torch.no_grad() #no gradients στην αξιολογηση (για ΟΛΗ αυτή τη συνάρτηση ΔΕΝ καταγράφεις gradien)
def evaluate(model,dataset):
    all_ade,all_fde,all_offroad = [],[],[]

    for node_features, future, mask, edges, penalty_map, _ in dataset:
        means, scales, weights,goals = model(node_features,edges)
        K, T = means.shape[1], means.shape[2]
        trajs = (means.reshape(-1, T, 2) * POS_SCALE)      # [N*K, T, 2] ολες οι τροχιες
        all_offroad.append(offroad_flags(trajs, penalty_map, MAP_SIZE_M))

        #only vehicles with full future (12 real steps (mask=1))
        full = (mask.sum(dim=1)==12)

        if full.sum()==0:
            continue
    
        m = means[full]
        gt = future[full].unsqueeze(1) #add the 1 in the corresponding k index

        dist = torch.norm(m-gt,dim=-1)
        ade = dist.mean(dim=2)
        min_ade= ade.min(dim=1).values
        fde= dist[:,:,-1]
        min_fde=fde.min(dim=1).values
        all_ade.append(min_ade)
        all_fde.append(min_fde)

    all_ade=torch.cat(all_ade)
    all_fde=torch.cat(all_fde)
    all_offroad = torch.cat(all_offroad)
    return all_ade.mean().item()*POS_SCALE, all_fde.mean().item()*POS_SCALE, all_offroad.mean().item()
