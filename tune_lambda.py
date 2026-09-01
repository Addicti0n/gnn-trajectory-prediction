import torch
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset, POS_SCALE,MAP_SIZE_M,split_by_scene   
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate
from map_utils import offroad_penalty
import random

#nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)
#dataset = build_dataset(nusc)
dataset = torch.load('dataset_trainval.pt', weights_only=False)

train_set, val_set, test_set = split_by_scene(dataset)
GOAL_W=1.0 #WEIGHT GOAL AUXILIARY LOSS

def train_one(LAMBDA,epochs=15):
    model = TrajectoryPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best = (float('inf'),0,0,0) #(minAde,minFDE,off road) best epoch
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_set)
        for nf,fut,mask,edges,pmap, _ in train_set:
            opt.zero_grad()
            means,scales,weights,goals = model(nf,edges)
            mdn = laplacian_mdn_loss(means,scales,weights,fut,mask)
            off = offroad_penalty(means.reshape(-1,2)*POS_SCALE,pmap,MAP_SIZE_M)
            valid = mask[:, -1] == 1
            if valid.any():
                gt_end = fut[valid, -1, :]
                g      = goals[valid]
                d      = ((g - gt_end.unsqueeze(1))**2).sum(-1)
                goal_loss = d.min(dim=1).values.mean()
            else:
                goal_loss = torch.zeros((), device=means.device)
            (mdn + LAMBDA*off + GOAL_W*goal_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) 

            opt.step()
        r = evaluate(model, val_set)

        if r['minADE'] < best[0]:
            best = (r['minADE'], r['minFDE'], r['MR'], r['offroad'])
    return best


print(f'{"LAMBDA":>7} | {"minADE":>8} | {"minFDE":>8} | {"MR":>6} | {"off-road":>8}')
print('-'*52)
for L in [0, 1, 2, 5, 10]:
    ade, fde, mr, off = train_one(float(L))
    print(f'{L:>7} | {ade:>7.2f}m | {fde:>7.2f}m | {mr*100:>5.1f}% | {off*100:>7.1f}%')
    import sys; sys.stdout.flush()
