import torch
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset, POS_SCALE,MAP_SIZE_M,split_by_scene   
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate
from map_utils import offroad_penalty
import random

nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)
#dataset = build_dataset(nusc)
dataset = torch.load('dataset_trainval.pt', weights_only=False)

train_set, val_set, test_set = split_by_scene(dataset)
GOAL_W=1.0 #WEIGHT GOAL AUXILIARY LOSS

def train_one(LAMBDA,epochs=10):
    model = TrajectoryPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best = (float('inf'),0,0) #(minAde,minFDE,off road) best epoch
    for epoch in range(epochs):
        random.shuffle(train_set)
        for nf,fut,mask,edges,pmap, _ in train_set:
            opt.zero_grad()
            means,scales,weights,goals = model(nf,edges)
            mdn = laplacian_mdn_loss(means,scales,weights,fut,mask)
            off = offroad_penalty(means.reshape(-1,2)*POS_SCALE,pmap,MAP_SIZE_M)
            goal_loss = ((goals - means[:, :, -1, :])**2).mean()
            (mdn + LAMBDA*off + GOAL_W*goal_loss).backward()
            opt.step()
        ade,fde,offroad=evaluate(model,val_set)
        if ade<best[0]:
            best=(ade,fde,offroad)
    return best


print(f'{"LAMBDA":>7} | {"minADE":>7} | {"minFDE":>7} | {"off-road":>8}')
print('-'*42)
for L in [0, 1, 2, 5, 10]:
    ade, fde, off = train_one(float(L))
    print(f'{L:>7} | {ade:>6.2f}m | {fde:>6.2f}m | {off*100:>6.1f}%')
    import sys; sys.stdout.flush()   