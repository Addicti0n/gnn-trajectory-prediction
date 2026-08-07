import torch
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset, POS_SCALE,MAP_SIZE_M
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate
from map_utils import offroad_penalty


nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)
dataset = build_dataset(nusc)
split = int(0.8*len(dataset)); train_set = dataset[:split]; val_set = dataset[split:]


def train_one(LAMBDA,epochs=30):
    model = TrajectoryPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best = (float('inf'),0,0) #(minAde,minFDE,off road) best epoch
    for epoch in range(epochs):
        for nf,fut,mask,edges,pmap in train_set:
            opt.zero_grad()
            means,scales,weights = model(nf,edges)
            mdn = laplacian_mdn_loss(means,scales,weights,fut,mask)
            off = offroad_penalty(means.reshape(-1,2)*POS_SCALE,pmap,MAP_SIZE_M)
            (mdn + LAMBDA*off).backward()
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