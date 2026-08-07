import torch, sys
from statistics import mean, stdev
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset, POS_SCALE, MAP_SIZE_M
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate
from map_utils import offroad_penalty

nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)
dataset = build_dataset(nusc)
split = int(0.8*len(dataset)); train_set = dataset[:split]; val_set = dataset[split:]

def train_one(LAMBDA, seed, epochs=30):
    torch.manual_seed(seed)                    # ← σταθερό seed για επαναληψιμότητα
    model = TrajectoryPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best = (float('inf'), 0, 0)
    for epoch in range(epochs):
        for nf, fut, mask, edges, pmap in train_set:
            opt.zero_grad()
            means, scales, weights = model(nf, edges)
            mdn = laplacian_mdn_loss(means, scales, weights, fut, mask)
            off = offroad_penalty(means.reshape(-1,2)*POS_SCALE, pmap, MAP_SIZE_M)
            (mdn + LAMBDA*off).backward()
            opt.step()
        ade, fde, offroad = evaluate(model, val_set)
        if ade < best[0]: best = (ade, fde, offroad)
    return best

SEEDS = 5
for L in [0.0, 5.0]:
    ades, offs = [], []
    for s in range(SEEDS):
        ade, fde, off = train_one(L, s)
        ades.append(ade); offs.append(off*100)
        print(f'  lambda={L} seed={s}: minADE={ade:.2f}m off-road={off*100:.1f}%'); sys.stdout.flush()
    print(f'==> lambda={L}: minADE {mean(ades):.2f} +/- {stdev(ades):.2f}m | off-road {mean(offs):.1f} +/- {stdev(offs):.1f}%\n'); sys.stdout.flush()
