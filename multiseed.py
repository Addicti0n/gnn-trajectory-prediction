import torch, sys
from statistics import mean, stdev
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset, POS_SCALE, MAP_SIZE_M,split_by_scene   
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

def train_one(LAMBDA, seed, epochs=120):
    torch.manual_seed(seed)     
    random.seed(seed)               # ← σταθερό seed για επαναληψιμότητα
    model = TrajectoryPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    best = (float('inf'), 0, 0)
    for epoch in range(epochs):
        random.shuffle(train_set)
        for nf, fut, mask, edges, pmap, _ in train_set:
            opt.zero_grad()
            means, scales, weights,goals = model(nf, edges)
            mdn = laplacian_mdn_loss(means, scales, weights, fut, mask)
            off = offroad_penalty(means.reshape(-1,2)*POS_SCALE, pmap, MAP_SIZE_M)
            valid = mask[:, -1] == 1
            if valid.any():
                gt_end = fut[valid, -1, :]
                g      = goals[valid]
                d      = ((g - gt_end.unsqueeze(1))**2).sum(-1)
                goal_loss = d.min(dim=1).values.mean()
            else:
                goal_loss = torch.zeros((), device=means.device)

            (mdn + LAMBDA*off + GOAL_W*goal_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # ← ΝΕΟ

            opt.step()
        r = evaluate(model, val_set)
        ade, fde, offroad = r['minADE'], r['minFDE'], r['offroad']

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
