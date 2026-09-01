import torch, sys, random
from statistics import mean, stdev
from data_builder import POS_SCALE, MAP_SIZE_M, split_by_scene
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate
from map_utils import offroad_penalty

dataset = torch.load('dataset_trainval.pt', weights_only=False)
train_set, val_set, test_set = split_by_scene(dataset)

SUBSET = 5000        # υποσυνολο train (not enough)
EPOCHS = 40          # αρκετα για να δρασει ο χαρτης
SEEDS  = 3
GOAL_W = 1.0

def train_one(LAMBDA, seed):
    torch.manual_seed(seed); random.seed(seed)
    tr = train_set[:SUBSET]
    model = TrajectoryPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    r = None
    for epoch in range(EPOCHS):
        model.train(); random.shuffle(tr)
        for nf, fut, mask, edges, pmap, _ in tr:
            opt.zero_grad()
            means, scales, weights, goals = model(nf, edges)
            mdn = laplacian_mdn_loss(means, scales, weights, fut, mask)
            off = offroad_penalty(means.reshape(-1,2)*POS_SCALE, pmap, MAP_SIZE_M)
            valid = mask[:, -1] == 1
            if valid.any():
                gt_end = fut[valid, -1, :]
                d = ((goals[valid] - gt_end.unsqueeze(1))**2).sum(-1)
                goal_loss = d.min(dim=1).values.mean()
            else:
                goal_loss = torch.zeros((), device=means.device)
            (mdn + LAMBDA*off + GOAL_W*goal_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        r = evaluate(model, val_set)     # καθε epoch -> για την καμπυλη
        print(f'  lam={LAMBDA:g} seed={seed} ep={epoch:2d}: '
              f'ADE={r["minADE"]:.2f} FDE={r["minFDE"]:.2f} off={r["offroad"]*100:.1f}%')
        sys.stdout.flush()
    return r                              # ΤΕΛΙΚΟ epoch (οχι best, fair συγκριση)

print(f'SUBSET={SUBSET} EPOCHS={EPOCHS} SEEDS={SEEDS}\n')
for L in [0.0, 5.0]:
    ades, fdes, offs, mrs = [], [], [], []
    for s in range(SEEDS):
        r = train_one(L, s)
        ades.append(r['minADE']); fdes.append(r['minFDE'])
        mrs.append(r['MR']*100); offs.append(r['offroad']*100)
    print(f'==> lambda={L:g}: ADE {mean(ades):.2f}+/-{stdev(ades):.2f}m | '
          f'FDE {mean(fdes):.2f}+/-{stdev(fdes):.2f}m | MR {mean(mrs):.1f}+/-{stdev(mrs):.1f}% | '
          f'off-road {mean(offs):.1f}+/-{stdev(offs):.1f}%\n')
    sys.stdout.flush()
