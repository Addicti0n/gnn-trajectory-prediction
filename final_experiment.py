import torch, sys, random
from statistics import mean, stdev
from data_builder import POS_SCALE, MAP_SIZE_M, split_by_scene
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate
from map_utils import offroad_penalty

dataset = torch.load('dataset_trainval.pt', weights_only=False)
train_set, val_set, test_set = split_by_scene(dataset)

SUBSET = 5000
EPOCHS = 40
SEEDS  = 5
GOAL_W = 1.0

def train_one(LAMBDA, seed):
    torch.manual_seed(seed); random.seed(seed)
    sub = list(train_set)              
    random.Random(seed).shuffle(sub)     
    tr = sub[:SUBSET]           

    model = TrajectoryPredictor()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
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
        if epoch % 5 == 0:                                  # progress καθε 5 epochs
            v = evaluate(model, val_set)
            print(f'   lam={LAMBDA:g} seed={seed} ep={epoch:2d}: '
                  f'val ADE={v["minADE"]:.2f} off={v["offroad"]*100:.1f}%'); sys.stdout.flush()
    return evaluate(model, test_set)                        # ΤΕΛΙΚΟ, στο TEST

print(f'FINAL EXPERIMENT: SUBSET={SUBSET} EPOCHS={EPOCHS} SEEDS={SEEDS} (metrics on TEST)\n')
results = {}
for L in [0.0, 5.0]:
    acc = {'minADE': [], 'minFDE': [], 'MR': [], 'offroad': [],
           'minADE_noego': [], 'offroad_noego': []}
    for s in range(SEEDS):
        r = train_one(L, s)
        for k in acc: acc[k].append(r[k])
        print(f'  -> lam={L:g} seed={s} TEST: ADE={r["minADE"]:.2f} FDE={r["minFDE"]:.2f} '
              f'MR={r["MR"]*100:.1f}% off={r["offroad"]*100:.1f}%'); sys.stdout.flush()
    results[L] = acc
    print(f'==> lambda={L:g} (TEST, n={SEEDS}): '
          f'ADE {mean(acc["minADE"]):.2f}+/-{stdev(acc["minADE"]):.2f}m | '
          f'FDE {mean(acc["minFDE"]):.2f}+/-{stdev(acc["minFDE"]):.2f}m | '
          f'MR {mean(acc["MR"])*100:.1f}+/-{stdev(acc["MR"])*100:.1f}% | '
          f'off-road {mean(acc["offroad"])*100:.1f}+/-{stdev(acc["offroad"])*100:.1f}%\n')
    sys.stdout.flush()

# --- Στατιστικος ελεγχος ---
from scipy import stats
print('Welch t-test (lambda=0 vs lambda=5) στο TEST:')
for k, name in [('minADE','minADE'), ('minFDE','minFDE'), ('MR','MR'), ('offroad','off-road')]:
    t, p = stats.ttest_ind(results[0.0][k], results[5.0][k], equal_var=False)
    sig = 'ΣΗΜΑΝΤΙΚΟ' if p < 0.05 else ('ταση' if p < 0.10 else 'οχι σημαντικο')
    print(f'  {name:9s}: t={t:+.2f}  p={p:.4f}  -> {sig}')
