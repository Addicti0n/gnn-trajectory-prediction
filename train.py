import torch
from data_builder import POS_SCALE, MAP_SIZE_M, split_by_scene
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate
from map_utils import offroad_penalty
import random

dataset = torch.load('dataset_trainval.pt', weights_only=False)
train_set, val_set, test_set = split_by_scene(dataset)
print(f'Graphs: {len(train_set)}/{len(val_set)}/{len(test_set)}')

model = TrajectoryPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 15 #45
LAMBDA = 0.0 #5.0        # βαρος off-road penalty
GOAL_W = 1.0        # βαρος goal auxiliary loss
best_minade = float('inf')

for epoch in range(EPOCHS):
    model.train()
    random.shuffle(train_set)
    total_loss = 0.0

    for node_features, future, mask, edges, penalty_map, _ in train_set:
        optimizer.zero_grad()
        means, scales, weights, goals = model(node_features, edges)
        mdn = laplacian_mdn_loss(means, scales, weights, future, mask)
        offroad = offroad_penalty(means.reshape(-1, 2) * POS_SCALE, penalty_map, MAP_SIZE_M)
        valid = mask[:, -1] == 1
        if valid.any():
            gt_end = future[valid, -1, :]
            g      = goals[valid]
            d      = ((g - gt_end.unsqueeze(1))**2).sum(-1)
            goal_loss = d.min(dim=1).values.mean()
        else:
            goal_loss = torch.zeros((), device=means.device)

        loss = mdn + LAMBDA*offroad + GOAL_W*goal_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(train_set)
    r = evaluate(model, val_set)

    tag = ''
    if r['minADE'] < best_minade:
        best_minade = r['minADE']
        torch.save(model.state_dict(), f'model_lambda{LAMBDA:g}.pth')
        tag = ' SAVE (BEST!)'

    print(f"epoch {epoch}: loss={avg:.2f} | ADE={r['minADE']:.2f}m | FDE={r['minFDE']:.2f}m | "
          f"MR={r['MR']*100:.0f}% | off={r['offroad']*100:.0f}% | ADE_noego={r['minADE_noego']:.2f}m{tag}")

# --- Τελικη αξιολογηση στο TEST (μια φορα!) ---
model.load_state_dict(torch.load(f'model_lambda{LAMBDA:g}.pth'))
r = evaluate(model, test_set)
print(f"--- TEST (unbiased) --- ADE={r['minADE']:.2f}m | FDE={r['minFDE']:.2f}m | "
      f"MR={r['MR']*100:.1f}% | off-road={r['offroad']*100:.1f}%")
print(f"    (no ego)           ADE={r['minADE_noego']:.2f}m | FDE={r['minFDE_noego']:.2f}m | "
      f"MR={r['MR_noego']*100:.1f}% | off-road={r['offroad_noego']*100:.1f}%")
