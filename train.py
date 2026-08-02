import torch
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate

nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)
dataset = build_dataset(nusc)
split = int(0.8 * len(dataset))
train_set = dataset[:split] #first part 80%
val_set = dataset[split:] #second part 20%
print(f'Train: {len(train_set)} graphs | val: {len(val_set)} graphs')
model = TrajectoryPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # μικρότερο LR για σταθερότητα
EPOCHS = 30
for epoch in range(EPOCHS):
    total_loss = 0.0
    for node_features, future, mask, edges in train_set:
        optimizer.zero_grad()
        means, scales, weights = model(node_features, edges)
        loss = laplacian_mdn_loss(means, scales, weights, future, mask)
        loss.backward()
        optimizer.step()                 # ⬅️ ΠΡΟΣΘΕΣΕ — χωρίς αυτό δεν μαθαίνει!
        total_loss += loss.item()        # ⬅️ ΠΡΟΣΘΕΣΕ — για σωστό avg

    avg = total_loss / len(dataset)
    if epoch % 5 == 0 or epoch == EPOCHS-1:
        min_ade,min_fde = evaluate(model,val_set)
        print(f'epoch {epoch}: train loss = {avg:.2f} | val minADE={min_ade:2f}m | minFDE={min_fde:.2f}m')
    else:
        print(f'epoch {epoch}: train loss={avg:.2f}')

