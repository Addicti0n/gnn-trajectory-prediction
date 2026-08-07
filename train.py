import torch
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset, POS_SCALE,MAP_SIZE_M
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate
from map_utils import offroad_penalty

nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)
dataset = build_dataset(nusc)
split = int(0.8 * len(dataset))
train_set = dataset[:split] #first part 80%
val_set = dataset[split:] #second part 20%
print(f'Train: {len(train_set)} graphs | val: {len(val_set)} graphs')
model = TrajectoryPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # μικρότερο LR για σταθερότητα
EPOCHS = 30
LAMBDA = 5.0 #WEIGHT parameter for OFF-ROAD PENALTY
best_minade = float('inf') #find the best minADE (starts from infinity!!)
for epoch in range(EPOCHS):
    total_loss = 0.0
    for node_features, future, mask, edges, penalty_map in train_set:
        optimizer.zero_grad()
        means, scales, weights = model(node_features, edges)
        mdn = laplacian_mdn_loss(means, scales, weights, future, mask)
        offroad = offroad_penalty(means.reshape(-1,2)* POS_SCALE,penalty_map,MAP_SIZE_M)
        loss = mdn + LAMBDA * offroad
        loss.backward()
        optimizer.step()                 #χωρίς αυτό δεν μαθαίνει
        total_loss += loss.item()        #για σωστό avg

    avg = total_loss / len(train_set)

    min_ade, min_fde, offroad = evaluate(model,val_set) #evaluate every epoch
 
    tag = ''
    if min_ade<best_minade:
        best_minade=min_ade
        torch.save(model.state_dict(), f'model_lambda{int(LAMBDA)}.pth')

        tag= ' SAVE (BEST!)'
    print(f'epoch {epoch}: train loss={avg:.2f} | minADE={min_ade:.2f}m | minFDE={min_fde:.2f}m | off-road={offroad*100:.1f}%{tag}')


print(f'Best val minADE: {best_minade:.2f}m -> model_lambda{int(LAMBDA)}.pth')
