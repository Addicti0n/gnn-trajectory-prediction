import torch
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset, POS_SCALE,MAP_SIZE_M,split_by_scene   
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss
from evaluate import evaluate
from map_utils import offroad_penalty
import random

#nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)
#dataset = build_dataset(nusc) # OLD
dataset= torch.load('dataset_trainval.pt', weights_only=False)
train_set, val_set, test_set = split_by_scene(dataset)

print(f'Γραφοι: {len(train_set)}/{len(val_set)}/{len(test_set)}')
model = TrajectoryPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # μικρότερο LR για σταθερότητα
EPOCHS = 10 #30
LAMBDA = 5.0 #WEIGHT parameter for OFF-ROAD PENALTY
GOAL_W =1.0 #weight of goal auxiliary loss
best_minade = float('inf') #find the best minADE (starts from infinity!!)
for epoch in range(EPOCHS):
    random.shuffle(train_set)
    total_loss = 0.0
    for node_features, future, mask, edges, penalty_map, _ in train_set:
        optimizer.zero_grad()
        means, scales, weights,goals = model(node_features, edges)
        mdn = laplacian_mdn_loss(means, scales, weights, future, mask)
        offroad = offroad_penalty(means.reshape(-1,2)* POS_SCALE,penalty_map,MAP_SIZE_M)
        goal_loss = ((goals - means[:, :, -1, :])**2).mean()

        loss = mdn + LAMBDA*offroad + GOAL_W*goal_loss


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
# Τελικη αξιολογηση στο TEST (το αγγιζουμε ΜΙΑ φορα!)
model.load_state_dict(torch.load(f'model_lambda{int(LAMBDA)}.pth'))
t_ade, t_fde, t_off = evaluate(model, test_set)
print(f'--- TEST (unbiased) --- minADE={t_ade:.2f}m | minFDE={t_fde:.2f}m | off-road={t_off*100:.1f}%')


