import torch
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset
from model import TrajectoryPredictor
from loss import laplacian_mdn_loss

nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)
dataset = build_dataset(nusc)

model = TrajectoryPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # μικρότερο LR για σταθερότητα
EPOCHS = 30
for epoch in range(EPOCHS):
    total_loss = 0.0
    for node_features, future, mask, edges in dataset:
        optimizer.zero_grad()
        means, scales, weights = model(node_features, edges)
        loss = laplacian_mdn_loss(means, scales, weights, future, mask)
        loss.backward()
        optimizer.step()                 # ⬅️ ΠΡΟΣΘΕΣΕ — χωρίς αυτό δεν μαθαίνει!
        total_loss += loss.item()        # ⬅️ ΠΡΟΣΘΕΣΕ — για σωστό avg

    avg = total_loss / len(dataset)
    print(f'epoch {epoch}: avg loss = {avg:.3f}')

