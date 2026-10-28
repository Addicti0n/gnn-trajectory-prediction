import torch
import matplotlib.pyplot as plt

from data_builder import split_by_scene,  POS_SCALE, MAP_SIZE_M
from model import TrajectoryPredictor
import sys                                                        # στην κορυφη


dataset = torch.load('dataset_trainval.pt', weights_only=False)
_, _, test_set = split_by_scene(dataset)          # ΜΟΝΟ test (τίμιο για figure)

path = sys.argv[1] if len(sys.argv) > 1 else 'model_lambda5.pth'
model = TrajectoryPredictor()
model.load_state_dict(torch.load(path))
model.eval()

node_features,future,mask,edges,penalty_map, _ = test_set[0] 
with torch.no_grad():
    means,scales,weights,goals = model(node_features,edges)

#choose our vehicle and de-normalize (we want meters)
agent=0

hist = node_features[agent, :, :2] * POS_SCALE    # [5, 2] μόνο x, y

gt = future[agent] * POS_SCALE
preds = means[agent] * POS_SCALE
w = weights[agent]

#VISUALIZATION
plt.figure(figsize=(8,8))

half = MAP_SIZE_M / 2
drivable = 1 - penalty_map
plt.imshow(drivable,cmap='Greys',origin='lower',extent=[-half,half,-half,half],alpha=0.25)

plt.plot(hist[:, 0], hist[:, 1], 'o-', color='dimgray', label='Ιστορικο')
plt.plot(gt[:, 0], gt[:, 1], '*-', color='black', linewidth=3, markersize=12, label='Αληθεια (GT)')
for k in range(preds.shape[0]):
    plt.plot(preds[k, :, 0], preds[k, :, 1], '--', alpha=0.7, label=f'Pred {k} (p={w[k]:.2f})')

plt.legend(fontsize=8); plt.axis('equal'); plt.title('Goal-Conditioned + Map-Aware')
plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.savefig('prediction.png', dpi=100)
print('prediction.png SAVED!!')