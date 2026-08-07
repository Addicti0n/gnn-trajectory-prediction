import torch
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset, POS_SCALE,MAP_SIZE_M
from model import TrajectoryPredictor

nusc = NuScenes(version='v1.0-mini',dataroot='data/raw',verbose=False)
dataset= build_dataset(nusc)

#load both models
def load(path):
    m= TrajectoryPredictor()
    m.load_state_dict(torch.load(path))
    m.eval()
    return m

baseline = load('model_lambda0.pth')
novelty = load('model_lambda5.pth')

#both scene and vehicle for both
node_features, future, mask, edges, penalty_map = dataset[-1]

agent = 0
hist= node_features[agent] * POS_SCALE
gt = future[agent] * POS_SCALE
half = MAP_SIZE_M / 2
drivable = 1 - penalty_map

fig, axes = plt.subplots(1,2, figsize=(16,8))
for ax, model, title in [(axes[0],baseline,'Baseline (LAMBDA=0)'),
                         (axes[1],novelty,'Map-Aware (LAMBDA=5)')]:
    with torch.no_grad():
        means,scales,weights = model(node_features,edges)
    preds = means[agent] * POS_SCALE
    ax.imshow(drivable,cmap='Greys',origin='lower',extent=[-half,half,-half,half], alpha=0.25)
    ax.plot(hist[:,0], hist[:,1], 'o-',color='dimgray',label='Past (ιστορικο)')
    ax.plot(gt[:,0], gt[:,1], 'o-',color='black',linewidth=3,markersize=12,label='Ground Truth)')
    for k in range(preds.shape[0]):
        ax.plot(preds[k,:,0],preds[k,:,1],'--',alpha=0.7)
    ax.set_title(title); ax.legend(fontsize=8); ax.axis('equal')
    ax.set_xlim(-half,half); ax.set_ylim(-half,half)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

plt.suptitle('Baseline vs Map-Aware - same scene')
plt.savefig('comparison.png',dpi=100)
print('comparison.png SAVED!')