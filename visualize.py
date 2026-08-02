import torch
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset, POS_SCALE
from model import TrajectoryPredictor

nusc = NuScenes(version='v1.0-mini',dataroot='data/raw',verbose=False)
dataset= build_dataset(nusc)
model = TrajectoryPredictor()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
node_features,future,mask,edges = dataset[-1]
with torch.no_grad():
    means,scales,weights = model(node_features,edges)

#choose our vehicle and de-normalize (we want meters)
agent=0
hist = node_features[agent] * POS_SCALE
gt = future[agent] * POS_SCALE
preds = means[agent] * POS_SCALE
w = weights[agent]

#VISUALIZATION
plt.figure(figsize=(8,8))
plt.plot(hist[:,0],hist[:,1], 'ko-', label='History (παρελθον)')
plt.plot(gt[:,0],gt[:,1], 'g*-', 'b-',linewidth=2,label='Future')
for k in range(preds.shape[0]):

    plt.plot(preds[k,:,0],preds[k,:,1], '--', alpha=0.6, label=f'Pred {k} (π={w[k]:.2f})')
plt.legend(); plt.axis('equal');plt.title('Trajectory Prediction (ego)')
plt.savefig('prediction.png',dpi=100)
print('prediction.png Saved!!!')