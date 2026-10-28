import torch
import matplotlib.pyplot as plt
from data_builder import POS_SCALE, MAP_SIZE_M, split_by_scene
from model import TrajectoryPredictor

dataset = torch.load('dataset_trainval.pt', weights_only=False)
_, _, test_set = split_by_scene(dataset)          # ΜΟΝΟ test (τίμιο για figure)

def load(path):
    m = TrajectoryPredictor()
    m.load_state_dict(torch.load(path))
    m.eval()
    return m

baseline = load('model_lambda0.pth')
novelty  = load('model_lambda5.pth')

node_features, future, mask, edges, penalty_map, _ = test_set[100]
agent = 0

                                   
hist = node_features[agent, :, :2] * POS_SCALE    # [5, 2] μόνο x, y

gt = future[agent] * POS_SCALE
half = MAP_SIZE_M / 2
drivable = 1 - penalty_map

fig, axes = plt.subplots(1,2, figsize=(16,8))

for ax, model, title in [(axes[0], baseline, 'Baseline λ=0  (off-road 100%, ADE 7.5m)'),
                         (axes[1], novelty,  'Map-Aware λ=5  (off-road 0%, ADE 3.0m)')]:


    with torch.no_grad():
        means,scales,weights,goals = model(node_features,edges)
    preds = means[agent] * POS_SCALE
    ax.imshow(drivable,cmap='Greys',origin='lower',extent=[-half,half,-half,half], alpha=0.25)
    ax.plot(hist[:,0], hist[:,1], 'o-',color='dimgray',label='Past (ιστορικο)')
    ax.plot(gt[:,0], gt[:,1], '-', color='black', linewidth=3, alpha=0.85,
        label='Ground Truth', zorder=3)                    # γραμμή, όχι μεγάλες κουκκίδες
    for k in range(preds.shape[0]):
        ax.plot(preds[k,:,0], preds[k,:,1], '--', alpha=0.9, linewidth=1.6, zorder=4)   # ΠΑΝΩ από το GT

    ax.set_title(title); ax.legend(fontsize=8); ax.axis('equal')
    ax.set_xlim(-half,half); ax.set_ylim(-half,half)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

plt.suptitle('Baseline vs Map-Aware - same scene')
plt.savefig('comparison.png',dpi=100)
print('comparison.png SAVED!')