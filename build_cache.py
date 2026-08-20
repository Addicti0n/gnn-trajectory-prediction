import torch
from nuscenes.nuscenes import NuScenes
from data_builder import build_dataset

VERSION = 'v1.0-trainval'

OUT='dataset_trainval.pt'

print(f'Build {VERSION} est:20-40min....')

nusc=NuScenes(version=VERSION,dataroot='data/raw',verbose=False)
dataset = build_dataset(nusc)
print(f'{len(dataset)} Graphs. Save -> {OUT}')
torch.save(dataset,OUT)
print('DONE!!!!')