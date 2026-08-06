from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
from pyquaternion import Quaternion
import torch
import torch.nn.functional as F

## Οταν φτιαχνω 244 γραφους ο καθενας καλει την get_drivable_mask που
## φορτωνει εναν NuScenesMap & η φορτωση χαρτη ειναι αργη
# 244 φορτωσεις = πολλα λεπτα αναμονης
## Για αυτο παμε σε caching (memoization) φορτωνω καθε χαρτη μια φορα
# οι 4 χαρτες φορτωνονται 1 φορα ο καθενας οχι 244

_MAP_CACHE = {}  #μνημη για φορτωμενους χαρτες

def get_map(location):
    if location not in _MAP_CACHE:
        _MAP_CACHE[location] = NuScenesMap(dataroot='data/raw', map_name=location)
    return _MAP_CACHE[location]



def offroad_penalty(points_xy,penalty_map,size_m):
     
    # points_xy: [P,2] σημεια σε ego meters | 
    # penalty_map: [H,W] (1=off-road) | -> μεσο penalty
    half = size_m/ 2.0
    grid = points_xy/half
    grid = grid.view(1,-1,1,2) #reshape gia grid_smple: [1,P,1,2]
    inp = penalty_map.view(1,1, *penalty_map.shape) #[1,1,h,w] 
    sampled= F.grid_sample(inp,grid,align_corners=False,padding_mode='border')
    return sampled.mean()

def get_drivable_mask(nusc,sample,size_m=60,canvas=200):
    #find loc (map name) of this sample
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    location = log['location']

    #load the map!
    nmap = get_map(location)                                     # ΝΕΟ (από την cache)

    #find the location of our vehicle (global x,y) (using lidar because it has ego_pose)
    lidar = nusc.get('sample_data',sample['data']['LIDAR_TOP'])
    ego = nusc.get('ego_pose', lidar['ego_pose_token'])
    x,y=ego['translation'][0],ego['translation'][1]
    yaw = Quaternion(ego['rotation']).yaw_pitch_roll[0] #gwnia strofis radians
    patch_angle = np.degrees(yaw) 
    mask = nmap.get_map_mask((x,y,size_m,size_m),patch_angle,['drivable_area'],canvas_size=(canvas,canvas))
    return mask[0]


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)
    sample = nusc.get('sample', nusc.scene[0]['first_sample_token'])
    m = get_drivable_mask(nusc, sample)
    print('mask shape:', m.shape, '| drivable %:', round(100*m.sum()/m.size))
