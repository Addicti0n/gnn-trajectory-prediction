from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
from pyquaternion import Quaternion
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

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


def compute_sdf(mask,meters_per_px,clip_m):
    mask_np = mask.numpy()
    dist_px = distance_transform_edt(mask_np)
    dist_m  = meters_per_px * dist_px
    dist_m = np.clip(dist_m,0,clip_m)
    dist_m = dist_m/clip_m

    return torch.from_numpy(dist_m).float()

def compute_cost_map(mask, meters_per_px, clip_m):
    # mask: [H,W] binary (1 = off-road)
    # returns: [H,W] float in [0,1] = 0.5 * (binary + SDF)
    sdf = compute_sdf(mask,meters_per_px,clip_m)
    return (sdf + mask.float())/2


@torch.no_grad()
def offroad_flags(trajs_xy, penalty_map, size_m, thresh=0.5):
    """
    trajs_xy: [M, T, 2] τροχιες σε ego meters
    Επιστρεφει: [M] με 1.0 αν η τροχια βγαινει εκτος δρομου εστω σε ΕΝΑ σημειο
    """
    M, T, _ = trajs_xy.shape
    half = size_m / 2.0
    grid = (trajs_xy / half).reshape(1, -1, 1, 2)
    inp = penalty_map.float().view(1, 1, *penalty_map.shape)
    sampled = F.grid_sample(inp, grid, mode='nearest',
                            align_corners=False, padding_mode='border')
    sampled = sampled.view(M, T)                    # [M,T] τιμες 0/1 ανα σημειο

     #     ποιες τροχιες εχουν ΕΣΤΩ ΕΝΑ; -> .any(dim=1)
    offroad_any = (sampled > thresh).any(dim=1)

    return offroad_any.float()


def offroad_penalty(points_xy,penalty_map,size_m):
     
    # points_xy: [P,2] σημεια σε ego meters | 
    # penalty_map: [H,W] (1=off-road) | -> μεσο penalty
    half = size_m/ 2.0
    grid = points_xy/half
    grid = grid.view(1,-1,1,2) #reshape gia grid_smple: [1,P,1,2]
    inp = penalty_map.float().view(1,1, *penalty_map.shape) #[1,1,h,w] 
    sampled= F.grid_sample(inp,grid,align_corners=False,padding_mode='border')
    return sampled.mean()


def offroad_penalty_weighted(means_xy, weights, mask, penalty_map, size_m):
    N, K, T, _ = means_xy.shape
    half = size_m / 2.0
    grid = (means_xy / half).view(1, N*K*T, 1, 2)
    inp = penalty_map.float().view(1, 1, *penalty_map.shape)

    sampled = F.grid_sample(inp, grid, align_corners=False, padding_mode='border')
    sampled = sampled.view(N, K, T)

    mask_exp = mask.unsqueeze(1)
    weights_exp = weights.detach().unsqueeze(-1)

    weighted_penalty = sampled * weights_exp * mask_exp

    valid_points = mask.sum()
    if valid_points > 0:
        return weighted_penalty.sum() / valid_points
    return torch.zeros((), device=means_xy.device)



def get_drivable_mask(nusc, sample, size_m=120, canvas=200):     # 60 → 120

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
