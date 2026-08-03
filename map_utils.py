from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
from pyquaternion import Quaternion
def get_drivable_mask(nusc,sample,size_m=60,canvas=200):
    #find loc (map name) of this sample
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    location = log['location']

    #load the map!
    nmap = NuScenesMap(dataroot='data/raw',map_name=location)

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
