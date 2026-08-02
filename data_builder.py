import os
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


POS_SCALE = 25.0 #NORMALIZATION OF POSITIONS FOR EASIER LEARNING 
#(IN EVALUATION REMEMBER TO MULTIPLY AGAIN WITH X25 IN ORDER TO HAVE THE ACTUAL METERS!)

# ---  Ιστορικό του Δικού μας Οχήματος ---
def get_ego_history(nusc, current_sample, ego_translation, inverse_quat, frames_back=4):
    trajectory = []
    curr_sample_token = current_sample['token']
    
    for _ in range(frames_back + 1): # +1 για να πάρουμε και το τωρινό (t=0)
        if curr_sample_token == '':
            # Αν δεν υπάρχει πιο πίσω (πχ ξεκίνησε τώρα η σκηνή), αντιγράφουμε την τελευταία γνωστή θέση
            trajectory.append(trajectory[-1] if len(trajectory) > 0 else [0.0, 0.0])
            continue
            
        sample = nusc.get('sample', curr_sample_token)
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Agent-Centric Μετασχηματισμός (σε σχέση με το ΠΟΥ ΕΙΜΑΣΤΕ ΤΩΡΑ)
        global_pos = np.array(ego_pose['translation'])
        shifted = global_pos - ego_translation
        local_pos = inverse_quat.rotate(shifted)
        
        trajectory.append([local_pos[0], local_pos[1]])
        curr_sample_token = sample['prev'] # Πάμε ένα καρέ πίσω!
        
    # Επειδή τα μαζέψαμε ανάποδα (t=0, t-1, t-2...), τα γυρνάμε στη σωστή σειρά
    trajectory.reverse() 
    return trajectory



# ---  mellon του Δικού μας Οχήματος ---
def get_ego_future(nusc, current_sample, ego_translation, inverse_quat, frames_forward=12):
    trajectory = []
    mask = []
    curr_sample_token = current_sample['next']
    
    for _ in range(frames_forward): 
        if curr_sample_token == '':
            trajectory.append(trajectory[-1] if len(trajectory) > 0 else [0.0, 0.0])
            mask.append(0.0)
            continue
            
        sample = nusc.get('sample', curr_sample_token)
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        global_pos = np.array(ego_pose['translation'])
        shifted = global_pos - ego_translation
        local_pos = inverse_quat.rotate(shifted)
        
        trajectory.append([local_pos[0], local_pos[1]])
        mask.append(1.0)
        curr_sample_token = sample['next'] # Πάμε ένα καρέ μπροστα!

    return trajectory,mask



# --- ΒΟΗΘΗΤΙΚΗ ΣΥΝΑΡΤΗΣΗ 2: Ιστορικό των Άλλων Οχημάτων ---
def get_target_history(nusc, ann_token, ego_translation, inverse_quat, frames_back=4):
    trajectory = []
    curr_ann_token = ann_token
    
    for _ in range(frames_back + 1):
        if curr_ann_token == '':
            trajectory.append(trajectory[-1] if len(trajectory) > 0 else [0.0, 0.0])
            continue
            
        ann = nusc.get('sample_annotation', curr_ann_token)
        
        global_pos = np.array(ann['translation'])
        shifted = global_pos - ego_translation
        local_pos = inverse_quat.rotate(shifted)
        
        trajectory.append([local_pos[0], local_pos[1]])
        curr_ann_token = ann['prev']
        
    trajectory.reverse()
    return trajectory

#-- get target future  --
def get_target_future(nusc, ann_token, ego_translation, inverse_quat, frames_forward=12):
    trajectory = []
    mask = []
    ann = nusc.get('sample_annotation',ann_token)
    curr_ann_token = ann['next'] #<-- t+1
    
    for _ in range(frames_forward):
        if curr_ann_token == '':
            #to oxima eksafanistike h teleiwse h skini padding
            trajectory.append(trajectory[-1] if len(trajectory) > 0 else [0.0, 0.0])
            mask.append(0.0)
            continue
            
        ann = nusc.get('sample_annotation', curr_ann_token)
        
        global_pos = np.array(ann['translation'])
        shifted = global_pos - ego_translation
        local_pos = inverse_quat.rotate(shifted)
        
        trajectory.append([local_pos[0], local_pos[1]])
        mask.append(1.0)
        curr_ann_token = ann['next']
        
    return trajectory,mask



#creating the scene
def get_scene_sample_tokens(nusc, scene):
    tokens = []
    token = scene['first_sample_token']   # το πρώτο sample
    
    while token != '':
     
        tokens.append(token)
        token = nusc.get('sample',token)['next']
    return tokens




def build_graph_from_sample(nusc,sample):
   
    # 1.find where we are
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    inverse_quat = Quaternion(ego_pose['rotation']).inverse

    # 2. a list of nodes not just [x,y]  but 5 time spots
    nodes_data = []
    future_data=[]
    future_mask_data=[]
    ego_fut,ego_mask = get_ego_future(nusc,sample,ego_translation,inverse_quat)
    future_data.append(ego_fut)
    future_mask_data.append(ego_mask)
    # get our history (node 0)
    ego_traj = get_ego_history(nusc, sample, ego_translation, inverse_quat)
    nodes_data.append(ego_traj)

    #get history loc (nodes) of the other cars
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if 'vehicle' in ann['category_name']:
            target_global = np.array(ann['translation'])
            shifted = target_global - ego_translation
            target_local = inverse_quat.rotate(shifted)
            
            if abs(target_local[0]) < 50 and abs(target_local[1]) < 50:
                target_traj = get_target_history(nusc, ann_token, ego_translation, inverse_quat)
                nodes_data.append(target_traj)

                target_fut,target_mask = get_target_future(nusc,ann_token,ego_translation,inverse_quat)
                future_data.append(target_fut)
                future_mask_data.append(target_mask)
    # 3. TENSORINGGGGGGGGGG
    node_features = torch.tensor(nodes_data, dtype=torch.float32)
    num_nodes = node_features.shape[0]

    #create the FUTUREE
    future_features = torch.tensor(future_data,dtype=torch.float32)
    future_mask = torch.tensor(future_mask_data,dtype=torch.float32)

    
    # 4. Ακμές (Όπως ακριβώς πριν)
    sources = [i for i in range(num_nodes) for j in range(num_nodes) if i != j]
    targets = [j for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    node_features = node_features / POS_SCALE
    future_features  = future_features  / POS_SCALE
    return node_features,future_features,future_mask,edge_index



def build_dataset(nusc):
    HIST, FUT = 4, 12
    all_graphs = []
    
    for scene in nusc.scene:                          # για κάθε σκηνή
        tokens = get_scene_sample_tokens(nusc, scene) # όλα τα samples της
        anchors = tokens[HIST : len(tokens) - FUT]    # μόνο οι έγκυρες άγκυρες
        
        for anchor_token in anchors:                  # για κάθε άγκυρα
            # get anchor obj
            sample =   nusc.get('sample',anchor_token)

            # build da graph
            graph = build_graph_from_sample(nusc,sample)
            # add to all_graphs
            all_graphs.append(graph)
    
    return all_graphs



if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataroot = os.path.join(current_dir, 'data', 'raw')
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
    
    dataset = build_dataset(nusc)
    print(f"✅ Έτοιμα {len(dataset)} γράφοι!")