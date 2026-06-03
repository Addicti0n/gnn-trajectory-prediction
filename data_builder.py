import os
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

# --- ΒΟΗΘΗΤΙΚΗ ΣΥΝΑΡΤΗΣΗ 1: Ιστορικό του Δικού μας Οχήματος ---
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

def build_spatiotemporal_graph():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataroot = os.path.join(current_dir, 'data', 'raw')
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
    
    scene = nusc.scene[0]
    sample = nusc.get('sample', scene['first_sample_token'])
    
    # 1. Βρίσκουμε τη βάση μας (Πού είμαστε ΣΗΜΕΡΑ)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    inverse_quat = Quaternion(ego_pose['rotation']).inverse

    print("⏳ Σάρωση του χάρτη και εξαγωγή Ιστορικού (Ταξίδι στον Χρόνο)...")

    # 2. Φτιάχνουμε τους Κόμβους (Τώρα δεν είναι απλά [x,y], αλλά μια λίστα με 5 χρονικά σημεία!)
    nodes_data = []
    
    # Α. Παίρνουμε το δικό μας ιστορικό (Κόμβος 0)
    ego_traj = get_ego_history(nusc, sample, ego_translation, inverse_quat)
    nodes_data.append(ego_traj)

    # Β. Παίρνουμε το ιστορικό των άλλων (Κόμβοι 1, 2, 3...)
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if 'vehicle' in ann['category_name']:
            target_global = np.array(ann['translation'])
            shifted = target_global - ego_translation
            target_local = inverse_quat.rotate(shifted)
            
            if abs(target_local[0]) < 50 and abs(target_local[1]) < 50:
                target_traj = get_target_history(nusc, ann_token, ego_translation, inverse_quat)
                nodes_data.append(target_traj)

    # 3. Μετατροπή σε PyTorch Tensor
    node_features = torch.tensor(nodes_data, dtype=torch.float32)
    num_nodes = node_features.shape[0]

    print("-" * 60)
    print(f"📍 Κόμβοι: {num_nodes} οχήματα")
    print(f"Διαστάσεις Πίνακα (Tensor Shape): {node_features.shape}")
    print("Εξήγηση Shape: [Αριθμός Οχημάτων, Χρονικά Βήματα, (X, Y)]")
    print("-" * 60)
    
    # Ας δούμε την πορεία του Κόμβου 0 (Του εαυτού μας)
    print("Τροχιά (Trajectory) του δικού μας οχήματος τα τελευταία 2 δευτερόλεπτα:")
    for t in range(5):
        print(f"Στιγμή t-{4-t}: X={node_features[0][t][0]:.2f}, Y={node_features[0][t][1]:.2f}")
    
    # 4. Ακμές (Όπως ακριβώς πριν)
    sources = [i for i in range(num_nodes) for j in range(num_nodes) if i != j]
    targets = [j for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    print("-" * 60)
    print("✅ Ο Χωρο-Χρονικός Γράφος (Spatial-Temporal Graph) είναι 100% Έτοιμος!")

if __name__ == '__main__':
    build_spatiotemporal_graph()