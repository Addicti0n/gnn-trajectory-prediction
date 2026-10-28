import torch
from data_builder import POS_SCALE, MAP_SIZE_M
from map_utils import offroad_flags

MISS_THRESH = 2.0     # μετρα: miss αν ΚΑΝΕΝΑ mode δεν μενει ολο μεσα στα 2m (max pointwise, επισημο nuScenes)

@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    all_ade, all_fde, all_off, all_isego, all_static, all_gt_off,all_maxd = [], [], [], [], [], [], []

    for node_features, future, mask, edges, penalty_map, _ in dataset:
        means, scales, weights, _ = model(node_features, edges)
        K, T = means.shape[1], means.shape[2]

        # (2.3) ΙΔΙΟΣ παρονομαστης παντου: μονο οχηματα με ΠΛΗΡΕΣ μελλον
        full = (mask.sum(dim=1) == 12)
        if full.sum() == 0:
            continue

        m  = means[full]                    # [M,K,T,2]
        gt = future[full].unsqueeze(1)      # [M,1,T,2]
        M  = m.shape[0]

        dist = torch.norm(m - gt, dim=-1)                    # [M,K,T]
        all_ade.append(dist.mean(dim=2).min(dim=1).values)   # [M] best-of-K
        all_fde.append(dist[:, :, -1].min(dim=1).values)     # [M]
        #original nuscenes MR rules (all spots of the mode must be <2 to be corret)
        worst = dist.max(dim=2).values        # [M, K]  χειρότερο σημείο κάθε mode
        best_worst = worst.min(dim=1).values  # [M]     καλύτερο mode
        all_maxd.append(best_worst)

        # off-road στα ΙΔΙΑ οχηματα (συνεπης παρονομαστης)
        trajs = m.reshape(-1, T, 2) * POS_SCALE              # [M*K,T,2]
        all_off.append(offroad_flags(trajs, penalty_map, MAP_SIZE_M).view(M, K))
        #GT in meters
        gt_m= future[full] * POS_SCALE   #[M,12,2]
        #static vehicles (parked)
        moved= torch.norm(gt_m[:,-1] - gt_m[:,0],dim=-1) # [M]
        all_static.append(moved<1.0)

        #OFF-ROAD OF GT
        all_gt_off.append(offroad_flags(gt_m,penalty_map,MAP_SIZE_M))

        # (2.2) ποιος ειναι το ego -> κομβος 0
        is_ego = torch.zeros(mask.shape[0], dtype=torch.bool)
        is_ego[0] = True
        all_isego.append(is_ego[full])                       # [M]


    ade   = torch.cat(all_ade) * POS_SCALE      # -> μετρα
    fde   = torch.cat(all_fde) * POS_SCALE
    off   = torch.cat(all_off)                  # [total_M, K]
    isego = torch.cat(all_isego)
    noego = ~isego
    static = torch.cat(all_static)
    gt_off = torch.cat(all_gt_off)
    maxd = torch.cat(all_maxd) * POS_SCALE            # → μέτρα

    moving = ~static

    return {
        'minADE':  ade.mean().item(),
        'minFDE':  fde.mean().item(),
        'MR':      (maxd > MISS_THRESH).float().mean().item(),    # (2.5)
        'offroad': off.mean().item(),
        'minADE_noego':  ade[noego].mean().item(),               # (2.2)
        'minFDE_noego':  fde[noego].mean().item(),
        'MR_noego':      (maxd[noego] > MISS_THRESH).float().mean().item(),
        'offroad_noego': off[noego].mean().item(),

                # --- μόνο ΚΙΝΟΥΜΕΝΑ (οι «δύσκολες ερωτήσεις») ---
        'minADE_moving':  ade[moving].mean().item(),
        'minFDE_moving':  fde[moving].mean().item(),
        'MR_moving':      (maxd[moving] > MISS_THRESH).float().mean().item(),
        'offroad_moving': off[moving].mean().item(),
        'MR_fde':  (fde > MISS_THRESH).float().mean().item(),    # παλιός ορισμός (μόνο τελικό σημείο)

        # --- το «πάτωμα»: off-road της ΠΡΑΓΜΑΤΙΚΟΤΗΤΑΣ ---
        'offroad_gt':        gt_off.mean().item(),
        'offroad_gt_moving': gt_off[moving].mean().item(),

    }
