from metric_learn import ITML, MMC
# from metric_learn import ITML
import torch
import torch.nn.functional as nn_F
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from PIL import Image


def metric_learning_finetune(feats_0, feats_1, neg_pair_num = 2):
    """
    Args:
        feats_0 (np.ndarray): [N, C]
        feats_1 (np.ndarray): [N, C]
    """
    feats = np.concatenate([feats_0, feats_1], axis=0)
    pos_pair = []
    for i in range(feats_0.shape[0]):
        pos_pair.append([i, i + feats_0.shape[0]])
    neg_pair = []
    for i in range(feats_0.shape[0] * neg_pair_num):
        neg_1 = np.random.randint(0, feats_0.shape[0] - 2)
        neg_2 = np.random.randint(neg_1 + 1, feats_0.shape[0] - 1) + feats_0.shape[0]
        neg_pair.append([neg_1, neg_2])
    y = [1] * len(pos_pair) + [-1] * len(neg_pair)
    pairs = np.concatenate([np.array(pos_pair), np.array(neg_pair)], axis=0)

    model = ITML(preprocessor=feats, max_iter=20, verbose=True)
    model.fit(pairs, y)
    return model, model.pair_score(pairs)


def metric_learning_finetune_outlier(feats_0, feats_1, outlier_feats0, outlier_feats1, neg_pair_num = 2):
    """
    Args:
        feats_0 (np.ndarray): [N, C]
        feats_1 (np.ndarray): [N, C]
        outlier_feats0 (np.ndarray): [N, C]
        outlier_feats1 (np.ndarray): [N, C]
    """
    feats = np.concatenate([feats_0, feats_1, outlier_feats0, outlier_feats1], axis=0)
    pos_pair = []
    for i in range(feats_0.shape[0]):
        pos_pair.append([i, i + feats_0.shape[0]])
    neg_pair = []
    for i in range(feats_0.shape[0] * neg_pair_num):
        neg_1 = np.random.randint(0, outlier_feats0.shape[0]) + feats_0.shape[0] * 2
        neg_2 = np.random.randint(0, feats_1.shape[0]) + feats_0.shape[0]
        neg_pair.append([neg_1, neg_2])
    for i in range(feats_0.shape[0] * neg_pair_num):
        neg_1 = np.random.randint(0, outlier_feats1.shape[0]) + feats_0.shape[0] * 2 + outlier_feats0.shape[0]
        neg_2 = np.random.randint(0, feats_0.shape[0])
        neg_pair.append([neg_1, neg_2])
    y = [1] * len(pos_pair) + [-1] * len(neg_pair)
    pairs = np.concatenate([np.array(pos_pair), np.array(neg_pair)], axis=0)

    model = ITML(preprocessor=feats, max_iter=10, verbose=True)
    model.fit(pairs, y)
    return model, model.pair_score(pairs)


def metric_learning_distance(feats_0, feats_1, points_0, points_1, neg_pair_num=2, min_distance_ratio=0.3):
    """
    Args:
        feats_0 (np.ndarray): [N, C] features from first image
        feats_1 (np.ndarray): [N, C] features from second image
        points_0 (np.ndarray): [N, 2] 2D points from first image (normalized to [0,1]x[0,1])
        points_1 (np.ndarray): [N, 2] 2D points from second image (normalized to [0,1]x[0,1])
        neg_pair_num: number of negative pairs per positive pair
        min_distance_ratio: minimum distance ratio (relative to image diagonal) for negative pairs
    """
    assert len(feats_0) == len(feats_1) == len(points_0) == len(points_1)
    num_points = len(feats_0)
    
    # Combine features for the metric learning model
    feats = np.concatenate([feats_0, feats_1], axis=0)
    
    # Calculate image diagonal for distance normalization
    img_diagonal = np.linalg.norm([1.0, 1.0])  # Since points are in [0,1]x[0,1]
    min_distance = min_distance_ratio * img_diagonal
    
    # Positive pairs (matched points)
    pos_pairs = np.column_stack((np.arange(num_points), 
                               np.arange(num_points) + num_points))
    
    # Generate negative pairs
    neg_pairs = []
    for i in range(num_points):
        # Get the corresponding point in the second image
        corresponding_pt = points_1[i]
        # Calculate distances from the corresponding point to all other points in the second image
        distances = np.linalg.norm(points_1 - corresponding_pt, axis=1)
        # Find points that are sufficiently far from the corresponding point
        far_indices = np.where(distances > min_distance)[0]
        # Remove the current point's index if it's in the far_indices
        far_indices = far_indices[far_indices != i]
        # If we have enough far points, sample from them
        if len(far_indices) > 0:
            # Sample negative pairs
            selected = np.random.choice(far_indices, 
                                      size=min(neg_pair_num, len(far_indices)),
                                      replace=False)
            for idx in selected:
                # Create negative pair: point i from first image with far point from second image
                neg_pairs.append([i, num_points + idx])  # i is in first image, idx is in second image
    
    # Convert to numpy arrays
    pos_pairs = np.array(pos_pairs)
    neg_pairs = np.array(neg_pairs) if len(neg_pairs) > 0 else np.empty((0, 2), dtype=int)
    # Combine positive and negative pairs
    pairs = np.vstack([pos_pairs, neg_pairs]) if len(neg_pairs) > 0 else pos_pairs
    y = np.array([1] * len(pos_pairs) + [-1] * len(neg_pairs))
    # Train metric learning model
    model = ITML(preprocessor=feats, max_iter=20, verbose=True)
    model.fit(pairs, y)
    # Calculate score (e.g., accuracy on training pairs)
    distances = model.pairwise_distance(pairs)
    predictions = np.where(distances < model.threshold_, 1, -1)
    score = np.mean(predictions == y)
    return model, score


def outliers(points_masked, image_size, patch_size, margin=1):
    h, w = image_size 
    patch_points = (points_masked / patch_size).int()
    margined_patch_points = []
    for i in range(-margin, margin + 1):
        for j in range(-margin, margin + 1):
            margined_patch_points.append(patch_points + torch.tensor([i, j]).to(patch_points))
    points = torch.unique(
        torch.cat(
            margined_patch_points, 
            dim=0
        ), 
        dim=0
    )
    points_set = set(map(tuple, points.cpu().numpy()))
    h_patch = h // patch_size
    w_patch = w // patch_size
    # all_points = torch.meshgrid(
    #     torch.arange(h_patch), torch.arange(w_patch), indexing="xy"
    # )
    all_points = torch.meshgrid(
        torch.arange(w_patch), torch.arange(h_patch), indexing="ij"
    )
    all_points = torch.stack(all_points, dim=-1).view(-1, 2)
    all_points_set = set(map(tuple, all_points.numpy()))
    outlier_points = all_points_set - points_set

    outlier_points = torch.tensor(list(outlier_points), dtype=torch.float32, device=points.device)
    outlier_points = outlier_points * patch_size
    # outlier_points = outlier_points * patch_size + patch_size / 2
    return outlier_points
    

def interpolate_features(feature, pts, image_size, patch_size=14):
    """
        Args:
        feature (torch.Tensor): [N, H, W, C]
        pts (torch.Tensor): [N, N_p, 2]
        image_size (tuple): (2, )
        Return:
        interpolated_feature: [N, N_p, C]
    """
    h, w = image_size
    stride = patch_size
    last_coord_h = ( (h - patch_size) // stride ) * stride + (patch_size / 2)
    last_coord_w = ( (w - patch_size) // stride ) * stride + (patch_size / 2)
    ah = 2 / (last_coord_h - (patch_size / 2))
    aw = 2 / (last_coord_w - (patch_size / 2))
    bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
    bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))

    pts = pts[:, None, ...]
    pts_grid = torch.zeros_like(pts).float()
    pts_grid[:, :, :, 0] = pts[:, :, :, 0] * aw + bw
    pts_grid[:, :, :, 1] = pts[:, :, :, 1] * ah + bh
    # descs = nn_F.grid_sample(feature, pts_grid, "bilinear", align_corners=False)
    descs = nn_F.grid_sample(feature, pts_grid, "bilinear", align_corners=True, padding_mode="border")
    descs = descs[:, :, 0, :].permute(0, 2, 1)
    return descs


def conv_training(feats_0: torch.Tensor, feats_1: torch.Tensor, points_0: torch.Tensor, points_1: torch.Tensor, 
                 outlier_points_0: torch.Tensor, outlier_points_1: torch.Tensor, image_size: tuple, 
                 neg_pair_num: int = 2, iter_num: int = 100, temperature: float = 0.1, weight_decay: float = 0.0001) -> torch.nn.Module:
    """
    Train a 1x1 convolutional layer to align features between two views.
    
    Args:
        feats_0: [N, C, H, W] features from view 0
        feats_1: [N, C, H, W] features from view 1
        points_0: [N, num_points, 2] corresponding points in view 0
        points_1: [N, num_points, 2] corresponding points in view 1
        outlier_points_0: [N, num_outliers, 2] non-matching points in view 0
        outlier_points_1: [N, num_outliers, 2] non-matching points in view1
        image_size: (H, W) size of the input images
        neg_pair_num: number of negative pairs per positive pair
        iter_num: number of training iterations
        
    Returns:
        Trained 1x1 convolutional layer
    """
    conv_layer = torch.nn.Conv2d(in_channels=feats_0.shape[1], 
                               out_channels=feats_1.shape[1], 
                               kernel_size=3, stride=1, padding=1, bias=False).to(feats_0.device)
    # proj_layer = InvertibleLinear(feats_0.shape[1]).to(feats_0.device)
    # conv_layer.weight.data.fill_(1)

    optimizer = torch.optim.Adam(
        params=conv_layer.parameters(), lr=0.0001, weight_decay=weight_decay
    )
    
    # Get number of positive pairs and available negative points
    pos_pair_num = points_0.shape[1]
    neg_point_num = min(outlier_points_0.shape[1], outlier_points_1.shape[1])
    # Create eye mask for positive pairs
    eye_mask = torch.eye(pos_pair_num, device=feats_0.device) > 0
    pbar = tqdm(range(iter_num), desc="Training")
    for _ in pbar:
        optimizer.zero_grad()
        
        # Forward pass through conv layer
        feats_0_transformed = conv_layer(feats_0)
        feats_1_transformed = conv_layer(feats_1)
        
        # Get features at corresponding points
        interp_feats_0 = nn_F.normalize(interpolate_features(feats_0_transformed, points_0, image_size), dim=2)
        interp_feats_1 = nn_F.normalize(interpolate_features(feats_1_transformed, points_1, image_size), dim=2)

        # Compute similarity matrices
        sim_matrix = interp_feats_0 @ interp_feats_1.permute(0, 2, 1)  # [N, P, P]
        pos_sim = sim_matrix[0][eye_mask]  # [P]
        
        if temperature == 0:
            loss = -torch.log(pos_sim).mean()
        else:
            # Get features at outlier points for negative pairs
            outlier_feats_0 = nn_F.normalize(interpolate_features(feats_0_transformed, outlier_points_0, image_size), dim=2)
            outlier_feats_1 = nn_F.normalize(interpolate_features(feats_1_transformed, outlier_points_1, image_size), dim=2)
            
            # Sample negative pairs
            neg_idx = torch.randperm(neg_point_num, device=feats_0.device)[:neg_pair_num]
            neg_feats_0 = outlier_feats_0[:, neg_idx]
            neg_feats_1 = outlier_feats_1[:, neg_idx]
        
            # Compute negative similarities
            neg_sim_0 = (neg_feats_1 @ interp_feats_0.permute(0, 2, 1))[0]  # [neg, P]
            neg_sim_1 = (neg_feats_0 @ interp_feats_1.permute(0, 2, 1))[0]  # [neg, P]
            
            # Compute softmax probabilities
            sim_mat0 = torch.softmax(torch.cat((pos_sim[None], neg_sim_0), dim=0) / temperature, dim=0)  # [1+neg, P]
            sim_mat1 = torch.softmax(torch.cat((pos_sim[None], neg_sim_1), dim=0) / temperature, dim=0)  # [1+neg, P]
            
            # Compute loss (negative log likelihood of positive pairs)
            loss = -torch.log(sim_mat0[0]).mean() - torch.log(sim_mat1[0]).mean()

        loss.backward(retain_graph=True)
        pbar.set_postfix(loss=loss.item())
        optimizer.step()
    
    return conv_layer


def viz_feat(feat, file_path):
    """
    Args:
        feat: [N, C, H, W]
        file_path: str
    """
    _, _, h, w = feat.shape
    feat = feat.squeeze(0).permute((1,2,0))
    projected_featmap = feat.reshape(-1, feat.shape[-1]).cpu()

    pca = PCA(n_components=3)
    pca.fit(projected_featmap)
    pca_features = pca.transform(projected_featmap)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    res_pred = Image.fromarray(pca_features.reshape(h, w, 3).astype(np.uint8))
    res_pred.save(file_path)
    print("... saved to: ", file_path)
