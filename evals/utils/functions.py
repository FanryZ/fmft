import torch
import numpy as np
import torch.nn.functional as F
import math

def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid

    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

'''
将 2D 图像坐标（加上深度）反投影（back-project）成物体自身坐标系下的 3D 点坐标
kp2d: shape = (N, 2)，是图像平面上的 2D 点（u, v）坐标。
depth: shape = (H, W)，对应视角下的深度图，单位是米。
k: shape = (3, 3)，相机内参矩阵。
pose_obj2cam: shape = (4, 4)，从物体坐标系到相机坐标系的变换矩阵，即 extrinsic matrix
x                u
y = D(u,v)⋅K^−1 ⋅ v
z                1
'''
def img_coord_2_obj_coord(kp2d, depth, k, pose_obj2cam):
    inv_k = np.linalg.inv(k[:3, :3]) # K^-1
    kp2d = kp2d[:, :2] # 去除多余列
    kp2d = np.concatenate((kp2d, np.ones((kp2d.shape[0], 1))), 1) # 将 (u,v) 转化为 (u, v, 1)
    # 将像素坐标四舍五入为整数并转换为 int 类型，用于图像索引。注意深度图是二维数组 (H, W)，需整数索引
    kp2d_int = np.round(kp2d).astype(int)[:, :2]
    kp_depth = depth[kp2d_int[:, 1], kp2d_int[:, 0]]  # num
    kp2d_cam = np.expand_dims(kp_depth, 1) * kp2d  # num, 3
    kp3d_cam = np.dot(inv_k, kp2d_cam.T).T  # num, 3
    # 把 (x, y, z) 转换为 (x, y, z, 1)，以便应用 4x4 齐次变换矩阵
    kp3d_cam_pad1 = np.concatenate((kp3d_cam, np.ones((kp2d_cam.shape[0], 1))), 1).T  # 4, num (4, N)
    # 乘以 pose_cam2obj = inv(pose_obj2cam)，将 3D 点从相机坐标系变换回物体本地坐标系
    kp3d_obj = np.dot(np.linalg.inv(pose_obj2cam), kp3d_cam_pad1).T  # num, 4
    return kp3d_obj[:, :3] # 返回 3D 点坐标


# dino patch size is even, so the pixel corner is not really aligned, potential improvements here, 
# borrowed from DINO-Tracker
# 只针对关键点进行插值
'''
DINO 使用 Vision Transformer（ViT），它把图像分成 patch（如 14×14），每个 patch 被投影成一个 token。
但这些 patch 特征图并不是 pixel-aligned（因为 patch size 是偶数），所以若希望从一个精确的像素点坐标获得其特征，
需要用双线性插值（bilinear sampling） 方式。
descriptors：patch 级 feature map
pts：关键点坐标
h, w：原始图像像素大小
normalize：对结果进行 L2 归一化
'''
def interpolate_features(descriptors, pts, h, w, normalize=True, patch_size=14, stride=14):
    # patch token 中心在像素坐标上的最大值
    last_coord_h = ( (h - patch_size) // stride ) * stride + (patch_size / 2)
    last_coord_w = ( (w - patch_size) // stride ) * stride + (patch_size / 2)
    ah = 2 / (last_coord_h - (patch_size / 2))
    aw = 2 / (last_coord_w - (patch_size / 2))
    bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
    bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))
    # 把关键点从像素坐标空间转换到 [-1, 1] 的网格坐标空间，匹配 F.grid_sample 的要求。
    a = torch.tensor([[aw, ah]]).to(pts).float()
    b = torch.tensor([[bw, bh]]).to(pts).float()
    keypoints = a * pts + b
    # Expand dimensions for grid sampling
    # F.grid_sample 要求 grid 输入形状为 [B, H_out, W_out, 2]，所以这里把关键点作为一行多个点。
    keypoints = keypoints.unsqueeze(-3)  # Shape becomes [batch_size, 1, num_keypoints, 2]
    # Interpolate using bilinear sampling
    interpolated_features = F.grid_sample(descriptors, keypoints, align_corners=True, padding_mode='border')
    # interpolated_features will have shape [batch_size, channels, 1, num_keypoints]
    interpolated_features = interpolated_features.squeeze(-2)
    return F.normalize(interpolated_features, dim=1) if normalize else interpolated_features

# pose 的辅助函数
def query_pose_error(pose_pred, pose_gt, unit='m'):
    """
    Input:
    -----------
    pose_pred: np.array 3*4 or 4*4
    前三列为旋转矩阵，后一列为平移向量
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]
    # Convert results' unit to cm
    # 平移向量欧氏距离
    if unit == 'm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    elif unit == 'cm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3])
    elif unit == 'mm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) / 10
    else:
        raise NotImplementedError
    # 从真实旋转到预测旋转的相对旋转
    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    # 计算旋转差矩阵的迹（矩阵对角元素之和），迹的值范围理论上是[-1, 3]。
    trace = np.trace(rotation_diff)
    # 由于数值误差可能导致迹大于3，这里限定最大值为3，避免数值问题
    trace = trace if trace <= 3 else 3
    # 旋转矩阵 R 的迹满足关系：trace(R) = 1 + 2 * cos(θ)，其中θ是旋转角度。所以 θ = arccos((trace(R) - 1) / 2)
    # np.arccos 返回弧度，np.rad2deg 转换成角度。这个角度即是预测旋转与真实旋转之间的旋转误差，单位是度。
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance

# 用于 semantic transfer
def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding, the location of key points needs to be updated.
    # This function applies that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    # 对关键点坐标进行变换（图像经过尺寸调整之后）
    # kps：kps: shape 为 (20, 3) 的关键点张量，每个关键点是 (x, y, mask)，其中 mask 为 0 或 1。
    kps = kps.clone()
    # img_width, img_height：原始宽高
    # size：模型要求输入尺寸
    scale = size / max(img_width, img_height)
    # 将关键点 (x, y) 根据图像缩放比例同步缩放
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        # 上下均匀填充（只改变 y 坐标）
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        # 左右均匀填充（只改变 x 坐标）
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    # zero-out any non-visible key points
    kps *= kps[:, 2:3].clone()  
    return kps

# video tracking 辅助函数
# 返回一个方法，然后会被绑定到 ViT 模型上以替换默认的插值策略
'''
在 ViT 中，图像被切分为 patch（例如 14x14），每个 patch 会加上一个位置编码（positional embedding）表示空间信息。
但：ViT 预训练时只支持固定尺寸（如 224x224 → 16x16 patch grid）；（224=16x14）
实际推理时可能输入的是 476x854，这个尺寸不是预训练时的 patch grid。
于是：需要将预训练时的 Positional Embedding 插值到新的空间尺寸。
'''
def _fix_pos_enc(patch_size, stride_hw):
    '''
    x: 当前输入的 patch token 序列（shape 为 [B, num_tokens, C]
    w, h: 当前输入图像的尺寸（以 patch 空间计，不是像素）
    '''
    def interpolate_pos_encoding(self, x, w, h):
        # 去除 class token
        # 当前输入图像的 patch token
        npatch = x.shape[1] - 1
        # 预训练的 patch token
        N = self.pos_embed.shape[1] - 1
        # 尺寸没变
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        # 计算特征图大小
        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                        stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    return interpolate_pos_encoding