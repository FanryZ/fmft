import torch
import numpy as np

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = None
model = None

def load_model():
    global model, dtype
    if model == None:
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def patch_track(image_names, patch_size=14):
    if model == None:
        load_model()
    # image_names = ["/data/fanry/Desktop/fmft/locft/sample/0a76e06478_DSC03620.JPG", 
    #             "/data/fanry/Desktop/fmft/locft/sample/0a76e06478_DSC03640.JPG"]
    size_w, size_h = Image.open(image_names[0]).size
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            _, n, c, h, w = images.shape
            aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        # point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

        query_x, query_y = torch.meshgrid([torch.arange(patch_size // 2, h, patch_size), 
                                           torch.arange(patch_size // 2, w, patch_size)], indexing="xy")
        # query_x, query_y = patch_size * query_x, patch_size * query_y
        # query_x, query_y = patch_size * query_x + patch_size / 2, patch_size * query_y + patch_size / 2
        query_points_ori = torch.cat([query_x.reshape(-1, 1), query_y.reshape(-1, 1)], axis = 1)
        query_points = query_points_ori.to(device)
        track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])
    return images, track_list, vis_score, conf_score
        
def track_point_filter(track_list, vis_score, conf_score, filter_ratio=0.5, min_corr=None):
    # filter_ratio = 0.5
    source_points = track_list[-1][0][0]
    target_points, target_conf, target_vis = track_list[-1][0][1], conf_score[0][1], vis_score[0][1]
    conf_thres = torch.quantile(target_conf, 1 - filter_ratio)
    vis_thres = torch.quantile(target_vis, 0.8)
    point_mask = (target_conf > conf_thres) & (target_vis > vis_thres)
    source_points_masked = source_points[point_mask, :]
    target_points_masked = target_points[point_mask, :]
    # if min_corr is not None:
    #     point_mask = point_mask & (target_conf > min_corr)
    return source_points_masked, target_points_masked

    # grid_x, grid_y = torch.meshgrid(indexing="xy")


def vis_track_points(images, source_points_masked, target_points_masked):
    """
    Visualize point matching between two images by drawing lines between matched points.
    Save the visualization as 'point_matching.png'.
    """
    images_np = images[0].cpu().numpy()  # shape: (n, c, h, w)
    n, c, h, w = images_np.shape
    assert n == 2, "This function expects exactly two images."

    # Convert images to PIL
    pil_images = []
    for i in range(n):
        img = images_np[i].transpose(1, 2, 0)  # (h, w, c)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        pil_images.append(Image.fromarray(img))

    # Create a new image side by side
    concat_width = w * 2
    concat_height = h
    concat_img = Image.new('RGB', (concat_width, concat_height))
    concat_img.paste(pil_images[0], (0, 0))
    concat_img.paste(pil_images[1], (w, 0))
    draw = ImageDraw.Draw(concat_img)

    # Draw lines between matched points
    src_pts = source_points_masked.cpu().numpy()
    tgt_pts = target_points_masked.cpu().numpy()
    for (x1, y1), (x2, y2) in zip(src_pts, tgt_pts):
        # Draw circles at the points
        r = 4
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline="red", width=2)
        draw.ellipse((x2 + w - r, y2 - r, x2 + w + r, y2 + r), outline="blue", width=2)
        # Draw line connecting the points
        draw.line([(x1, y1), (x2 + w, y2)], fill="yellow", width=1)
    concat_img.save("point_matching.png")


def sift_match_and_vis(img_path1, img_path2, out_path="sift_matching.png", max_matches=50):
    """
    Detect and match SIFT keypoints between two images and visualize the matches.
    """
    # Read images in grayscale
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print(f"Error loading images: {img_path1}, {img_path2}")
        return

    # SIFT detector
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good = sorted(good, key=lambda x: x.distance)[:max_matches]

    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(out_path, img_matches)
    print(f"SIFT matches visualization saved to {out_path}")


if __name__ == "__main__":
    # image_names = ["/data/fanry/Desktop/fmft/locft/data/scannet_test_1500/scene0707_00/color/15.jpg",
    #                "/data/fanry/Desktop/fmft/locft/data/scannet_test_1500/scene0707_00/color/585.jpg"]
    image_names = ["/data/fanry/Desktop/fmft/naviPair/3d_dollhouse_sink/45/pair2/004.jpg",
                   "/data/fanry/Desktop/fmft/naviPair/3d_dollhouse_sink/45/pair2/018.jpg"]
    images, track_list, vis_score, conf_score = patch_track(image_names, 14)
    source_points, target_points = track_point_filter(track_list, vis_score, conf_score)
    
    # sample_mask = np.random.binomial(n=1, p=0.02, size=source_points.shape[0])
    # sample_mask = np.random.binomial(n=1, p=0.02, size=source_points.shape[0])
    # source_points = source_points[sample_mask > 0.01, :]
    # target_points = target_points[sample_mask > 0.01, :]
    source_points_masked, target_points_masked = track_point_filter(track_list, vis_score, conf_score, 0.2)

    image_names = ["/data/fanry/Desktop/fmft/naviPair/3d_dollhouse_sink/45/pair2/018.jpg",
                   "/data/fanry/Desktop/fmft/naviPair/3d_dollhouse_sink/45/pair2/004.jpg"]
    images2, track_list, vis_score, conf_score = patch_track(image_names, 14)
    source_points, target_points = track_point_filter(track_list, vis_score, conf_score)
    source_points_masked2, target_points_masked2 = track_point_filter(track_list, vis_score, conf_score, 0.2)
    source_points_masked = torch.cat([source_points_masked, target_points_masked2], dim=0)
    target_points_masked = torch.cat([target_points_masked, source_points_masked2], dim=0)
    vis_track_points(images, source_points_masked, target_points_masked)

    # SIFT keypoint matching visualization
    # sift_match_and_vis(image_names[0], image_names[1], out_path="sift_matching.png", max_matches=50)
