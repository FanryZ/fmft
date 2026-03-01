python evaluate_scannet_correspondence.py visualize=True backbone=dinov2_b14 \
    vis_dir=/data/fanry/Desktop/fmft/probe3d/vis2/scannet_correspondence/dinov2_b14

python evaluate_scannet_correspondence.py visualize=True backbone=dinov2_b14_ft_conv \
    backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/scannet_finetune/conv_layer_0.pth \
    backbone.mode=finetune \
    vis_dir=/data/fanry/Desktop/fmft/probe3d/vis2/scannet_correspondence/dinov2_b14_ft_conv

python evaluate_navi_correspondence.py visualize=True backbone=dinov2_b14 \
    vis_dir=/data/fanry/Desktop/fmft/probe3d/vis2/navi_correspondence/dinov2_b14

python evaluate_navi_correspondence.py visualize=True backbone=dinov2_b14_ft_conv \
    backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/navi_finetune/conv_layer_41.pth \
    backbone.mode=finetune \
    vis_dir=/data/fanry/Desktop/fmft/probe3d/vis2/navi_correspondence/dinov2_b14_ft_conv

    
