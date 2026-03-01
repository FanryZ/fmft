python vis_sequence.py backbone=dinov2_b14_ft_conv backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/scannet_finetune/conv_layer_1.pth \
    backbone.mode=finetune vis_dir=ours
# python vis_sequence.py backbone=dinov2_b14 vis_dir=dinov2
# python vis_sequence.py backbone=dvt_dino vis_dir=dvt
# python vis_sequence.py backbone=corr_b14 vis_dir=corr
# python vis_sequence.py backbone=fit3d vis_dir=fit3d

python scannet_feat_vis.py vis_dir=ours
# python scannet_feat_vis.py vis_dir=dinov2
# python scannet_feat_vis.py vis_dir=dvt
# python scannet_feat_vis.py vis_dir=corr
# python scannet_feat_vis.py vis_dir=fit3d

