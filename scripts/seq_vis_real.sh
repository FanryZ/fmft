python evaluate_realworld.py backbone=dinov2_b14_ft_conv backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/realworld_finetune/conv_layer_0.pth \
    backbone.mode=finetune vis_dir=ours
python evaluate_realworld.py backbone=dinov2_b14 vis_dir=dinov2
    