CUDA_VISIBLE_DEVICES=7 python evaluate_nyu_depth.py backbone=dinov2_b14_ft_conv \
        backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/nyu_finetune/conv_layer_0.pth \
        backbone.mode=finetune