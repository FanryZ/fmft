export CUDA_VISIBLE_DEVICES=2

python finetune/scannet_finetune_conv.py \
    pair_num=1 iter_num=100 neg_pair_num=5 temperature=0.1 weight_decay=0.0001

python evaluate_scannet_correspondence.py \
    backbone=dinov2_b14_ft_conv \
    backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/scannet_finetune/conv_layer_0.pth
    