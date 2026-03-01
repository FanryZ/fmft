
for num in {0..63}; do
    CUDA_VISIBLE_DEVICES=7 python evaluate_scannet_correspondence.py backbone=dinov2_b14_ft_conv \
        backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/scannet_finetune/conv_layer_$num.pth \
        backbone.mode=finetune
done