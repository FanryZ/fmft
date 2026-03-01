
for num in {0..19}; do
    CUDA_VISIBLE_DEVICES=6 python evaluate_pascal_pf.py backbone=dinov2_b14_ft_conv \
        backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/pascal_finetune/conv_layer_$num.pth \
        backbone.mode=finetune
done