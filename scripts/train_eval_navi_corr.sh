python finetune/navi_finetune.py

for num in {0..119}; do
    python evaluate_navi_correspondence.py backbone=dinov2_b14_ft_conv \
        backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/navi_finetune/conv_layer_$num.pth \
        backbone.mode=finetune
done
