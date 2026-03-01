
for num in {2..70}; do
    CUDA_VISIBLE_DEVICES=7 python evaluate_tapvid_video.py backbone=dinov2_b14_ft_conv \
        backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/tapvid_finetune/conv_layer_$num.pth \
        backbone.mode=finetune
done