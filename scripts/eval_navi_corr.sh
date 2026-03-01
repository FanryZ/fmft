# lambda_ft=(0.1 0.2 0.5 0.8 1.5 5.0)
# temp_ft=(100.0 1.0)
# cache_feat=(
    # "/data/fanry/Desktop/fmft/locft/cache_feats/dino_base_scannet_2shot.pth"
    # "/data/fanry/Desktop/fmft/locft/cache_feats/dino_base_scannet.pth"
# )
export CUDA_VISIBLE_DEVICES=6

for num in {0..119}; do
    python evaluate_navi_correspondence.py backbone=dinov2_b14_ft_conv \
        backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/navi_finetune/conv_layer_$num.pth \
        backbone.mode=finetune
done
