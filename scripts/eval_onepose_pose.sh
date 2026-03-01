# lambda_ft=(0.1 0.2 0.5 0.8 1.5 5.0)
# temp_ft=(100.0 1.0)
# cache_feat=(
    # "/data/fanry/Desktop/fmft/locft/cache_feats/dino_base_scannet_2shot.pth"
    # "/data/fanry/Desktop/fmft/locft/cache_feats/dino_base_scannet.pth"
# )
export CUDA_VISIBLE_DEVICES=6
# mat_idx={0..29}

for mat_idx in {0..29}; do
    python evaluate_onepose_pose.py \
        backbone=dinov2_b14_ft_conv \
        backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/onepose_finetune/conv_layer_$mat_idx.pth
done
