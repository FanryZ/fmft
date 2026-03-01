backbones=(dinov2_b14 clip_b16_laion mae_b16 deit3_b16 dinov2_b14_reg)

for bck in ${backbones[@]}; do
    python evaluate_scannet_pose.py backbone=$bck
done