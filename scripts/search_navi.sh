export CUDA_VISIBLE_DEVICES=0

weight_decay=(0.0001 0.001 0.0)
temp=(0.0 0.1 0.5 1.0 10.0)
neg_pair_num=(1 2 5 10 20)

for neg_pair_num in ${neg_pair_num[@]}; do
    for temp in ${temp[@]}; do
        for weight_decay in ${weight_decay[@]}; do
            python -m finetune.navi_finetune neg_pair_num=${neg_pair_num} pair_num=1 temperature=${temp} weight_decay=${weight_decay}
            python evaluate_navi_correspondence.py \
                backbone=dinov2_b14_ft_conv \
                backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/navi_finetune/conv_layer_0.pth \
                backbone.mode=finetune
done
done
done
