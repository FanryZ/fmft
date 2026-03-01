export CUDA_VISIBLE_DEVICES=0

weight_decay=(0.0001 0.001 1.0)
temp=(0.1 0.5 1.0 10.0)
neg_pair_num=(1 2 5 10 20)

# for weight_decay in ${weight_decay[@]}; do
#     python -m finetune.tapvid_finetune weight_decay=${weight_decay} pair_num=1
#     python evaluate_tapvid_video.py video_num=4 \
#         backbone=dinov2_b14_ft_conv \
#         backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/tapvid_search/conv_layer_0.pth \
#         backbone.mode=finetune
# done

# for temp in ${temp[@]}; do
#     python -m finetune.tapvid_finetune temperature=${temp} pair_num=1
#     python evaluate_tapvid_video.py video_num=4 \
#         backbone=dinov2_b14_ft_conv \
#         backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/tapvid_search/conv_layer_0.pth \
#         backbone.mode=finetune
# done

for neg_pair_num in ${neg_pair_num[@]}; do
    python -m finetune.tapvid_finetune neg_pair_num=${neg_pair_num} pair_num=1 temperature=0.1
    python evaluate_tapvid_video.py video_num=4 \
        backbone=dinov2_b14_ft_conv \
        backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/tapvid_search/conv_layer_0.pth \
        backbone.mode=finetune
done
