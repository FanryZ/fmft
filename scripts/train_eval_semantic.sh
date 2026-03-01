backbones=(mae_b16 deit3_b16)
pascal_idx=4
log_file=./logs/semantic.log


for bck in ${backbones[@]}; do
    python finetune/pascal_finetune.py backbone=$bck select_idx=$pascal_idx
    python evaluate_pascal_pf.py backbone=${bck}_ft_conv log_file=$log_file \
        backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/pascal_finetune/conv_layer_select.pth \
        backbone.mode=finetune
done