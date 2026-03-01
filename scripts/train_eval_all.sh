backbone=$1
backbone_ft=${backbone}_ft_conv
log_file=./logs/${backbone}_finetune.log
scannet_idx=1
navi_idx=41
pascal_idx=4
onepose_idx=8
tapvid_idx=46

python finetune/scannet_finetune.py backbone=$backbone select_idx=$scannet_idx
python finetune/navi_finetune.py backbone=$backbone select_idx=$navi_idx

python evaluate_navi_correspondence.py backbone=$backbone_ft log_file=$log_file \
    backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/navi_finetune/conv_layer_select.pth \
    backbone.mode=finetune
python evaluate_scannet_correspondence.py backbone=$backbone_ft log_file=$log_file \
    backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/scannet_finetune/conv_layer_select.pth \
    backbone.mode=finetune
python evaluate_scannet_pose.py backbone=$backbone_ft log_file=$log_file \
    backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/scannet_finetune/conv_layer_select.pth \
    backbone.mode=finetune

python finetune/pascal_finetune.py backbone=$backbone select_idx=$pascal_idx
python evaluate_pascal_pf.py backbone=$backbone_ft log_file=$log_file \
    backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/pascal_finetune/conv_layer_select.pth \
    backbone.mode=finetune

python finetune/onepose_finetune.py backbone=$backbone select_idx=$onepose_idx
python finetune/tapvid_finetune.py backbone=$backbone select_idx=$tapvid_idx

python evaluate_onepose_pose.py backbone=$backbone_ft log_file=$log_file \
    backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/onepose_finetune/conv_layer_select.pth \
    backbone.mode=finetune
python evaluate_tapvid_video.py backbone=$backbone_ft log_file=$log_file \
    backbone.conv_layer=/data/fanry/Desktop/fmft/probe3d/exp/tapvid_finetune/conv_layer_select.pth \
    backbone.mode=finetune
