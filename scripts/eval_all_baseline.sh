backbone=$1
log_file=./logs/${backbone}_baseline.log

python evaluate_navi_correspondence.py backbone=$backbone log_file=$log_file
python evaluate_scannet_correspondence.py backbone=$backbone log_file=$log_file
python evaluate_scannet_pose.py backbone=$backbone log_file=$log_file
python evaluate_onepose_pose.py backbone=$backbone log_file=$log_file
python evaluate_pascal_pf.py backbone=$backbone log_file=$log_file
python evaluate_tapvid_video.py backbone=$backbone log_file=$log_file
