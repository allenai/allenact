# python main.py projects/objectnav_baselines/experiments/robothor/objectnav_robothor_rgb_rawgru_ddppo.py \
#     -o experiment_output/objectnav/ --seed 302
# python main.py projects/pointnav_baselines/experiments/robothor/pointnav_robothor_rgb_simpleconvgru_ddppo.py\
#     -o experiment_output/pointnav/ --seed 302
python main.py projects/manipulathor_disturb_free/armpointnav_baselines/experiments/ithor/armpointnav_depth.py \
    -o experiment_output/armpointnav/ --seed 302 --save_dir_fmt NESTED

