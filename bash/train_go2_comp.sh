# FastTD3
# Note 2025.11.1: 
#   Default num_envs = 4096
#   and will exceed 4090 GRAM limit

python fast_td3/train.py \
    --env_name Isaac-Velocity-Flat-Unitree-Go2-Comp-Norm-v0 \
    --exp_name FastTD3 \
    --render_interval 0 \
    --seed 1 \
    --num_envs 2048