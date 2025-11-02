# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import torch

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for environment and policy.")
parser.add_argument("--headless", action="store_true", default=False, help="Run the simulation in headless mode.")
parser.add_argument("--device", type=str, default="cpu")

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# # always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# # device selection
# if torch.cuda.is_available():
#     args_cli.device = "cuda:0"

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

################ User Define Dependencies ######################
from fast_td3.fast_td3_deploy import load_policy
from fast_td3.environments.isaaclab_env import IsaacLabEnv


def main():

    # create isaac environment
    env = IsaacLabEnv(
        task_name = args_cli.task,
        device    = args_cli.device,
        num_envs  = args_cli.num_envs,
        seed      = args_cli.seed,
        isheadless = args_cli.headless,
    )

    # 录像 + 各种信息 存放路径
    log_root_path = os.path.join("models", "test")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")


    # 模型路径
    # TODO: 修改为你自己的模型路径
    # resume_path = "/home/ece-486/Documents/Robotics/FastTD3/models/Isaac-Velocity-Flat-G1-v0_FastTD3/2025-11-02_15-13/Isaac-Velocity-Flat-G1-v0__FastTD3__1_45000.pt"
    resume_path = "/home/ece-486/Documents/Robotics/FastTD3/models/Isaac-Velocity-Flat-Unitree-Go2-v0_FastTD3/2025-11-02_15-42/Isaac-Velocity-Flat-Unitree-Go2-v0__FastTD3__1_45000.pt"  
    print(f"****\n\n[INFO]: Loading model checkpoint from: {resume_path}\n\n****")

    log_dir = os.path.dirname(resume_path)


    # 加载策略
    policy = load_policy(resume_path)

    dt = 0.02

    # reset environment
    obs = env.reset()
    timestep = 0
    # simulate environment
    for _ in range(args_cli.video_length):
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    # simulation_app.close()