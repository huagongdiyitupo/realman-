import gymnasium as gym
import time
import argparse
import mani_skill  # 确保 mani_skill 被 import 以注册环境
import lerobot_sim2real.my_env.RealmanGraspCubeEnv  # 确保你的自定义环境被 import
from mani_skill.utils import gym_utils
def main(env_id, env_kwargs_json_path=None):
    import json

    # 读取 env kwargs
    env_kwargs = {}
    if env_kwargs_json_path:
        with open(env_kwargs_json_path, "r") as f:
            env_kwargs = json.load(f)

    # 创建环境
    env = gym.make(env_id, render_mode="human", obs_mode="state", **env_kwargs)
    obs = env.reset()
    env.render()
    while True:  # 渲染 100 步
        obs = env.reset()
        env.render()
        time.sleep(0.2)  # 控制刷新速度
    #     action = None
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     # if terminated or truncated:
    #     #     obs = env.reset()
    #     time.sleep(0.05)  # 控制刷新速度

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="RealmanGraspCube-v1")
    parser.add_argument("--env-kwargs-json-path", type=str, default=None)
    args = parser.parse_args()

    main(args.env_id, args.env_kwargs_json_path)
