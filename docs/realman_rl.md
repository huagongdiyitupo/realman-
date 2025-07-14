

## 1: 环境初始化

启动conda环境
```bash
conda activate ManiSkill_RL
```

环境和机器人都放在lerobot_sim2real->my_env->RealmanGraspCubeEnv.py中

运行以下指令查看环境随机初始化

```bash
python lerobot_sim2real/scripts/record_reset_distribution.py --env-id="RealmanGraspCube-v1" --env-kwargs-json-path=env_config.json
```

## 2: 训练

通过以下命令训练

```bash
seed=3
python lerobot_sim2real/scripts/train_ppo_rgb.py --env-id="RealmanGraspCube-v1" --env-kwargs-json-path=env_config.json \
  --ppo.seed=${seed} \
  --ppo.num_envs=1024 --ppo.num-steps=16 --ppo.update_epochs=8 --ppo.num_minibatches=32 \
  --ppo.total_timesteps=100_000_000 --ppo.gamma=0.9 \
  --ppo.num_eval_envs=16 --ppo.num-eval-steps=64 --ppo.no-partial-reset \
  --ppo.exp-name="ppo-SO100GraspCube-v1-rgb-${seed}" \
  --ppo.track --ppo.wandb_project_name "RealmanGraspCube-v1"
```

实际训练过程中发现会超显存，可能是由于SAPIEN 仍然完整加载所有 16 个自由度（机械臂14个自由度+夹爪2个自由度），但实际只需要控制左臂7个自由度加夹爪1个自由度（夹爪在环境中是主从控制）。后续优化可以改 URDF 只保留左臂尝试。

