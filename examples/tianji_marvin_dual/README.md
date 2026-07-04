# Dual Tianji Marvin + Zhixing Flow

This adapter keeps the same 16D policy interface as `examples/rokae_zhixing_dual`:

```text
[left_j1..left_j7 rad, left_gripper_m, right_j1..right_j7 rad, right_gripper_m]
```

At the SDK boundary, Tianji Marvin joint commands are converted to degrees and TCP reset poses are converted from repository `m/rad` to SDK `mm/deg`.

## Hardware Check

```bash
python examples/tianji_marvin_dual/check_hardware.py \
  --robot_ip 192.168.1.190
```

The default check connects through `MarvinSDK`, reads A/B joints and TCP poses, and does not send motion commands. Add `--enable_position_state` only after confirming the robot is ready for a mode switch.

## OpenPI Inference

```bash
python examples/tianji_marvin_dual/main.py \
  --host POLICY_SERVER_IP \
  --port 8000 \
  --robot_ip 192.168.1.190
```

## Online RL / DAgger

Machine B services:

```bash
python rlt_online_rl/launch/launch_machine_b.py \
  --config rlt_online_rl/configs/tasks/tianji_marvin_dual/online_rl.yaml
```

Robot rollout machine:

```bash
python rlt_online_rl/launch/launch_robot_rollout.py \
  --config rlt_online_rl/configs/tasks/tianji_marvin_dual/online_rl.yaml \
  --ros_script tianji_marvin_dual \
  --robot_ip 192.168.1.190
```

Actor-only evaluation:

```bash
python rlt_online_rl/launch/launch_actor_eval.py \
  --config rlt_online_rl/configs/tasks/tianji_marvin_dual/online_rl.yaml \
  --ros_script tianji_marvin_dual \
  --robot_ip 192.168.1.190
```

The SDK is vendored at `third_party/tianji_marvin_sdk`; override `--tianji_sdk_python_dir` only when testing a different local SDK build.

Before hardware rollout, calibrate `LEFT_RESET_END_POSE` / `RIGHT_RESET_END_POSE` or provide 16D reset actions in the task config. All-zero dual-arm joint reset poses are rejected by default.
