# Dual Rokae + Zhixing Inference

This example mirrors `examples/dobot_umi`, but the robot backend is two Rokae AR arms driven by the vendored `pyrokae` SDK and two Zhixing grippers.

Action/state layout:

```text
[left_j1..left_j7, left_gripper_m, right_j1..right_j7, right_gripper_m]
```

Run:

```bash
python examples/rokae_zhixing_dual/main.py \
  --host 192.168.3.5 \
  --port 8000 \
  --left-arm-remote-ip 192.168.10.162 \
  --left-arm-local-ip 192.168.10.160 \
  --right-arm-remote-ip 192.168.10.163 \
  --right-arm-local-ip 192.168.10.160 \
  --left-gripper-port /dev/ttyUSB0 \
  --right-gripper-port /dev/ttyUSB1
```

Before hardware rollout, verify the default reset end poses in `constants.py` or provide a policy server `reset_pose`. The Rokae SDK path defaults to `third_party/rokae_xcore_sdk/pyrokae`.

Hardware check:

```bash
python examples/rokae_zhixing_dual/check_hardware.py
python examples/rokae_zhixing_dual/check_hardware.py --gripper_test
python examples/rokae_zhixing_dual/check_hardware.py --skip_tactile
```

The arm check is read-only: it creates `pyrokae.RokaeAR` and reads state, joint position, velocity, and end pose. It does not power on, change mode, or send motion commands.

The vendored Rokae SDK binary is built for Linux x86_64 and CPython 3.10. Run the hardware scripts with a matching Python interpreter unless you rebuild the SDK.

Keyboard Cartesian jog:

```bash
python examples/rokae_zhixing_dual/keyboard_control.py --step-mm 0.5 --movej-speed 5
```

Use `l` / `r` to switch the selected arm. Arrow keys move X/Y, PageUp/PageDown move Z. Each key press sends one small IK + `moveJ_joint` step; default step is 0.5 mm.

RLT online RL rollout:

```bash
# Machine B: replay / learner / actor services
python rlt_online_rl/launch/launch_machine_b.py \
  --config rlt_online_rl/configs/tasks/rokae_zhixing_dual/online_rl.yaml

# Robot machine: dual Rokae rollout
python rlt_online_rl/launch/launch_robot_rollout.py \
  --config rlt_online_rl/configs/tasks/rokae_zhixing_dual/online_rl.yaml \
  --ros_script rokae_zhixing_dual
```

The dual Rokae RLT entrypoint is `rlt_online_rl/train_deploy_alignment/rokae_zhixing_dual_ros.py`. It supports policy rollout and manual scoring/episode services, but does not implement UMI human override.

Safety defaults:

- Policy actions are clamped to `0.03 rad` per joint per step and `0.01 m` per gripper per step.
- Default reset uses 6D Rokae end poses and `inverseKinematics()`. All-zero dual-arm joint reset poses are rejected by default.
- Left/right arm `remote_ip` values must be distinct.
- Two grippers cannot share the same serial port and slave id.
