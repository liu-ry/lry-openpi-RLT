import os
import shutil
import json
import numpy as np
from glob import glob
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import cv2
from pathlib import Path

def extract_video_to_frames(video_path, target_height=480, target_width=640):
    """将视频文件转换为帧列表（使用 opencv）"""
    frames = []
    # 目标分辨率 (高度, 宽度)，注意 cv2.resize 的尺寸参数是 (width, height)
    # target_height, target_width = 480, 640
    target_channels = 3 
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")
    
    # 逐帧读取
    while True:
        ret, frame = cap.read()  # ret 表示是否读取成功，frame 是帧数据（BGR 格式）
        if not ret:
            break  # 读取完毕
        
        # 转换为 RGB 格式（可选，根据需求是否需要）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测分辨率和通道数
        h, w, c = frame_rgb.shape
        if (h, w, c) != (target_height, target_width, target_channels):
            # print(f"{video_path}: 当前尺寸与预期的图像尺寸不匹配，即将进行分辨率的resize")
            # 调整尺寸为 (target_width, target_height)，注意 cv2.resize 的参数顺序是 (宽, 高)
            frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
            # 确保通道数为 3（若原始图像为灰度图，resize 后仍可能为单通道，这里强制转换）
            if frame_rgb.ndim == 2:
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)       
        
        frames.append(frame_rgb)  # 存储 RGB 格式的帧
    
    # 释放资源
    cap.release()
    return frames


def get_action_from_state_pos(state_pos):
    """
    从关节位置序列生成动作序列。
    
    每一帧的动作是下一帧的关节位置。
    最后一帧的动作用前一帧的动作拷贝补齐。
    
    Args:
        state_pos: numpy array, shape (num_steps, joint_dim)
        
    Returns:
        actions: numpy array, shape (num_steps, joint_dim)
                其中 actions[i] = state_pos[i+1]，
                而 actions[-1] = actions[-2]
    """
    num_steps = state_pos.shape[0]
    
    # 创建action数组，初始为state_pos向前偏移一帧
    # actions[i] = state_pos[i+1]，所以 actions[:-1] = state_pos[1:]
    actions = np.zeros_like(state_pos)
    actions[:-1] = state_pos[1:]
    
    # 最后一帧用前一帧拷贝补齐
    actions[-1] = actions[-2]
    
    return actions

def main(root_dir: str = "/home/lry/temp/sync", 
        output_dir: str = "/home/lry/temp/lerobot_output",
        single_dir: bool = False):
    error_num = 0
    succ_num = 0
    output_path = Path(output_dir)
    # 清理现有输出目录
    if output_path.exists():
        # 提示用户确认，说明即将删除的目录
        user_input = input(
            f"目录 '{output_path}' 已存在，是否删除该目录及其所有内容？[y/N] "
        ).strip().lower()  # 转为小写，方便判断
        
        # 仅当用户输入 'y' 或 'yes' 时执行删除
        if user_input in ("y", "yes"):
            print(f"正在删除目录: {output_path}")
            shutil.rmtree(output_path)
            print("删除完成")
        else:
            print("已取消删除以及数据转换操作")   
            return     
    # 创建LeRobot数据集，定义要存储的特征
    # 新数据集格式：
    #   robot_joint_velocity_force.npy -> (N, 18): 只取前6维（6自由度关节位置）
    #   gripper.npy                    -> (N,)   : 夹爪开合值
    #   robot_tcp_pose.npy             -> (N, 7) : TCP末端位姿 (xyz + quaternion)
    #   realsense_top_rgb.mp4          -> 640x480 俯视相机
    #   realsense_wrist_rgb.mp4        -> 640x480 腕部相机
    #   tactile_left_warped_image.mp4  -> 240x240 左触觉图像
    #   tactile_right_warped_image.mp4 -> 240x240 右触觉图像
    #   timestamps.npy                 -> (N,)   : 时间戳
    JOINT_DIM = 6    # 6自由度机械臂，只取 robot_joint_velocity_force 前6维
    STATE_DIM = JOINT_DIM + 1   # 6 (joint) + 1 (gripper) = 7
    state_names = [f"joint_{i}" for i in range(JOINT_DIM)] + ["gripper"]

    dataset = LeRobotDataset.create(
        repo_id=output_dir,
        robot_type="vitai",
        fps=30,
        features={
            # 俯视 RGB 相机
            "observation.images.cam_top": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            # 腕部 RGB 相机
            "observation.images.cam_wrist": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            # 左触觉图像
            "observation.images.tactile_left": {
                "dtype": "image",
                "shape": (240, 240, 3),
                "names": ["height", "width", "channel"],
            },
            # 右触觉图像
            "observation.images.tactile_right": {
                "dtype": "image",
                "shape": (240, 240, 3),
                "names": ["height", "width", "channel"],
            },
            # 机器人状态：关节位置前6维 + 夹爪 (1) = 7 维
            "observation.state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": state_names,
            },
            # 动作：下一帧状态，与 observation.state 同维度
            "actions": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": state_names,
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # 定义数据目录路径
    count = 0
    for subdir_name in os.listdir(root_dir):
        if single_dir:
            if count > 0:
                break
            subdir_path = root_dir
            count = count + 1
        else:
            subdir_path = os.path.join(root_dir, subdir_name)
    
        if not os.path.exists(subdir_path):
            raise ValueError(f"Episodes目录不存在: {subdir_path}")
        # 确保是目录（排除文件）
        if not os.path.isdir(subdir_path):
            continue
        
        print(f"进入文件夹：{subdir_path}")

        # 获取所有episode文件夹（只保留目录，排除文件）
        episode_folders = sorted([
            f for f in glob(os.path.join(subdir_path, "episode_*"))
            if os.path.isdir(f)
        ])
        print(f"找到 {len(episode_folders)} 个episode")

        # 处理每个episode
        for episode_idx, episode_folder in enumerate(episode_folders):
            print(f"处理 episode {episode_idx + 1}/{len(episode_folders)}: {os.path.basename(episode_folder)}")

            try:
                # 加载元数据（如果不存在则使用默认值）
                metadata_path = os.path.join(episode_folder, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {"prompt": "robot manipulation task"}

                default_task = metadata.get("prompt", "robot manipulation task")

                # ── 加载 numpy 数据 ──────────────────────────────────────────
                # robot_joint_velocity_force: (N, 18)，只取前 JOINT_DIM=6 维
                joint_vel_force = np.load(
                    os.path.join(episode_folder, "robot_joint_velocity_force.npy")
                ).astype(np.float32)[:, :JOINT_DIM]
                # gripper: (N,) → reshape 为 (N, 1) 方便拼接
                gripper = np.load(
                    os.path.join(episode_folder, "gripper.npy")
                ).astype(np.float32).reshape(-1, 1)
                # timestamps
                timestamps = np.load(os.path.join(episode_folder, "timestamps.npy"))

                # 拼接状态向量：(N, 6+1) = (N, 7)
                state_pos = np.concatenate([joint_vel_force, gripper], axis=1)

                # 动作 = 下一帧状态（最后一帧复制前一帧）
                actions = get_action_from_state_pos(state_pos)

                # ── 加载视频帧 ───────────────────────────────────────────────
                video_files = {
                    "observation.images.cam_top":   "realsense_top_rgb.mp4",
                    "observation.images.cam_wrist":  "realsense_wrist_rgb.mp4",
                }
                tactile_video_files = {
                    "observation.images.tactile_left":  "tactile_left_warped_image.mp4",
                    "observation.images.tactile_right": "tactile_right_warped_image.mp4",
                }

                video_frames = {}
                for key, filename in video_files.items():
                    video_path = os.path.join(episode_folder, filename)
                    video_frames[key] = extract_video_to_frames(video_path)
                for key, filename in tactile_video_files.items():
                    video_path = os.path.join(episode_folder, filename)
                    video_frames[key] = extract_video_to_frames(
                        video_path, target_height=240, target_width=240
                    )

                # ── 检查所有数据长度是否一致 ─────────────────────────────────
                lengths = [
                    len(state_pos),
                    len(actions),
                    len(timestamps),
                    len(video_frames["observation.images.cam_top"]),
                    len(video_frames["observation.images.cam_wrist"]),
                    len(video_frames["observation.images.tactile_left"]),
                    len(video_frames["observation.images.tactile_right"]),
                ]
                if len(set(lengths)) > 1:
                    print(f"警告: 数据长度不一致 {lengths}，跳过此episode")
                    continue

                # ── 逐帧写入数据集 ───────────────────────────────────────────
                num_steps = len(timestamps)
                for step_idx in range(num_steps):
                    frame_data = {
                        "observation.images.cam_top":    video_frames["observation.images.cam_top"][step_idx],
                        "observation.images.cam_wrist":  video_frames["observation.images.cam_wrist"][step_idx],
                        "observation.images.tactile_left":  video_frames["observation.images.tactile_left"][step_idx],
                        "observation.images.tactile_right": video_frames["observation.images.tactile_right"][step_idx],
                        "observation.state": state_pos[step_idx],
                        "actions":           actions[step_idx],
                        "task":              default_task,
                    }
                    dataset.add_frame(frame_data)

                # 保存当前episode
                dataset.save_episode()
                succ_num = succ_num + 1
                print(f" {episode_folder} 处理成功， 总计成功处理并保存了 {succ_num} 个")

            except Exception as e:
                error_num = error_num + 1
                print(f"处理episode时出错 {episode_folder}: {str(e)} ,总计错误了 {error_num} 个")
                continue

    print("数据集转换完成！")


if __name__ == "__main__":
    tyro.cli(main)