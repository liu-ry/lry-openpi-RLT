#!/usr/bin/env bash
# 一键编译仓库内的 ROS2 消息定义（common_msgs 等）
# 用法：bash third_party/ros2_msgs_ws/build_msgs.sh
# 只需在第一次克隆或消息定义变更后运行一次，后续无需任何 source 操作。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检测 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    # 按常见路径自动 source ROS2 base
    for _ros_setup in /opt/ros/humble/setup.bash /opt/ros/iron/setup.bash /opt/ros/foxy/setup.bash; do
        if [ -f "$_ros_setup" ]; then
            # shellcheck source=/dev/null
            source "$_ros_setup"
            echo "[build_msgs] sourced $_ros_setup"
            break
        fi
    done
fi

if [ -z "$ROS_DISTRO" ]; then
    echo "[build_msgs] ERROR: ROS2 环境未找到，请先 source /opt/ros/<distro>/setup.bash"
    exit 1
fi

echo "[build_msgs] 使用 ROS_DISTRO=$ROS_DISTRO"
echo "[build_msgs] 工作空间: $SCRIPT_DIR"

cd "$SCRIPT_DIR"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

echo ""
echo "[build_msgs] ✅ 编译完成！"
echo "  install 目录: $SCRIPT_DIR/install"
echo "  无需手动 source，dobot_umi_ros.py 会自动加载。"
