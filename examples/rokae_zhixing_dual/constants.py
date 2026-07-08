"""Constants for dual Rokae AR arms with Zhixing grippers."""
# ruff: noqa
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Control cycle. Rokae realtime examples use 0.008 s; keep 50 Hz by default to
# match the existing rollout runtime and reduce command pressure.
DT = 0.033

# Vendored Rokae xCoreSDK Python wrapper location.
ROKAE_SDK_PYTHON_DIR = str(REPO_ROOT / "third_party" / "rokae_xcore_sdk" / "pyrokae")

# Dual arm network parameters. Adjust these to your actual left/right arms.
LEFT_ARM_REMOTE_IP = "192.168.9.162"
LEFT_ARM_LOCAL_IP = "192.168.9.160"
RIGHT_ARM_REMOTE_IP = "192.168.10.161"
RIGHT_ARM_LOCAL_IP = "192.168.10.160"

# Rokae AR is 7-DoF. Runtime hardware wrappers use [j1..j7 rad, gripper_m].
# The RLT/policy/replay layer for this task uses converted dataset units:
# [j1..j7 deg, gripper SDK raw position / ROKAE_POLICY_GRIPPER_RAW_SCALE] per arm.
ARM_DOF = 7
SINGLE_ARM_ACTION_DIM = ARM_DOF + 1
DUAL_ACTION_DIM = SINGLE_ARM_ACTION_DIM * 2

# Realtime servo parameters.
ROKAE_SERVO_DT = DT
ROKAE_SERVO_LOOKAHEAD = DT * 5
ROKAE_SERVO_KP = 1.0
ROKAE_MOVEJ_SPEED = 20.0
ROKAE_RESET_MOVEJ_SPEED = 20.0  # reset的默认速度，保持最慢恢复
ROKAE_MOVEJ_ZONE = -1.0

# Safety clamps for policy/teleop absolute joint targets.
MAX_JOINT_DELTA_RAD = 0.03
MAX_GRIPPER_DELTA_M = 10.0
JOINT_LIMITS_DEG = (
    (-178.0, 178.0),
    (-120.0, 120.0),
    (-178.0, 178.0),
    (-60.0, 145.0),
    (-178.0, 178.0),
    (-50.0, 50.0),
    (-50.0, 50.0),
)
JOINT_LIMIT_MARGIN_DEG = 5.0

# Default reset end poses in Rokae getEndPose() format:
# [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad].
LEFT_RESET_END_POSE = [0.490056, 0.290866, -0.307963, -2.299171, 0.072474, -2.550856]
RIGHT_RESET_END_POSE = [0.537451, -0.306208, -0.286341, 2.169089, 0.236971, 2.488939]

# Optional joint reset poses in radians. If provided by the policy server or
# by code, these take precedence over the 6D end-pose IK reset above.
# The runtime refuses to move to all-zero joint reset poses unless explicitly allowed.
LEFT_RESET_JOINT_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
RIGHT_RESET_JOINT_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ALLOW_ZERO_RESET_POSE = False

# Zhixing grippers. Use separate serial adapters if both grippers share slave id.
LEFT_GRIPPER_SERIAL_PORT = "/dev/ttyUSB2"
LEFT_GRIPPER_SLAVE_ID = 1
RIGHT_GRIPPER_SERIAL_PORT = "/dev/ttyUSB0"
RIGHT_GRIPPER_SLAVE_ID = 1
GRIPPER_BAUDRATE = 115200
GRIPPER_SPEED_PCT = 5
GRIPPER_FORCE_PCT = 50
GRIPPER_POS_OPEN = 0
GRIPPER_POS_CLOSE = 12000
# Converted Rokae datasets store Zhixing SDK raw position divided by this scale.
ROKAE_POLICY_GRIPPER_RAW_SCALE = 100.0
GRIPPER_OPEN_M = 0.085
GRIPPER_CLOSE_M = 0.0

# RealSense cameras. Names are kept compatible with common OpenPI configs.
REALSENSE_FRONT_SERIAL = "242322073782"
REALSENSE_LEFT_WRIST_SERIAL = "260322276574"
REALSENSE_RIGHT_WRIST_SERIAL = "260322271611"
REALSENSE_WIDTH = 640
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 30
REALSENSE_FRONT_EXPOSURE = 100
REALSENSE_FRONT_GAIN = 64
REALSENSE_WRIST_EXPOSURE = 20000
REALSENSE_WRIST_GAIN = 16

IMAGE_KEY_FRONT = "cam_high"
IMAGE_KEY_LEFT_WRIST = "cam_left_wrist"
IMAGE_KEY_RIGHT_WRIST = "cam_right_wrist"

# Optional tactile cameras, reused from the Dobot+UMI example if available.
TACTILE_LEFT_SERIAL = "GF22511615AAF"
TACTILE_RIGHT_SERIAL = "GF22513812AF6"
IMAGE_KEY_TACTILE_LEFT = "tactile_left"
IMAGE_KEY_TACTILE_RIGHT = "tactile_right"
