# 设备状态消息
string device_name          # 设备名称，如 "vitai_left1", "ds80"
uint8 level                 # 状态级别：0=OK, 1=ERROR
string message              # 状态描述或错误详情
builtin_interfaces/Time stamp  # 时间戳

# 状态级别常量
uint8 OK = 0
uint8 ERROR = 1
