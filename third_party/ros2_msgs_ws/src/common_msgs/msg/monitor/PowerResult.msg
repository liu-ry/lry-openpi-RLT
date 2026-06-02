# PowerResult.msg
string level                # normal/warning/critical（告警级别）
float64 percentage          # 电池电量百分比（0-100）
bool is_charging            # 是否正在充电
float64 voltage             # 电池电压（V，部分系统无数据则为0）
float64 remaining_time      # 剩余使用时间（分钟，-1表示未知）
string power_status         # 电源状态（Charging/Discharging/Full/Unknown）
float64 low_threshold       # 低电量警告阈值（%）
float64 critical_threshold  # 紧急电量阈值（%）
builtin_interfaces/Time timestamp  # 时间戳
string description          # 易读描述（人工查看）