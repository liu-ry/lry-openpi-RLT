# CpuResult.msg
string level                # normal/warning/critical
float64 usage               # CPU占用率（%）
float64 warning_threshold   # 警告阈值
float64 critical_threshold  # 紧急阈值
builtin_interfaces/Time timestamp  # 时间戳
string description          # 易读描述