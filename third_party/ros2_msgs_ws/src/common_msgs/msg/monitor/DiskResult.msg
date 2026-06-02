# DiskResult.msg
string level
float64 usage               # 硬盘占用率（%）
string disk_path            # 监控路径（如/）
float64 total_disk          # 总硬盘（GB）
float64 used_disk           # 已用硬盘（GB）
float64 warning_threshold
float64 critical_threshold
builtin_interfaces/Time timestamp
string description