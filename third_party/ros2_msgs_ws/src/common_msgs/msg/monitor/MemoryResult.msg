# MemoryResult.msg
string level
float64 usage               # 内存占用率（%）
float64 total_memory        # 总内存（GB）
float64 used_memory         # 已用内存（GB）
float64 warning_threshold
float64 critical_threshold
builtin_interfaces/Time timestamp
string description