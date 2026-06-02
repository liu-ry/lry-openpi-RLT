# NodeAliveResult.msg
string level
int32 alive_count           # 存活节点数
int32 total_count           # 总节点数
string[] missing_nodes      # 失活节点名
builtin_interfaces/Time timestamp
string description