std_msgs/Header header  

# LED模式 
uint8 led_mode

# LED命令常量
uint8 MODE_OFF = 0 
uint8 MODE_GREEN_SOLID = 1         # 绿色常亮：自检通过进入待机
uint8 MODE_GREEN_BLIK =2           # 绿色闪烁：上电自检中
uint8 MODE_RED_SOLID = 3           # 红色常亮：故障告警
uint8 MODE_BLUE_BLINK = 4          # 蓝色闪烁：数采准备/收尾状态
uint8 MODE_BLUE_SOLID = 5          # 蓝色常亮：数采工作状态
uint8 MODE_YELLOW_SOLID = 6        # 黄色常亮：已上电/待机

# 蜂鸣器模式
uint8 beep_mode

# 蜂鸣器命令常量
uint8 MODE_BEEP_OFF   = 0
uint8 MODE_SHORT_BEEP = 1
uint8 MODE_LONG_BEEP  = 2
uint8 MODE_BEEPING    = 3

# 重置encoder
bool reset_zero_point          # 是否重系统状态常量置