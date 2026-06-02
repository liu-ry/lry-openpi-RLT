std_msgs/Header header

# 按键状态
uint8 event_type

# 按钮状态常量
uint8 EVENT_LONG_PRESS=1        #长按(>2秒)开机/关机
uint8 EVENT_REC_TOGGLE=2        #单击开始/结束录制
uint8 EVENT_DOUBLE_CLICK=3      #双击进行标定/归零
uint8 EVENT_ERROR_ACK=4         #故障
