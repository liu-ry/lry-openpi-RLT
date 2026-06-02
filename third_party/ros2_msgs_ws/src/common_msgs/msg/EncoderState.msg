std_msgs/Header header          # 时间戳和frame_id  

# 编码器数值
float32 gripper_angle           # 夹爪角度 (度) 
uint16 encoder_raw_value        # 编码器原始值 (14bit?)  

# 编码器状态
uint8 encoder_status            # 编码器状态 (正常/异常) 

# 编码器状态常量
uint8 ENCODER_OK=0 
uint8 ENCODER_ERROR=1 
uint8 ENCODER_CALIBRATING=2    #编码器标定/归零
