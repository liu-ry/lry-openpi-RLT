from ..tianji_marvin_sdk.fx_robot import Marvin_Robot, DCSS
from ..tianji_marvin_sdk.fx_kine import Marvin_Kine, FX_InvKineSolvePara
import time
import os



class MarvinRobot:
    def __init__(self, ip:str, config_path:str=None):
        """初始化 MarvinSDK
        
        :param ip: 机器人 IP 地址
        :param config_path: 可选的运动学配置文件路径，默认同目录下 ccs_m6_40
        """
        self.robot = Marvin_Robot()
        self.kine_a = Marvin_Kine()
        self.kine_b = Marvin_Kine()
        self.dcss = DCSS()
        self.ip = ip
        self.pln_initialized = False

        default_path = os.path.join(os.path.dirname(__file__), 'ccs_m6_40.MvKDCfg')
        self.config_path = config_path if config_path else default_path

        self._initialize_kinematics()

    def init_kinematics(self, config_path):
        """重新初始化运动学模块
        
        :param config_path: 配置文件路径（*.MvKDCfg）
        """
        self.config_path = config_path
        self._initialize_kinematics()
        self.pln_initialized = False
        self._initialize_planning()

    def _get_kine(self, arm):
        """根据臂选择对应的运动学实例"""
        return self.kine_a if arm == 'A' else self.kine_b
    
    def _initialize_kinematics(self):
        """初始化运动学模块（仅需一次）"""
        if self.config_path is None:
            raise Exception("Kinematics config path is not set")

        for kine in (self.kine_a, self.kine_b):
            kine.log_switch(0)
            config = kine.load_config(0, self.config_path)
            if config is None:
                raise Exception("Failed to load kinematics configuration")
            if not kine.initial_kine(config['TYPE'][0], config['DH'][0], 
                                     config['PNVA'][0], config['BD'][0]):
                raise Exception("Failed to initialize kinematics")
            
    def _initialize_planning(self):
        """初始化规划模块（仅需一次）"""
        if self.pln_initialized:
            return
        if self.config_path is None:
            raise Exception("Kinematics not initialized. Call init_kinematics() first")
        if not self.robot.pln_init(self.config_path):
            raise Exception("Failed to initialize planning")
        self.pln_initialized = True

    
    def connect(self):
        '''连接到机器人并验证连接成功
        
        如果连接失败或数据通道连接失败，抛出异常
        '''
        init = self.robot.connect(self.ip)
        if not init:
            raise Exception('failed to connect to the robot, port is occupied')

        self.robot.clear_set()
        self.robot.check_error_and_clear(self.dcss)
        self.robot.send_cmd()

        '''通过确认 frame 数据的刷新，确认 UDP 数据通道连接成功'''
        motion_tag = 0
        frame_update = None
        for i in range(5):
            sub_data = self.robot.subscribe(self.dcss)
            if sub_data and sub_data['outputs'][0]['frame_serial'] != 0:
                if frame_update != sub_data['outputs'][0]['frame_serial']:
                    motion_tag += 1
                    frame_update = sub_data['outputs'][0]['frame_serial']
            time.sleep(0.01)
        if not motion_tag > 0:
            raise Exception('failed: robot connection failed')
        
        self.robot.log_switch('0')
        self.robot.local_log_switch('0')
        
        self._initialize_planning()
        sdk_version = self.robot.SDK_version()
        ret, version = self.robot.get_param('int', 'VERSION')
        if ret < 0:
            print(f"Failed to get Controller version")
        print(f"Robot connected successfully, SDK version: {sdk_version}, Controller version: {version}")

    def get_info(self):
        """获取机器人当前状态信息"""
        return self.robot.subscribe(self.dcss)
    
    def release(self):
        """释放机器人连接"""
        time.sleep(0.5)
        return self.robot.release_robot()
    
    def _get_current_data(self, arm, name):
        """获取当前数据
        
        :param arm: 'A' 或 'B'
        :return: 数据列表
        """
        sub_data = self.get_info()
        if sub_data is None:
            raise Exception("Failed to get robot state")
        
        arm_idx = 0 if arm == 'A' else 1
        # 从订阅数据中提取当前关节位置
        try:
            data = sub_data['outputs'][arm_idx].get(name, None)
            if data is None:
                raise Exception(f"Failed to extract {name} for arm {arm}")
            return list(data)
        except (KeyError, TypeError, IndexError):
            raise Exception(f"Failed to extract {name} for arm {arm}")
        
    def get_current_vel(self, arm):
        """获取当前关节速度（7 个值，单位：度/秒）

        :param arm: 'A' 或 'B'
        :return: 当前关节速度列表
        :raises Exception: 当读取失败时抛出
        """
        return self._get_current_data(arm, "fb_joint_vel")
    
    def get_current_force(self, arm):
        """获取当前关节力（7 个值，单位：N）

        :param arm: 'A' 或 'B'
        :return: 当前关节力列表
        :raises Exception: 当读取失败时抛出
        """
        return self._get_current_data(arm, "fb_joint_sToq")
    
    def get_current_joints(self, arm):
        """获取当前关节位置（7 个值，单位：度）

        :param arm: 'A' 或 'B'
        :return: 当前关节角度列表
        :raises Exception: 当读取失败时抛出
        """
        return self._get_current_data(arm, "fb_joint_pos")

    def get_current_tcppose(self, arm):
        """获取当前末端 TCP 位姿 [x, y, z, a, b, c]（mm 和度）

        :param arm: 'A' 或 'B'
        :return: 当前 TCP 位姿
        :raises Exception: 当读取失败时抛出
        """
        joints = self.get_current_joints(arm)
        kine = self._get_kine(arm)
        fk_result = kine.fk(joints)
        if fk_result is False or fk_result is None:
            raise Exception("Failed to calculate forward kinematics for current joints")
        pose = kine.mat4x4_to_xyzabc(fk_result)
        if pose is False or pose is None:
            raise Exception("Failed to convert FK result to TCP pose")
        return pose
    
    # ============ 运动学计算 ============
    def fk(self, arm, joints):
        """正向运动学：从关节角度计算末端位姿
        
        :param joints: 关节角度列表，7个值（度）
        :param arm: 'A' 或 'B'，默认 'A'
        :return: 末端位姿列表 [x, y, z, a, b, c] (mm 和度)
        :raises Exception: 如果计算失败
        """
        if arm not in ('A', 'B'):
            raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
        
        if len(joints) != 7:
            raise ValueError(f"joints must have 7 values, got {len(joints)}")
        
        kine = self._get_kine(arm)
        fk_result = kine.fk(joints)
        if fk_result is False or fk_result is None:
            raise Exception(f"FK calculation failed for arm {arm}")
        
        pose = kine.mat4x4_to_xyzabc(fk_result)
        if pose is False or pose is None:
            raise Exception(f"Failed to convert FK result for arm {arm}")
        return pose
    
    def ik(self, arm, pose, ref_joints=None) -> list[float]:
        """逆向运动学：从末端位姿计算关节角度
        
        :param pose: 末端位姿列表 [x, y, z, a, b, c] (mm 和度)
        :param arm: 'A' 或 'B'
        :param ref_joints: 参考关节角度，7个值（度），用于约束解的构型，可选
        :return: 关节角度列表（7个值，度）
        :raises Exception: 如果计算失败（无可达解、奇异等）
        """    
        if arm not in ('A', 'B'):
            raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
        
        if len(pose) != 6:
            raise ValueError(f"pose must have 6 values [x,y,z,a,b,c], got {len(pose)}")
        
        kine = self._get_kine(arm)
        
        # 将 xyzabc 转换为 4x4 矩阵
        mat4x4 = kine.xyzabc_to_mat4x4(pose)
        if mat4x4 is False:
            raise Exception("Failed to convert pose to matrix")
        
        # 创建 IK 参数结构体
        sp = FX_InvKineSolvePara()
        sp.set_input_ik_target_tcp(kine.mat4x4_to_mat1x16(mat4x4))
        
        # 设置 ZSP 类型为 0（最小欧式距离）
        sp.set_input_ik_zsp_type(0)
        
        # 设置参考关节（如果提供则使用，否则使用当前位置）
        if ref_joints is not None:
            if len(ref_joints) != 7:
                raise ValueError(f"ref_joints must have 7 values, got {len(ref_joints)}")
            sp.set_input_ik_ref_joint(ref_joints)
        else:
            # 使用当前关节位置作为参考
            current_joints = self.get_current_joints(arm)
            sp.set_input_ik_ref_joint(current_joints)
        
        # 执行 IK 计算
        result:FX_InvKineSolvePara|bool = kine.ik(sp)
        if result is False:
            # 检查失败原因
            msg = f"IK calculation failed for arm {arm}"
            if sp.m_Output_IsOutRange:
                msg = f"IK failed: pose is out of reach for arm {arm}"
            if sp.m_Output_IsDeg[3]:
                msg = f"IK failed: joint 4 singularity for arm {arm}"
            if sp.m_Output_IsJntExd:
                msg = f"IK failed: over limit for arm {arm}"
            # clean error
            self.robot.clear_set()
            self.robot.check_error_and_clear(self.dcss)
            self.robot.send_cmd()
            raise Exception(msg)
        else:
            return result.m_Output_RetJoint.to_list()
        
    # 急停
    def emergency_stop(self, arm):
        """紧急停止机器人
        
        :param arm: ‘A’, 'B', 'AB', 可以让一条臂软急停，或者两条臂都软急停。
        """
        if arm not in ('A', 'B', 'AB'):
            raise ValueError(f"arm must be 'A', 'B', or 'AB', got '{arm}'")
        self.robot.soft_stop(arm)
    
    def set_state(self,arm:str=None,state:int=1):
        '''设置状态
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param state:
                   ARM_STATE_IDLE = 0,            //////// 下伺服
                   ARM_STATE_POSITION = 1,		//////// 位置跟随
                   ARM_STATE_PVT = 2,			//////// PVT
                   ARM_STATE_TORQ = 3,			//////// 扭矩
                   ARM_STATE_RELEASE = 4,		//////// 协作释放

        :return:
        '''
        if arm is None:
            arms = ['A', 'B']
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
            arms = [arm]

        for arm in arms:
            self.robot.clear_set()
            self.robot.set_state(arm, state)
            self.robot.send_cmd()

    def set_impedance_type(self, arm:str=None,type: int=0):
        '''设置阻抗类型
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param type:
            Type = 1 关节阻抗
            Type = 2 坐标阻抗
            Type = 3 力控
            注：需要在ARM_STATE_TORQ状态: set_state(arm='A',state=3)  才能以阻抗模式控制!!!
        '''
        if arm is None:
            arms = ['A', 'B']
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
            arms = [arm]

        for arm in arms:
            self.robot.clear_set()
            self.robot.set_impedance_type(arm, type)
            self.robot.send_cmd()

    def set_vel_acc(self, arm:str=None, velRatio: int=100, AccRatio: int=100):
        '''设置速度和加速度比例
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param velRatio: 速度比例0-100
        :param AccRatio: 加速度比例0-100
        '''
        if arm is None:
            arms = ['A', 'B']
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
            arms = [arm]

        for arm in arms:
            self.robot.clear_set()
            self.robot.set_vel_acc(arm, velRatio, AccRatio)
            self.robot.send_cmd()

    # ============ 设置工具参数 ============
    def set_tool(self, arm=None, kine_para=None, dyn_para=None):
        """设置机械臂末端工具参数
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param kine_para: list(6,1). 运动学参数 XYZABC 单位毫米和度
        :param dyn_para: list(10,1). 动力学参数分别为 质量M  质心[3]:mx,my,mz 惯量I[6]:XX,XY,XZ,YY,YZ,ZZ
        :raises Exception: 如果操作失败
        """
        try:
            if kine_para is None:
                kine_para = [0.0] * 6
            if dyn_para is None:
                dyn_para = [0.0] * 10
            
            if len(kine_para) != 6:
                raise ValueError(f"kine_para must have 6 elements, got {len(kine_para)}")
            if len(dyn_para) != 10:
                raise ValueError(f"dyn_para must have 10 elements, got {len(dyn_para)}")
            
            if arm is None:
                # 两臂都设置
                arms = ['A', 'B']
            else:
                if arm not in ('A', 'B'):
                    raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
                arms = [arm]
            
            for a in arms:
                self.robot.clear_set()  # 清除设置
                self.robot.set_tool(a, kine_para, dyn_para)  # 设置工具参数
                self.robot.send_cmd()
                
                # 同步运动学（解算）端的工具参数，保证 FK/IK 计算基于最新工具
                kine = self._get_kine(a)
                tool_mat = kine.xyzabc_to_mat4x4(kine_para)
                if tool_mat is False or tool_mat is None:
                    raise Exception(f"set_tool failed for arm {a}: convert kine_para to 4x4 matrix failed")
                if not kine.set_tool_kine(tool_mat=tool_mat):
                    raise Exception(f"set_tool_kine failed for arm {a}")
        except Exception as e:
            raise Exception(f"set_tool failed: {e}")
        
    def set_joint_kd_params(self, arm=None, K=None, D=None):
        """设置关节阻尼参数
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param K: list(7,1). 刚度 牛米 / 度 。 设置每个轴的的力为刚度系数。 如K=[2，2,2,1,1,1,1]，第1到3轴有2N作为刚度系数参与控制计算，第4到7轴有1N作为刚度系数参与控制计算。
        :param D: list(7,1). 阻尼 牛米 / (度 / 秒)。 设置每个轴的的阻尼系数。1-7关节阻尼0-1之间
        :raises Exception: 如果操作失败
        """
        if K is None:
            K = [0.0] * 7
        if D is None:
            D = [0.0] * 7
        
        if len(K) != 7:
            raise ValueError(f"K must have 7 elements, got {len(K)}")
        if len(D) != 7:
            raise ValueError(f"D must have 7 elements, got {len(D)}")
        
        if arm is None:
            arms = ['A', 'B']
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
            arms = [arm]

        for a in arms:
            self.robot.clear_set()
            self.robot.set_joint_kd_params(a, K, D)
            self.robot.send_cmd()

    def set_cart_kd_params(self, arm=None, K=None, D=None, type=None):
        """设置坐标阻尼参数
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        # 在笛卡尔阻抗模式下：
            刚度系数： 1-3平移方向刚度系数不超过3000, 4-6旋转方向不超过100。 零空间刚度系数不超过20
            阻尼系数： 平移和旋转阻尼系数0-1之间。 零空间阻尼系数不超过1
            零空间控制是保持末端固定不动，手臂角度运动的控制方式。接口未开放
        :param arm: 机械手臂ID “A” OR “B”
        :param K: list(7,1). K[0]-k[2] N*m，x,y,z 平移方向每米的控制力; K[3]-k[5] N*m/rad, rx,ry,rz旋转弧度的控制力;K[6]N*m/rad,零空间总和刚度系数
        :param D: list(7,1). D[0]-D[5]  阻尼比例系数, D[6] 零空间总和阻尼比例系数,范围0-1
        :param type:int. set_A_arm_impedance_type设置的阻抗类型
        :raises Exception: 如果操作失败
        """
        if K is None:
            K = [0.0] * 6
        if D is None:
            D = [0.0] * 6
        if type is None:
            type = 0
        
        if len(K) != 6:
            raise ValueError(f"K must have 6 elements, got {len(K)}")
        if len(D) != 6:
            raise ValueError(f"D must have 6 elements, got {len(D)}")
        if arm is None:
            arms = ['A', 'B']
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
            arms = [arm]

        for a in arms:
            self.robot.clear_set()
            self.robot.set_cart_kd_params(a, K, D, type)
            self.robot.send_cmd()

    def set_force_control_params(self, arm=None, fcType=None, fxDirection=None, fcCtrlpara=None, fcAdjLmt=1):
        """设置力控参数
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param fcType: 力控类型 0:坐标空间力控;1:工具空间力控(暂未实现)
        :param fxDirection: list(6,1). 力控方向 需要控制方向设1，目前只支持 X,Y,Z控制方向.如力控方向为z,fxDirection=[0,0,1,0,0,0]
        :param fcCtrlpara: list(7,1). 控制参数 目前全0
        :param fcAdjLmt:毫米，允许的调节范围
        :raises Exception: 如果操作失败
        """
        if arm is None:
            arms = ['A', 'B']
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
            arms = [arm]
        if fcType is None:
            fcType = 0
        if fxDirection is None:
            fxDirection = [0.0] * 6
        if fcCtrlpara is None:
            fcCtrlpara = [0.0] * 7
        if fcAdjLmt is None:
            fcAdjLmt = 1
        
        if len(fxDirection) != 6:
            raise ValueError(f"fxDirection must have 6 elements, got {len(fxDirection)}")
        if len(fcCtrlpara) != 7:
            raise ValueError(f"fcCtrlpara must have 7 elements, got {len(fcCtrlpara)}")
        
        if fcType not in (0, 1):
            raise ValueError(f"fcType must be 0 or 1, got {fcType}")  # 力控类型 0:坐标空间力控;1:工具空间力控(暂未实现)

        for a in arms:
            self.robot.clear_set()
            self.robot.set_force_control_params(a, fcType, fxDirection, fcCtrlpara, fcAdjLmt)
            self.robot.send_cmd()
         
    def set_EefCart_control_params(self, arm=None, fcType=None, CartCtrlPara=None):
        """设置末端坐标控制参数
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param fcType:
              fcType=1，为自定义末端旋转方向。 笛卡尔方向：CartCtrlPara前三个参数置为末端基于基座X Y Z顺序的旋转，后四个为保留参数，填0
              fcType=2，为系统自动计算末端笛卡尔旋转。 CartCtrlPara全填0
        :param CartCtrlPara: list(7,1). 控制参数前三个为旋转信息，基于基座的XYZ旋转。
        :raises Exception: 如果操作失败
        """
        if arm is None:
            arms = ['A', 'B']
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
            arms = [arm]
        
        if fcType is None:
            fcType = 0
            CartCtrlPara = [0.0] * 7
        if CartCtrlPara is None:
            CartCtrlPara = [0.0] * 7
        
        if len(CartCtrlPara) != 7:
            raise ValueError(f"CartCtrlPara must have 7 elements, got {len(CartCtrlPara)}")
        
        if fcType not in (1, 2):
            raise ValueError(f"fcType must be 1 or 2, got {fcType}")  # 力控类型 0:坐标空间力控;1:工具空间力控(暂未实现)
        
        for a in arms:
            self.robot.clear_set()
            self.robot.set_EefCart_control_params(a, fcType, CartCtrlPara)
            self.robot.send_cmd()

    def set_joint_cmd_pose(self, arm, joints):
        """关节运动
        
        :param arm: 'A', 'B'
        :param joints: list(7,1). 关节角度，单位度,在位置跟随和扭矩模式下均有效
        :raises Exception: 如果操作失败
        """
        if arm not in ('A', 'B'):
            raise ValueError(f"arm must be 'A', 'B', got '{arm}'")
        if len(joints) != 7:
            raise ValueError(f"joints must have 7 elements, got {len(joints)}")
        
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm, joints)
        self.robot.send_cmd()

    def set_force(self, arm, force):
        """设置力
        
        :param arm: 'A', 'B'
        :param f: 目标力 单位牛或者牛米
        :raises Exception: 如果操作失败
        """
        if arm not in ('A', 'B'):
            raise ValueError(f"arm must be 'A', 'B', got '{arm}'")
        
        self.robot.clear_set()
        self.robot.set_force(arm, force)
        self.robot.send_cmd()

    def set_drag(self, arm=None, dgType=0):
        """设置拖拽空间
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param dgType:
                0 退出拖动模式
                1 关节空间拖动
                2 笛卡尔空间x方向拖动
                3 笛卡尔空间y方向拖动
                4 笛卡尔空间z方向拖动
                5 笛卡尔空间旋转方向拖动
        """
        if arm is None:
            arms = ['A', 'B']
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
            arms = [arm]
        if dgType not in (0, 1, 2, 3, 4, 5):
            raise ValueError(f"dgType must be 0, 1, 2, 3, 4, 5, got {dgType}")
        for a in arms:
            self.robot.clear_set()
            self.robot.set_drag_space(a, dgType)
            self.robot.send_cmd()

    def movel(self, arm, end_pose, vel=50, acc=100, blocking=True):
        """笛卡尔空间线性运动规划
        
        使用当前末端位姿作为起点，目标位姿为 end_pose。
        
        :param arm: 'A' 或 'B'
        :param end_pose: 目标末端位姿 [x, y, z, a, b, c]（mm 和度）
        :param vel:约束了输出的规划文件的速度。单位毫米/秒， 最小为0.1mm/s， 最大为1000 mm/s
        :param acc:约束了输出的规划文件的加速度。单位毫米/平方秒， 最小为0.1mm/s^2， 最大为1000 mm/s^2
        :param blocking: 是否阻塞等待运动完成，默认 True
        :raises Exception: 如果参数错误或运动失败
        """
        data, pset = self.movLA(arm, end_pose, vel, acc, 50)
        
        if pset is None or not data:
            raise Exception(f"movel planning failed for arm {arm}")
        
        if not self.robot.setPln_Cart(arm, pset):
            raise Exception(f"movel execution failed for arm {arm}")
        
        if blocking:
            self.wait_for_motion_complete(arm)
        
    def movej(self, arm, end_joints, vel_ratio=0.1, acc_ratio=0.1, blocking=True):
        """关节空间线性运动规划
        
        :param arm: 'A' 或 'B'
        :param end_joints: 目标关节角度，7个值（度）
        :param vel_ratio: 速度比例 0-1，默认 0.1（10%）
        :param acc_ratio: 加速度比例 0-1，默认 0.1（10%）
        :param blocking: 是否阻塞等待运动完成，默认 True
        :raises Exception: 如果规划或运动失败
        """
        try:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
            if len(end_joints) != 7:
                raise ValueError("end_joints must have 7 elements")
            
            start_joints = self.get_current_joints(arm)
            # 如果start_joints和end_joints相同，则无需运动，判断到小数点后2位
            if all(abs(s - e) < 0.01 for s, e in zip(start_joints, end_joints)):
                return
            
            if not self.pln_initialized:
                self._initialize_planning()
            
            if not self.robot.setPln_joint(arm, start_joints, end_joints, vel_ratio, acc_ratio):
                raise Exception(f"movej planning failed for arm {arm}")
            
            if blocking:
                self.wait_for_motion_complete(arm)
        except Exception as e:
            raise Exception(f"movej failed: {e}")
        
    def moveStop(self, arm):
        '''
        停止运动
        :param arm: 'A' 或 'B'
        '''
        self.robot.stopRunPln_joint(arm)

    def movel_all(self, target_pose_A: list[float],
                        target_pose_B: list[float],
                        vel: float,
                        acc: float, 
                        blocking=True):
        """笛卡尔空间线性运动规划
        
        使用当前末端位姿作为起点
        
        :param target_pose_A: 左臂目标末端位姿 [x, y, z, a, b, c]（mm 和度）
        :param target_pose_B: 右臂目标末端位姿 [x, y, z, a, b, c]（mm 和度）
        :param vel:约束了输出的规划文件的速度。单位毫米/秒， 最小为0.1mm/s， 最大为1000 mm/s
        :param acc:约束了输出的规划文件的加速度。单位毫米/平方秒， 最小为0.1mm/s^2， 最大为1000 mm/s^2
        :param blocking: 是否阻塞等待运动完成，默认 True
        :raises Exception: 如果参数
        """
        
        data_A, pset_A = self.movLA('A', target_pose_A, vel, acc, 50)
        data_B, pset_B = self.movLA('B', target_pose_B, vel, acc, 50)
        
        if not data_A or not data_B or not pset_A or not pset_B:
            raise Exception("movel_all planning failed")
        if not self.robot.setPln_Cart_AB(pset_A, pset_B):
            raise Exception("movel_all execution failed")
        if blocking:
            self.wait_for_motion_complete('A')
            self.wait_for_motion_complete('B')
        
    def movej_all(self, target_joints_A: list[float],
                        target_joints_B: list[float],
                        vel_ratio: float,
                        acc_ratio: float, 
                        blocking=True):
        """关节空间线性运动规划
        
        :param target_joints_A: 左臂目标关节角度，7个值（度）
        :param target_joints_B: 右臂目标关节角度，7个值（度）
        :param vel_ratio: 速度比例 0-1，默认 0.1（10%）
        :param acc_ratio: 加速度比例 0-1，默认 0.1（10%）
        :param blocking: 是否阻塞等待运动完成，默认 True
        :raises Exception: 如果规划或运动失败
        """
        if len(target_joints_A) != 7:
            raise ValueError("target_joints_A must have 7 elements")
        if len(target_joints_B) != 7:
            raise ValueError("target_joints_B must have 7 elements")
        if not self.pln_initialized:
            self._initialize_planning()
        
        start_joints_A = self.get_current_joints('A')
        start_joints_B = self.get_current_joints('B')

        if not self.robot.setPln_joint_AB(start_joints_A, target_joints_A, 
                                          start_joints_B, target_joints_B, 
                                          vel_ratio, acc_ratio):
            raise Exception("movej_all planning failed")
        if blocking:
            self.wait_for_motion_complete('A')
            self.wait_for_motion_complete('B')

    def moveStop_all(self):
        '''
        停止运动
        '''
        return self.robot.stopPln_AB()
    
    def wait_for_motion_complete(self, arm, timeout=60, poll_interval=0.1):
        """等待运动完成（阻塞）
        
        用关节速度和关节位置稳定性判断运动是否结束。

        :param arm: 'A' 或 'B'
        :param timeout: 超时时间（秒），默认 60
        :param poll_interval: 轮询间隔（秒），默认 0.1
        :return: True 表示完成，超时则抛出异常
        """
        start_time = time.time()
        arm_idx = 0 if arm == 'A' else 1
        stable_count = 0
        last_positions = None
        velocity_threshold = 0.5  # deg/s
        position_threshold = 0.05  # deg
        stable_required = 5

        while time.time() - start_time < timeout:
            sub_data = self.get_info()
            if sub_data is None:
                time.sleep(poll_interval)
                continue

            try:
                output = sub_data['outputs'][arm_idx]
                velocities = output['fb_joint_vel']
                positions = output['fb_joint_pos']
            except (KeyError, TypeError, IndexError):
                time.sleep(poll_interval)
                continue

            if len(velocities) < 7 or len(positions) < 7:
                time.sleep(poll_interval)
                continue

            max_speed = max(abs(v) for v in velocities[:7])
            current_positions = [float(p) for p in positions[:7]]

            if max_speed <= velocity_threshold:
                if last_positions is not None:
                    max_delta = max(abs(current_positions[i] - last_positions[i]) for i in range(7))
                else:
                    max_delta = 0.0

                if max_delta <= position_threshold:
                    stable_count += 1
                else:
                    stable_count = 0
            else:
                stable_count = 0

            last_positions = current_positions

            if stable_count >= stable_required:
                return True

            time.sleep(poll_interval)

        raise Exception(f"Motion did not complete within {timeout} seconds for arm {arm}")
    
    def movLA(self, arm, target_pose, vel, acc, freq_hz=50):
        if arm not in ('A', 'B'):
            raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
        if len(target_pose) != 6:
            raise ValueError("target_pose must have 6 elements [x, y, z, a, b, c]")
        if not self.pln_initialized:
            self._initialize_planning()
        
        start_joints = self.get_current_joints(arm)
        kine = self._get_kine(arm)
        fk_result = kine.fk(start_joints)
        if fk_result is False or fk_result is None:
            raise Exception("Failed to calculate current forward kinematics")
        start_pose = kine.mat4x4_to_xyzabc(fk_result)
        if start_pose is False or start_pose is None:
            raise Exception("Failed to convert current FK result to XYZABC")
        
        data, pset = kine.movLA(start_pose, target_pose, start_joints, vel, acc, freq_hz)
        if pset is None or not data:
            raise Exception(f"movel planning failed for arm {arm}")
        return data, pset

