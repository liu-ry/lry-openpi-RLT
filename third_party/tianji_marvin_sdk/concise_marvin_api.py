from .fx_robot import Concise_Marvin_Robot, DCSS
from .fx_kine import Marvin_Kine
import time
import os


class MarvinSDK:
    """天机-孚晞 MARVIN 机器人 SDK 简明高级接口
    
    集成控制接口和运动学接口，提供简化的 API 调用方式
    """
    
    def __init__(self, ip:str, config_path:str=None):
        """初始化 MarvinSDK
        
        :param ip: 机器人 IP 地址
        :param config_path: 可选的运动学配置文件路径，默认同目录下 ccs_m6_40
        """
        self.robot = Concise_Marvin_Robot()
        self.kine_a = Marvin_Kine()
        self.kine_b = Marvin_Kine()
        self.dcss = DCSS()
        self.ip = ip
        self.pln_initialized = False

        default_path = os.path.join(os.path.dirname(__file__), 'ccs_m6_40.MvKDCfg')
        self.config_path = config_path if config_path else default_path

        self._initialize_kinematics()

    def connect(self):
        '''连接到机器人并验证连接成功
        
        如果连接失败或数据通道连接失败，抛出异常
        '''
        init = self.robot.connect(self.ip, 0)
        if not init:
            raise Exception('failed to connect to the robot, port is occupied')

        '''检查机械臂和伺服当前是否存错误，有错误清错'''
        if not self.robot.check_error_and_clear():
            raise Exception('robot error detected, please clear the error first')

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
        
        self._initialize_planning()
        sdk_version = self.robot.SDK_version()
        ret, version = self.robot.get_param('int', 'VERSION')
        if ret < 0:
            print(f"Failed to get Controller version")
        print(f"Robot connected successfully, SDK version: {sdk_version}, Controller version: {version}")
        
    def get_info(self):
        """获取机器人当前状态信息"""
        return self.robot.subscribe(self.dcss)

    def get_current_joints(self, arm):
        """获取当前关节位置（7 个值，单位：度）

        :param arm: 'A' 或 'B'
        :return: 当前关节角度列表
        :raises Exception: 当读取失败时抛出
        """
        return self._get_current_joints(arm)

    def get_current_tcppose(self, arm):
        """获取当前末端 TCP 位姿 [x, y, z, a, b, c]（mm 和度）

        :param arm: 'A' 或 'B'
        :return: 当前 TCP 位姿
        :raises Exception: 当读取失败时抛出
        """
        joints = self._get_current_joints(arm)
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
    
    def ik(self, arm, pose, ref_joints=None):
        """逆向运动学：从末端位姿计算关节角度
        
        :param pose: 末端位姿列表 [x, y, z, a, b, c] (mm 和度)
        :param arm: 'A' 或 'B'
        :param ref_joints: 参考关节角度，7个值（度），用于约束解的构型，可选
        :return: 关节角度列表（7个值，度）
        :raises Exception: 如果计算失败（无可达解、奇异等）
        """
        from .fx_kine import FX_InvKineSolvePara
        
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
            current_joints = self._get_current_joints(arm)
            sp.set_input_ik_ref_joint(current_joints)
        
        # 执行 IK 计算
        result:FX_InvKineSolvePara|bool = kine.ik(sp)
        if result is False:
            # 检查失败原因
            if sp.m_Output_IsOutRange:
                raise Exception(f"IK failed: pose is out of reach for arm {arm}")
            if sp.m_Output_IsDeg[3]:
                raise Exception(f"IK failed: joint 4 singularity for arm {arm}")
            if sp.m_Output_IsJntExd:
                raise Exception(f"IK failed: over limit for arm {arm}")
            raise Exception(f"IK calculation failed for arm {arm}")
        else:
            return result.m_Output_RetJoint.to_list()
    
    def release(self):
        """释放机器人连接"""
        time.sleep(0.5)
        return self.robot.release_robot()

    # ============ 软急停 ============
    def soft_stop(self, arm='AB'):
        """机械臂软急停
        
        :param arm: 'A', 'B', 或 'AB'，指定哪条臂进行软急停
        :raises Exception: 如果操作失败
        """
        if arm not in ('A', 'B', 'AB'):
            raise ValueError(f"arm must be 'A', 'B', or 'AB', got '{arm}'")
        try:
            self.robot.soft_stop(arm)
        except Exception as e:
            raise Exception(f"soft_stop failed: {e}")

    # ============ 获取伺服错误码 ============
    def get_servo_error_code(self, arm=None, lang='CN'):
        """获取机械臂伺服错误码
        
        :param arm: 'A' 或 'B'，如果为 None 则获取两臂错误码
        :param lang: 'CN' 或 'EN'
        :return: 错误码列表或字典（两臂情况）
        :raises Exception: 如果操作失败
        """
        if arm is None:
            # 获取两臂错误码
            try:
                err_a = self.robot.get_servo_error_code('A', lang)
                err_b = self.robot.get_servo_error_code('B', lang)
                return {'A': err_a, 'B': err_b}
            except Exception as e:
                raise Exception(f"get_servo_error_code failed: {e}")
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
            try:
                return self.robot.get_servo_error_code(arm, lang)
            except Exception as e:
                raise Exception(f"get_servo_error_code failed for arm {arm}: {e}")

    # ============ 伺服软复位 ============
    def servo_reset(self, arm=None, axis=None):
        """指定轴伺服软复位
        
        :param arm: 'A', 'B', 或 None（None 则两臂都复位）
        :param axis: 关节轴 0-6，或 None（None 则所有轴都复位）
        :raises Exception: 如果操作失败
        """
        try:
            if arm is None:
                # 两臂都复位
                arms = ['A', 'B']
            else:
                if arm not in ('A', 'B'):
                    raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
                arms = [arm]
            
            if axis is None:
                # 所有轴都复位
                axes = list(range(7))
            else:
                if not isinstance(axis, int) or axis < 0 or axis > 6:
                    raise ValueError(f"axis must be 0-6, got {axis}")
                axes = [axis]
            
            for a in arms:
                for ax in axes:
                    self.robot.servo_reset(a, ax)
        except Exception as e:
            raise Exception(f"servo_reset failed: {e}")

    # ============ 设置工具参数 ============
    def set_tool(self, arm=None, kine_para=None, dyn_para=None):
        """设置机械臂末端工具参数
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param kine_para: 工具运动学参数，长度 6（mm 和度），默认 [0,0,0,0,0,0]
        :param dyn_para: 工具动力学参数，长度 10，默认 [0,...,0]
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
                if not self.robot.set_tool(a, kine_para, dyn_para):
                    raise Exception(f"set_tool failed for arm {a}")
                # 同步运动学（解算）端的工具参数，保证 FK/IK 计算基于最新工具
                kine = self._get_kine(a)
                tool_mat = kine.xyzabc_to_mat4x4(kine_para)
                if tool_mat is False or tool_mat is None:
                    raise Exception(f"set_tool failed for arm {a}: convert kine_para to 4x4 matrix failed")
                if not kine.set_tool_kine(tool_mat=tool_mat):
                    raise Exception(f"set_tool_kine failed for arm {a}")
        except Exception as e:
            raise Exception(f"set_tool failed: {e}")

    # ============ 切换控制状态 ============
    def set_position_state(self, arm=None, vel_ratio=10, acc_ratio=10):
        """设置位置模式（高刚度）
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param vel_ratio: 速度百分比 0-100，默认 10
        :param acc_ratio: 加速度百分比 0-100，默认 10
        :raises Exception: 如果设置失败
        """
        try:
            if arm is None:
                arms = ['A', 'B']
            else:
                if arm not in ('A', 'B'):
                    raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
                arms = [arm]
            
            for a in arms:
                if not self.robot.set_position_state(a, vel_ratio, acc_ratio):
                    raise Exception(f"set_position_state failed for arm {a}")
            time.sleep(0.5)
        except Exception as e:
            raise Exception(f"set_position_state failed: {e}")

    def set_imp_joint_state(self, arm=None, vel_ratio=10, acc_ratio=10, K=None, D=None):
        """设置关节阻抗模式
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param vel_ratio: 速度百分比 0-100，默认 10
        :param acc_ratio: 加速度百分比 0-100，默认 10
        :param K: 刚度系数，7个值，默认 [2,2,2,1,1,1,1]
        :param D: 阻尼系数，7个值 0-1，默认 [0.6,0.6,0.6,0.4,0.2,0.2,0.2]
        :raises Exception: 如果设置失败
        """
        try:
            if K is None:
                K = [2, 2, 2, 1, 1, 1, 1]
            if D is None:
                D = [0.6, 0.6, 0.6, 0.4, 0.2, 0.2, 0.2]
            
            if arm is None:
                arms = ['A', 'B']
            else:
                if arm not in ('A', 'B'):
                    raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
                arms = [arm]
            
            for a in arms:
                if not self.robot.set_imp_joint_state(a, vel_ratio, acc_ratio, K, D):
                    raise Exception(f"set_imp_joint_state failed for arm {a}")
            time.sleep(0.5)
        except Exception as e:
            raise Exception(f"set_imp_joint_state failed: {e}")

    def set_imp_cart_state(self, arm=None, vel_ratio=10, acc_ratio=10, K=None, D=None, rot_type=0, cart_ctrl_para=None):
        """设置笛卡尔阻抗模式
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param vel_ratio: 速度百分比 0-100，默认 10
        :param acc_ratio: 加速度百分比 0-100，默认 10
        :param K: 刚度系数，7个值，默认 [1000,1000,1000,50,50,50,10]
        :param D: 阻尼系数，7个值 0-1，默认 [0.6,0.6,0.6,0.3,0.3,0.3,0.3]
        :param rot_type: 旋转模式 0/1/2
        :param cart_ctrl_para: 笛卡尔控制参数，7个值，默认 [0,0,0,0,0,0,0]
        :raises Exception: 如果设置失败
        """
        try:
            if K is None:
                K = [1000, 1000, 1000, 50, 50, 50, 10]
            if D is None:
                D = [0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.3]
            if cart_ctrl_para is None:
                cart_ctrl_para = [0.0] * 7
            
            if arm is None:
                arms = ['A', 'B']
            else:
                if arm not in ('A', 'B'):
                    raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
                arms = [arm]
            
            for a in arms:
                if not self.robot.set_imp_cart_state(a, vel_ratio, acc_ratio, K, D, rot_type, cart_ctrl_para):
                    raise Exception(f"set_imp_cart_state failed for arm {a}")
            time.sleep(0.5)
        except Exception as e:
            raise Exception(f"set_imp_cart_state failed: {e}")

    def set_imp_force_state(self, arm=None, fx_dir=None, fc_adj_lmt=10.0):
        """设置力控模式
        
        :param arm: 'A', 'B', 或 None（None 则两臂都设置）
        :param fx_dir: 力控方向，6个值，默认 [0,0,1,0,0,0]（Z轴）
        :param fc_adj_lmt: 调节范围（mm），默认 10
        :raises Exception: 如果设置失败
        """
        try:
            if fx_dir is None:
                fx_dir = [0, 0, 1, 0, 0, 0]
            
            if arm is None:
                arms = ['A', 'B']
            else:
                if arm not in ('A', 'B'):
                    raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
                arms = [arm]
            
            for a in arms:
                if not self.robot.set_imp_force_state(a, fx_dir, fc_adj_lmt):
                    raise Exception(f"set_imp_force_state failed for arm {a}")
            time.sleep(0.5)
        except Exception as e:
            raise Exception(f"set_imp_force_state failed: {e}")

    def disable(self, arm=None):
        """下使能/复位指定手臂
        
        :param arm: 'A', 'B', 或 None（None 则两臂都复位）
        :raises Exception: 如果操作失败
        """
        try:
            if arm is None:
                arms = ['A', 'B']
            else:
                if arm not in ('A', 'B'):
                    raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
                arms = [arm]
            
            for a in arms:
                if not self.robot.disable(a):
                    raise Exception(f"disable failed for arm {a}")
        except Exception as e:
            raise Exception(f"disable failed: {e}")

    # ============ 设置关节指令 ============
    def set_joint_position_cmd(self, arm, joints):
        """设置关节位置指令
        
        :param arm: 'A' 或 'B'
        :param joints: 7个关节角度（度）
        :raises Exception: 如果设置失败
        """
        try:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
            if len(joints) != 7:
                raise ValueError(f"joints must have 7 elements, got {len(joints)}")
            
            if not self.robot.set_joint_position_cmd(arm, joints):
                raise Exception(f"set_joint_position_cmd failed for arm {arm}")
        except Exception as e:
            raise Exception(f"set_joint_position_cmd failed: {e}")

    # ============ 力控指令 ============
    def set_force_cmd(self, arm, force):
        """设置力控指令（牛或牛米）
        
        :param arm: 'A' 或 'B'
        :param force: 目标力（可以是任意实数）
        :raises Exception: 如果设置失败
        """
        try:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
            
            if not self.robot.set_force_cmd(arm, force):
                raise Exception(f"set_force_cmd failed for arm {arm}")
        except Exception as e:
            raise Exception(f"set_force_cmd failed: {e}")

    # ============ 离线轨迹相关 ============
    def send_pvt(self, arm, local_file, serial):
        """上传本地 PVT 轨迹文件
        
        :param arm: 'A' 或 'B'
        :param local_file: 轨迹文件路径
        :param serial: 轨迹 ID（0-99）
        :raises Exception: 如果上传失败
        """
        try:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
            if not (0 <= serial <= 99):
                raise ValueError(f"serial must be 0-99, got {serial}")
            
            if not self.robot.send_pvt(arm, local_file, serial):
                raise Exception(f"send_pvt failed for arm {arm}")
        except Exception as e:
            raise Exception(f"send_pvt failed: {e}")

    def run_pvt(self, arm, id_):
        """运行指定 ID 的 PVT 轨迹
        
        :param arm: 'A' 或 'B'
        :param id_: 轨迹 ID（0-99）
        :raises Exception: 如果运行失败
        """
        try:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
            if not (0 <= id_ <= 99):
                raise ValueError(f"id must be 0-99, got {id_}")
            
            if not self.robot.run_pvt(arm, id_):
                raise Exception(f"run_pvt failed for arm {arm}")
        except Exception as e:
            raise Exception(f"run_pvt failed: {e}")

    # ============ 拖动示教相关 ============
    def set_joint_drag(self, arm=None):
        """进入关节拖动模式
        
        :param arm: 'A', 'B', 或 None（None 则两臂都进入）
        :raises Exception: 如果操作失败
        """
        try:
            if arm is None:
                arms = ['A', 'B']
            else:
                if arm not in ('A', 'B'):
                    raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
                arms = [arm]
            
            for a in arms:
                if not self.robot.set_joint_drag(a):
                    raise Exception(f"set_joint_drag failed for arm {a}")
        except Exception as e:
            raise Exception(f"set_joint_drag failed: {e}")

    def set_cart_drag(self, arm=None, direction='Z'):
        """进入笛卡尔拖动模式
        
        :param arm: 'A', 'B', 或 None（None 则两臂都进入）
        :param direction: 拖动方向 'X', 'Y', 'Z', 或 'R'（旋转），默认 'Z'
        :raises Exception: 如果操作失败
        """
        try:
            if arm is None:
                arms = ['A', 'B']
            else:
                if arm not in ('A', 'B'):
                    raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
                arms = [arm]
            
            if direction not in ('X', 'Y', 'Z', 'R'):
                raise ValueError(f"direction must be 'X', 'Y', 'Z', or 'R', got '{direction}'")
            
            for a in arms:
                if not self.robot.set_cart_drag(a, direction):
                    raise Exception(f"set_cart_drag failed for arm {a}")
        except Exception as e:
            raise Exception(f"set_cart_drag failed: {e}")

    def exit_drag(self, arm=None):
        """退出拖动模式
        
        :param arm: 'A', 'B', 或 None（None 则两臂都退出）
        :raises Exception: 如果操作失败
        """
        if arm is None:
            arms = ['A', 'B']
        else:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A', 'B', or None, got '{arm}'")
            arms = [arm]
        
        failed = []
        for a in arms:
            try:
                if not self.robot.exit_drag(a):
                    failed.append(a)
            except Exception as e:
                failed.append(f"{a}({e})")
        
        if failed:
            raise Exception(f"exit_drag failed for arms: {failed}")

    # ============ 规划运行 - MOVEJ ============
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
            
            start_joints = self._get_current_joints(arm)
            # 如果start_joints和end_joints相同，则无需运动，判断到小数点后2位
            if all(abs(s - e) < 0.01 for s, e in zip(start_joints, end_joints)):
                return
            
            if not self.pln_initialized:
                self._initialize_planning()
            
            if not self.robot.run_pln_joint(arm, start_joints, end_joints, vel_ratio, acc_ratio):
                raise Exception(f"movej planning failed for arm {arm}")
            
            if blocking:
                self._wait_for_motion_complete(arm)
        except Exception as e:
            raise Exception(f"movej failed: {e}")

    # ============ 规划运行 - MOVEL ============
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
        try:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
            if len(end_pose) != 6:
                raise ValueError("end_pose must have 6 elements [x, y, z, a, b, c]")
            
            start_joints = self._get_current_joints(arm)
            kine = self._get_kine(arm)
            fk_result = kine.fk(start_joints)
            if fk_result is False or fk_result is None:
                raise Exception("Failed to calculate current forward kinematics")
            start_pose = kine.mat4x4_to_xyzabc(fk_result)
            if start_pose is False or start_pose is None:
                raise Exception("Failed to convert current FK result to XYZABC")
            
            if not self.pln_initialized:
                self._initialize_planning()
            
            data, pset = kine.movLA(start_pose, end_pose, start_joints,
                                     vel, acc, freq_hz=50)
            
            if pset is None or not data:
                raise Exception(f"movel planning failed for arm {arm}")
            
            if not self.robot.run_pln_cart(arm, pset):
                raise Exception(f"movel execution failed for arm {arm}")
            
            if blocking:
                self._wait_for_motion_complete(arm)
        except Exception as e:
            raise Exception(f"movel failed: {e}")

    def stop_pln(self, arm):
        """停止规划运动
        
        :param arm: 'A' 或 'B'
        :raises Exception: 如果操作失败
        """
        try:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
            
            if not self.robot.stop_pln(arm):
                raise Exception(f"stop_pln failed for arm {arm}")
        except Exception as e:
            raise Exception(f"stop_pln failed: {e}")
        
    def movl(self, arm, end_pose, vel=50, acc=100, freq_hz=500):
        '''
        移动到指定位置，使用当前末端位姿作为起点，目标位姿为 end_pose。
        
        :param arm: 'A' 或 'B'
        :param end_pose: 目标末端位姿 [x, y, z, a, b, c]（mm 和度）
        :param vel:约束了输出的规划文件的速度。单位毫米/秒， 最小为0.1mm/s， 最大为1000 mm/s
        :param acc:约束了输出的规划文件的加速度。单位毫米/平方秒， 最小为0.1mm/s^2， 最大为1000 mm/s^2
        :param freq_hz: 轨迹点发送频率，单位 Hz，默认 500 Hz，不能超过 1000 Hz
        :raises Exception: 如果参数错误或运动失败
        '''
        try:
            if arm not in ('A', 'B'):
                raise ValueError(f"arm must be 'A' or 'B', got '{arm}'")
            if len(end_pose) != 6:
                raise ValueError("end_pose must have 6 elements [x, y, z, a, b, c]")
            
            start_joints = self._get_current_joints(arm)
            kine = self._get_kine(arm)
            fk_result = kine.fk(start_joints)
            if fk_result is False or fk_result is None:
                raise Exception("Failed to calculate current forward kinematics")
            start_pose = kine.mat4x4_to_xyzabc(fk_result)
            if start_pose is False or start_pose is None:
                raise Exception("Failed to convert current FK result to XYZABC")
            
            data, pset = kine.movLA(start_pose, end_pose, start_joints,
                                     vel, acc, freq_hz)
            
            if pset is None or not data:
                raise Exception(f"movel planning failed for arm {arm}")
            
            for point in data:
                self.robot.set_joint_position_cmd(arm, point)
                time.sleep(1/freq_hz)
        except Exception as e:
            raise Exception(f"movl failed: {e}")

    # ============ 辅助方法 ============
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

    def _wait_for_motion_complete(self, arm, timeout=60, poll_interval=0.1):
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

    def _get_current_joints(self, arm):
        """获取当前关节位置
        
        :param arm: 'A' 或 'B'
        :return: 关节角度列表（7个值）
        """
        sub_data = self.get_info()
        if sub_data is None:
            raise Exception("Failed to get robot state")
        
        arm_idx = 0 if arm == 'A' else 1
        # 从订阅数据中提取当前关节位置
        try:
            current_joints = sub_data['outputs'][arm_idx]['fb_joint_pos']
            return list(current_joints[:7])
        except (KeyError, TypeError, IndexError):
            raise Exception(f"Failed to extract current joint positions for arm {arm}")





