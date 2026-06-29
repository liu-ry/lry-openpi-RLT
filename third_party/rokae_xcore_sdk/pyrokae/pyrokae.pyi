"""
Rokae SDK Python Wrapper
"""
from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['FrameType', 'Info', 'Load', 'MotionControlMode', 'OperateMode', 'OperationState', 'PowerState', 'RokaeAR']
class FrameType:
    """
    Members:
    
      world
    
      base
    
      flange
    
      tool
    
      wobj
    
      path
    
      rail
    """
    __members__: typing.ClassVar[dict[str, FrameType]]  # value = {'world': <FrameType.world: 0>, 'base': <FrameType.base: 1>, 'flange': <FrameType.flange: 2>, 'tool': <FrameType.tool: 3>, 'wobj': <FrameType.wobj: 4>, 'path': <FrameType.path: 5>, 'rail': <FrameType.rail: 6>}
    base: typing.ClassVar[FrameType]  # value = <FrameType.base: 1>
    flange: typing.ClassVar[FrameType]  # value = <FrameType.flange: 2>
    path: typing.ClassVar[FrameType]  # value = <FrameType.path: 5>
    rail: typing.ClassVar[FrameType]  # value = <FrameType.rail: 6>
    tool: typing.ClassVar[FrameType]  # value = <FrameType.tool: 3>
    wobj: typing.ClassVar[FrameType]  # value = <FrameType.wobj: 4>
    world: typing.ClassVar[FrameType]  # value = <FrameType.world: 0>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Info:
    id: str
    type: str
    version: str
    def __init__(self) -> None:
        ...
    @property
    def joint_num(self) -> int:
        ...
    @joint_num.setter
    def joint_num(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class Load:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, mass: typing.SupportsFloat | typing.SupportsIndex, cog: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"], inertia: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"]) -> None:
        ...
    @property
    def cog(self) -> typing.Annotated[list[float], "FixedSize(3)"]:
        ...
    @cog.setter
    def cog(self, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"]) -> None:
        ...
    @property
    def inertia(self) -> typing.Annotated[list[float], "FixedSize(3)"]:
        ...
    @inertia.setter
    def inertia(self, arg0: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"]) -> None:
        ...
    @property
    def mass(self) -> float:
        ...
    @mass.setter
    def mass(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class MotionControlMode:
    """
    Members:
    
      Idle
    
      NrtCommand
    
      NrtRLTask
    
      RtCommand
    """
    Idle: typing.ClassVar[MotionControlMode]  # value = <MotionControlMode.Idle: 0>
    NrtCommand: typing.ClassVar[MotionControlMode]  # value = <MotionControlMode.NrtCommand: 1>
    NrtRLTask: typing.ClassVar[MotionControlMode]  # value = <MotionControlMode.NrtRLTask: 2>
    RtCommand: typing.ClassVar[MotionControlMode]  # value = <MotionControlMode.RtCommand: 3>
    __members__: typing.ClassVar[dict[str, MotionControlMode]]  # value = {'Idle': <MotionControlMode.Idle: 0>, 'NrtCommand': <MotionControlMode.NrtCommand: 1>, 'NrtRLTask': <MotionControlMode.NrtRLTask: 2>, 'RtCommand': <MotionControlMode.RtCommand: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class OperateMode:
    """
    Members:
    
      manual
    
      automatic
    
      unknown
    """
    __members__: typing.ClassVar[dict[str, OperateMode]]  # value = {'manual': <OperateMode.manual: 0>, 'automatic': <OperateMode.automatic: 1>, 'unknown': <OperateMode.unknown: -1>}
    automatic: typing.ClassVar[OperateMode]  # value = <OperateMode.automatic: 1>
    manual: typing.ClassVar[OperateMode]  # value = <OperateMode.manual: 0>
    unknown: typing.ClassVar[OperateMode]  # value = <OperateMode.unknown: -1>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class OperationState:
    """
    Members:
    
      idle
    
      jog
    
      rtControlling
    
      drag
    
      rlProgram
    
      demo
    
      dynamicIdentify
    
      frictionIdentify
    
      loadIdentify
    
      moving
    
      jogging
    
      unknown
    """
    __members__: typing.ClassVar[dict[str, OperationState]]  # value = {'idle': <OperationState.idle: 0>, 'jog': <OperationState.jog: 1>, 'rtControlling': <OperationState.rtControlling: 2>, 'drag': <OperationState.drag: 3>, 'rlProgram': <OperationState.rlProgram: 4>, 'demo': <OperationState.demo: 5>, 'dynamicIdentify': <OperationState.dynamicIdentify: 6>, 'frictionIdentify': <OperationState.frictionIdentify: 7>, 'loadIdentify': <OperationState.loadIdentify: 8>, 'moving': <OperationState.moving: 9>, 'jogging': <OperationState.jogging: 10>, 'unknown': <OperationState.unknown: -1>}
    demo: typing.ClassVar[OperationState]  # value = <OperationState.demo: 5>
    drag: typing.ClassVar[OperationState]  # value = <OperationState.drag: 3>
    dynamicIdentify: typing.ClassVar[OperationState]  # value = <OperationState.dynamicIdentify: 6>
    frictionIdentify: typing.ClassVar[OperationState]  # value = <OperationState.frictionIdentify: 7>
    idle: typing.ClassVar[OperationState]  # value = <OperationState.idle: 0>
    jog: typing.ClassVar[OperationState]  # value = <OperationState.jog: 1>
    jogging: typing.ClassVar[OperationState]  # value = <OperationState.jogging: 10>
    loadIdentify: typing.ClassVar[OperationState]  # value = <OperationState.loadIdentify: 8>
    moving: typing.ClassVar[OperationState]  # value = <OperationState.moving: 9>
    rlProgram: typing.ClassVar[OperationState]  # value = <OperationState.rlProgram: 4>
    rtControlling: typing.ClassVar[OperationState]  # value = <OperationState.rtControlling: 2>
    unknown: typing.ClassVar[OperationState]  # value = <OperationState.unknown: -1>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PowerState:
    """
    Members:
    
      on
    
      off
    
      estop
    
      gstop
    
      unknown
    """
    __members__: typing.ClassVar[dict[str, PowerState]]  # value = {'on': <PowerState.on: 0>, 'off': <PowerState.off: 1>, 'estop': <PowerState.estop: 2>, 'gstop': <PowerState.gstop: 3>, 'unknown': <PowerState.unknown: -1>}
    estop: typing.ClassVar[PowerState]  # value = <PowerState.estop: 2>
    gstop: typing.ClassVar[PowerState]  # value = <PowerState.gstop: 3>
    off: typing.ClassVar[PowerState]  # value = <PowerState.off: 1>
    on: typing.ClassVar[PowerState]  # value = <PowerState.on: 0>
    unknown: typing.ClassVar[PowerState]  # value = <PowerState.unknown: -1>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RokaeAR:
    @typing.overload
    def __init__(self, remoteIP: str = '', localIP: str = '') -> None:
        ...
    @typing.overload
    def __init__(self, remoteIP: str) -> None:
        ...
    def disableCollisionDetection(self) -> None:
        """
        Disable collision detection
        """
    def disableForceControl(self) -> None:
        """
        Disable force control
        """
    def disableRealtimeMotion(self) -> None:
        """
        Disable realtime motion
        """
    def enableCollisionDetection(self, sensitivity: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(7)"], fallback_compliance: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Enable collision detection
        """
    def enableForceControl(self, frame_type: FrameType, control_type: typing.SupportsInt | typing.SupportsIndex, stiffness: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(6)"]) -> None:
        """
        Enable force control
        """
    def enableRealtimeMotion(self, dt: typing.SupportsFloat | typing.SupportsIndex = 0.008, ServoJ_Lookahead: typing.SupportsFloat | typing.SupportsIndex = -1, ServoJ_Kp: typing.SupportsFloat | typing.SupportsIndex = -1) -> None:
        """
        Enable realtime motion
        """
    def forwardKinematics(self, joints: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(7)"]) -> typing.Annotated[list[float], "FixedSize(6)"]:
        """
        Forward kinematics
        """
    def getBaseFrame(self) -> typing.Annotated[list[float], "FixedSize(6)"]:
        """
        Get base frame
        """
    def getEndPose(self) -> typing.Annotated[list[float], "FixedSize(6)"]:
        """
        Get end pose
        """
    def getFlangePose(self) -> typing.Annotated[list[float], "FixedSize(6)"]:
        """
        Get flange pose
        """
    def getJointPos(self) -> typing.Annotated[list[float], "FixedSize(7)"]:
        """
        Get joint positions
        """
    def getJointTorque(self) -> typing.Annotated[list[float], "FixedSize(7)"]:
        """
        Get joint torques
        """
    def getJointVel(self) -> typing.Annotated[list[float], "FixedSize(7)"]:
        """
        Get joint velocities
        """
    def getLoad(self) -> Load:
        """
        Get load
        """
    def getOperateMode(self) -> OperateMode:
        """
        Get operate mode
        """
    def getOperationState(self) -> OperationState:
        """
        Get operation state
        """
    def getPowerState(self) -> PowerState:
        """
        Get power state
        """
    def getTcpOffset(self) -> typing.Annotated[list[float], "FixedSize(6)"]:
        """
        Get TCP offset
        """
    def inverseKinematics(self, pose: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(6)"]) -> typing.Annotated[list[float], "FixedSize(7)"]:
        """
        Inverse kinematics
        """
    def moveJ_joint(self, joint: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(7)"], speed: typing.SupportsFloat | typing.SupportsIndex = -1, zone: typing.SupportsFloat | typing.SupportsIndex = -1) -> None:
        """
        Move joint (joint)
        """
    def movePause(self) -> None:
        """
        Pause motion
        """
    def moveReset(self) -> None:
        """
        Reset motion
        """
    def moveStop(self) -> None:
        """
        Stop motion
        """
    def moveWait(self, timeout: typing.SupportsFloat | typing.SupportsIndex = -1) -> None:
        """
        Wait for motion to finish
        """
    def recoverState(self) -> None:
        """
        Recover from error state
        """
    def robotInfo(self) -> Info:
        """
        Get robot info
        """
    def sdkVersion(self) -> str:
        """
        Get SDK version
        """
    def servoJ(self, joint: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(7)"], ServoJ_T: typing.SupportsFloat | typing.SupportsIndex = -1, ServoJ_Lookahead: typing.SupportsFloat | typing.SupportsIndex = -1, ServoJ_Kp: typing.SupportsFloat | typing.SupportsIndex = -1) -> None:
        """
        ServoJ command
        """
    def setAcceleration(self, acc: typing.SupportsFloat | typing.SupportsIndex, jerk: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set acceleration and jerk
        """
    def setCartesianDesiredForce(self, force: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(6)"]) -> None:
        """
        Set desired force
        """
    def setDefaultSpeed(self, level: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set default speed
        """
    def setDefaultZone(self, level: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Set default zone
        """
    def setDragMode(self, enable: bool) -> None:
        """
        Enable/disable drag mode
        """
    def setLoad(self, mass: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Set load (mass only)
        """
    def setLoadFull(self, mass: typing.SupportsFloat | typing.SupportsIndex, cog: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"], inertia: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"]) -> None:
        """
        Set load (mass, cog, inertia)
        """
    def setMotionControlMode(self, mode: MotionControlMode) -> None:
        """
        Set motion control mode
        """
    def setOperationMode(self, mode: OperateMode) -> None:
        """
        Set operation mode
        """
    def setPower(self, enable: bool) -> None:
        """
        Set power on/off
        """
    def setSpeedOnline(self, speed: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Adjust speed online
        """
    def setTcpOffset(self, tcp_offset: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(6)"]) -> None:
        """
        Set TCP offset
        """
