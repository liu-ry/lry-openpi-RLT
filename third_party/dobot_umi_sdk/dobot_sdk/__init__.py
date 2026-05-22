#!/usr/bin/env python3
# coding=utf-8
"""
Dobot 越疆机械臂 SDK

第三方 SDK，原始代码来自越疆官方。
提供底层 TCP 通信接口和控制指令。

主要类:
    - DobotApi: 基础 TCP 通信类
    - DobotApiDashboard: 控制和运动指令接口
    - DobotApiFeedBack: 实时反馈数据接口
"""

from third_party.dobot_sdk.dobot_api import (
    DobotApi,
    DobotApiDashboard,
    DobotApiFeedBack,
    MyType,
    alarmAlarmJsonFile,
)

__all__ = [
    "DobotApi",
    "DobotApiDashboard",
    "DobotApiFeedBack",
    "MyType",
    "alarmAlarmJsonFile",
]
