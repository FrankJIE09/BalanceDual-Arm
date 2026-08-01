#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轮式移动双臂机器人协同稳载控制系统
===============================================
项目：BalanceDual-Arm
任务：双任务并行控制
  1. 搬水稳水平（最高优先级，硬约束）
  2. 板面小球驻留（次级优化，零空间柔顺）

控制架构（5层严格层级化）：
  第0层：平板中心速度旋量 (v_plate, ω_plate) ← 顶层输入
  第1层：底盘姿态补偿 (ω_comp = -K * [roll, pitch, 0])
  第2层：速度旋量坐标变换 (plate → left/right TCP)
  第3层：雅可比伪逆解算 (dq = J⁺ * twist)
  第4层：数值积分 + 限位 (q_target = q + dq*dt)
  第5层：position 执行器驱动 (ctrl[] = q_target)

仿真规则：
  - 全部关节使用 position 位置型执行器
  - 禁止直接写 qpos 赋值控制机器人
  - 禁止纯速度开环控制
  - 速度积分转位置，MuJoCo PID 闭环跟踪

依赖：
  pip install mujoco numpy scipy

运行：
  python dual_arm_controller.py              # 无GUI模式
  python dual_arm_controller.py --gui        # 带GUI可视化
  python dual_arm_controller.py --gui --step # 逐步调试模式

版本：v1.0
日期：2026-07-31
"""

import mujoco
import mujoco.viewer
import numpy as np
import argparse
import os
import sys
import time

# =====================================================================
# 常量定义
# =====================================================================

# 执行器 ID 映射（对应 scene_dual_arm_plate.xml 中的执行器顺序）
ACT_LEGS_START  = 0       # 腿部 8 个执行器 (0-7)
ACT_TRUNK_START = 8       # 躯干 4 个执行器 (8-11)
ACT_LEFT_ARM_START  = 12  # 左臂 7 个执行器 (12-18) ★
ACT_LEFT_FINGER     = 19  # 左手指执行器 (19)
ACT_RIGHT_ARM_START = 20  # 右臂 7 个执行器 (20-26) ★
ACT_RIGHT_FINGER    = 27  # 右手指执行器 (27)
ACT_HEAD_START      = 28  # 头部 2 个执行器 (28-29)

LEFT_ARM_ACT_IDS  = list(range(ACT_LEFT_ARM_START,  ACT_LEFT_ARM_START + 7))   # [12,13,...,18]
RIGHT_ARM_ACT_IDS = list(range(ACT_RIGHT_ARM_START, ACT_RIGHT_ARM_START + 7))  # [20,21,...,26]

# 关节名称列表
LEFT_ARM_JOINT_NAMES = [f"left_arm_joint_{i}" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"right_arm_joint_{i}" for i in range(1, 8)]

# =====================================================================
# 控制器类
# =====================================================================

class DualArmPlateController:
    """双臂协同稳载控制器"""

    def __init__(self, model, data, args):
        """
        初始化控制器

        Args:
            model: MuJoCo MjModel
            data:   MuJoCo MjData
            args:   命令行参数 Namespace
        """
        self.model = model
        self.data = data
        self.args = args

        # ---- 控制参数 ----
        self.dt = model.opt.timestep           # 仿真步长
        self.kp_attitude = args.kp_attitude     # 姿态补偿增益
        self.kp_ball = args.kp_ball             # 小球驻留增益
        self.dls_lambda = 0.05                  # 阻尼最小二乘阻尼因子
        self.enable_ball = not args.no_ball     # 是否启球小球驻留
        self.finger_grip = 0.015                # 手指夹紧位置 (0=张开, 0.023=全闭)

        # ---- 运动学 ID 映射 ----
        self._init_kinematics()

        # ---- 状态变量 ----
        self.step_count = 0
        self.plate_desired_pos = np.array([0.35, 0.0, 0.98])  # 平板期望位置
        self.prev_dq_left = np.zeros(7)   # 上一帧左臂关节速度
        self.prev_dq_right = np.zeros(7)  # 上一帧右臂关节速度

        # ---- 日志 ----
        self.log_interval = max(1, int(0.5 / self.dt))  # 每 0.5 秒打印一次

    def _init_kinematics(self):
        """初始化运动学链：获取关节/刚体/执行器 ID"""
        model = self.model

        # 关节 DOF 地址
        self.left_joint_ids = []
        self.right_joint_ids = []
        for name in LEFT_ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.left_joint_ids.append(jid)
        for name in RIGHT_ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.right_joint_ids.append(jid)

        # 执行器 ID
        self.left_act_ids = LEFT_ARM_ACT_IDS
        self.right_act_ids = RIGHT_ARM_ACT_IDS

        # 刚体 ID
        self.tcp_left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Joint7_L")
        self.tcp_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Joint7_R")
        self.plate_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self.mocap_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate_mocap")
        # mocap 体在 data.mocap_pos 中的索引不同于 body ID
        self.mocap_idx    = model.body_mocapid[self.mocap_id]
        self.ball_id      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.trunk_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Trunk4")

        # 关节范围（用于限位）
        self.left_joint_ranges  = np.array([model.jnt_range[j][:] for j in self.left_joint_ids])
        self.right_joint_ranges = np.array([model.jnt_range[j][:] for j in self.right_joint_ids])

        print(f"[Init] 左臂关节 DOF: {self.left_joint_ids}")
        print(f"[Init] 右臂关节 DOF: {self.right_joint_ids}")
        print(f"[Init] 左臂执行器 ID: {self.left_act_ids}")
        print(f"[Init] 右臂执行器 ID: {self.right_act_ids}")

    # ==================================================================
    # 第0层：读取状态
    # ==================================================================

    def get_chassis_orientation(self):
        """
        读取底盘（Trunk4）俯仰角和滚转角

        Returns:
            roll, pitch (radians)
        """
        xmat = self.data.xmat[self.trunk_id].reshape(3, 3)
        # 从旋转矩阵提取欧拉角 (ZYX = yaw, pitch, roll)
        # xmat 列向量：第0列=X轴，第1列=Y轴，第2列=Z轴
        # 但 MuJoCo 的 xmat 按行存储

        # 底盘 Z 轴在世界坐标系中的投影
        z_axis = xmat[:, 2]  # 底盘局部 Z 轴在世界系中的方向

        # pitch: 绕 Y 轴旋转（前后倾）
        pitch = np.arctan2(z_axis[0], z_axis[2])

        # roll: 绕 X 轴旋转（左右倾）
        roll = np.arctan2(-z_axis[1],
                          np.sqrt(z_axis[0]**2 + z_axis[2]**2))

        return roll, pitch

    def get_tcp_world_pose(self, body_id):
        """获取 TCP 刚体在世界坐标系中的位置和朝向"""
        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        return pos, xmat

    def get_current_joint_positions(self, side='left'):
        """读取当前关节位置 qpos"""
        ids = self.left_joint_ids if side == 'left' else self.right_joint_ids
        return np.array([self.data.qpos[self.model.jnt_qposadr[j]] for j in ids])

    # ==================================================================
    # 第1层：底盘姿态补偿
    # ==================================================================

    def compute_attitude_compensation(self):
        """
        计算保持平板绝对水平所需的补偿角速度

        原理：
          读取底盘俯仰/滚转角，产生反向角速度使平板恢复水平。
          ω_comp = -K_p * [roll, pitch, 0]

        Returns:
            w_comp:  补偿角速度向量 [wx, wy, wz] (world frame)
            roll:    当前滚转角 (rad)
            pitch:   当前俯仰角 (rad)
        """
        roll, pitch = self.get_chassis_orientation()

        w_comp = np.zeros(3)
        w_comp[0] = -self.kp_attitude * roll   # 绕世界 X 轴，补偿滚转
        w_comp[1] = -self.kp_attitude * pitch  # 绕世界 Y 轴，补偿俯仰
        w_comp[2] = 0.0                         # Z 轴严格禁止！

        return w_comp, roll, pitch

    # ==================================================================
    # 第2层（次级任务）：小球驻留
    # ==================================================================

    def compute_ball_centering(self):
        """
        计算使小球向平板中心移动所需的附加角速度

        原理：
          小球在倾斜平面上受重力分量作用。
          检测小球与平板中心的偏移，倾斜平板使小球滚回中心。
          如果小球在 +X 方向 → 使平板绕 +Y 轴倾斜（小球向 -X 滚）

        Returns:
            w_ball:  小球驻留角速度 [wx, wy, wz]
            offset:  小球偏移量 [dx, dy, dz] (world frame)
        """
        ball_pos = self.data.xpos[self.ball_id]
        plate_pos = self.data.xpos[self.plate_id]
        offset = ball_pos - plate_pos

        # 限制最大偏移量，防止小球离太远时增益过大
        offset_clipped = np.clip(offset, -0.15, 0.15)

        w_ball = np.zeros(3)
        # 小球 Y 轴偏移 → 绕 X 轴滚转（roll方向）
        w_ball[0] = -self.kp_ball * offset_clipped[1]
        # 小球 X 轴偏移 → 绕 Y 轴俯仰（pitch方向）
        w_ball[1] = self.kp_ball * offset_clipped[0]
        w_ball[2] = 0.0  # Z 轴严格禁止

        return w_ball, offset

    # ==================================================================
    # 第2层：平板速度旋量 → TCP 速度变换
    # ==================================================================

    def plate_twist_to_tcp_velocity(self, v_plate, w_plate, tcp_body_id):
        """
        将平板中心速度旋量变换为 TCP 端期望速度

        刚体速度变换公式：
          v_tcp = v_plate + w_plate × (p_tcp - p_plate)
          w_tcp = w_plate

        Args:
            v_plate:     平板中心线速度 [vx, vy, vz] (world frame)
            w_plate:     平板中心角速度 [wx, wy, wz] (world frame)
            tcp_body_id: TCP 刚体 ID

        Returns:
            twist_tcp:   TCP 速度旋量 [vx, vy, vz, wx, wy, wz]
        """
        tcp_pos = self.data.xpos[tcp_body_id]
        plate_pos = self.data.xpos[self.plate_id]
        r_tcp = tcp_pos - plate_pos  # 平板中心 → TCP 的向量

        v_tcp = v_plate + np.cross(w_plate, r_tcp)
        w_tcp = w_plate.copy()

        return np.concatenate([v_tcp, w_tcp])

    # ==================================================================
    # 第3层：雅可比矩阵计算
    # ==================================================================

    def compute_arm_jacobian(self, tcp_body_id, joint_ids):
        """
        计算单臂 TCP 的几何雅可比矩阵 J ∈ R^(6×7)

        使用 MuJoCo 内置函数 mj_jacBody 计算世界坐标系下的
        平移和旋转雅可比，然后提取指定关节列。

        Args:
            tcp_body_id: TCP 刚体 ID
            joint_ids:   关节 DOF 地址列表

        Returns:
            J: 6×7 雅可比矩阵 (3行平移 + 3行旋转)
        """
        n_joints = len(joint_ids)

        # 分配完整 nv 维雅可比临时数组
        jacp = np.zeros((3, self.model.nv))  # 平移雅可比 (3×nv)
        jacr = np.zeros((3, self.model.nv))  # 旋转雅可比 (3×nv)

        # MuJoCo 计算：世界坐标系下的平移+旋转雅可比
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, tcp_body_id)

        # 提取指定关节列
        J = np.zeros((6, n_joints))
        for i, jid in enumerate(joint_ids):
            dof_addr = self.model.jnt_dofadr[jid]
            J[:3, i] = jacp[:, dof_addr]
            J[3:, i] = jacr[:, dof_addr]

        return J

    # ==================================================================
    # 第3层：阻尼最小二乘（DLS）伪逆
    # ==================================================================

    def solve_ik_dls(self, J, twist_desired, lambda_dls=None):
        """
        阻尼最小二乘伪逆求解关节速度

        dq = J^T * (J * J^T + λ² * I)^(-1) * twist

        当接近奇异点时自动增大阻尼因子。

        Args:
            J:              6×7 雅可比矩阵
            twist_desired:  6维期望速度旋量
            lambda_dls:     阻尼因子 (None则使用默认值)

        Returns:
            dq: 7维关节角速度
        """
        if lambda_dls is None:
            lambda_dls = self.dls_lambda

        # 奇异值检测
        _, S, _ = np.linalg.svd(J)
        cond = S.max() / (S.min() + 1e-10)

        # 条件数过高时增大阻尼
        if cond > 50:
            lambda_dls = lambda_dls * (cond / 50.0)

        # 阻尼最小二乘
        JJT = J @ J.T
        damped = JJT + (lambda_dls ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(damped, twist_desired)

        return dq

    # ==================================================================
    # 第4层：积分 + 限位
    # ==================================================================

    def integrate_and_clamp(self, dq, q_current, joint_ranges):
        """
        关节速度积分 → 目标位置 + 限位钳制

        Args:
            dq:           关节角速度
            q_current:    当前关节位置
            joint_ranges: 关节限位 [[min, max], ...]

        Returns:
            q_target: 目标关节位置（已限位）
        """
        q_target = q_current + dq * self.dt

        # 钳制到关节限位
        for i in range(len(q_target)):
            lo, hi = joint_ranges[i]
            if lo < hi:  # 有关节限位
                q_target[i] = np.clip(q_target[i], lo, hi)

        return q_target

    # ==================================================================
    # 第5层：写入执行器
    # ==================================================================

    def write_arm_actuators(self, q_target_left, q_target_right):
        """
        将目标关节位置写入 position 执行器的 ctrl 数组

        MuJoCo position 执行器内部：
          τ = kp * (ctrl - qpos) + kd * (0 - qvel)

        Args:
            q_target_left:  左臂 7 维目标位置
            q_target_right: 右臂 7 维目标位置
        """
        for i, act_id in enumerate(self.left_act_ids):
            self.data.ctrl[act_id] = q_target_left[i]
        for i, act_id in enumerate(self.right_act_ids):
            self.data.ctrl[act_id] = q_target_right[i]

    def write_static_joints(self):
        """
        将非控制关节（腿、躯干、头）保持在零位，手指保持夹紧
        避免这些关节自由漂移
        """
        # 腿部 8 关节归零
        for act_id in range(ACT_LEGS_START, ACT_TRUNK_START):
            self.data.ctrl[act_id] = 0.0
        # 躯干 4 关节归零
        for act_id in range(ACT_TRUNK_START, ACT_LEFT_ARM_START):
            self.data.ctrl[act_id] = 0.0
        # 手指保持夹紧（不归零，使用设定值）
        self.data.ctrl[ACT_LEFT_FINGER] = self.finger_grip
        self.data.ctrl[ACT_RIGHT_FINGER] = self.finger_grip
        # 头部归零
        for act_id in range(ACT_HEAD_START, self.model.nu):
            self.data.ctrl[act_id] = 0.0

    # ==================================================================
    # 平板 Mocap 更新
    # ==================================================================

    def update_plate_mocap(self):
        """
        更新平板 mocap 驱动体位姿

        平板 mocap 始终保持世界坐标系绝对水平（俯仰=0, 滚转=0），
        仅允许绕 Z 轴（yaw）旋转以保持前进方向。

        姿态补偿由第1层（compute_attitude_compensation）和第3层（雅可比逆解）
        共同完成——通过调整关节角度来抵消底盘倾斜。

        mocap 体通过 weld 约束带动物理平板向水平方向收敛。
        """
        # 获取当前平板朝向的 yaw 角
        plate_xmat = self.data.xmat[self.plate_id].reshape(3, 3).copy()
        yaw = np.arctan2(plate_xmat[1, 0], plate_xmat[0, 0])

        # 目标取向：绝对水平 (roll=0, pitch=0), 保留 yaw
        # roll=pitch=0 → R_target = Rz(yaw)
        cy, sy = np.cos(yaw), np.sin(yaw)
        R_target = np.array([
            [cy, -sy, 0],
            [sy,  cy, 0],
            [0,   0, 1]
        ])

        # 更新 mocap 位置
        self.data.mocap_pos[self.mocap_idx] = self.plate_desired_pos

        # 旋转矩阵 → 四元数 (wxyz)
        trace = R_target[0, 0] + R_target[1, 1] + R_target[2, 2]
        if trace > 0:
            s_val = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s_val
            x = (R_target[2, 1] - R_target[1, 2]) * s_val
            y = (R_target[0, 2] - R_target[2, 0]) * s_val
            z = (R_target[1, 0] - R_target[0, 1]) * s_val
        elif R_target[0, 0] > R_target[1, 1] and R_target[0, 0] > R_target[2, 2]:
            s_val = 2.0 * np.sqrt(1.0 + R_target[0, 0] - R_target[1, 1] - R_target[2, 2])
            w = (R_target[2, 1] - R_target[1, 2]) / s_val
            x = 0.25 * s_val
            y = (R_target[0, 1] + R_target[1, 0]) / s_val
            z = (R_target[0, 2] + R_target[2, 0]) / s_val
        elif R_target[1, 1] > R_target[2, 2]:
            s_val = 2.0 * np.sqrt(1.0 + R_target[1, 1] - R_target[0, 0] - R_target[2, 2])
            w = (R_target[0, 2] - R_target[2, 0]) / s_val
            x = (R_target[0, 1] + R_target[1, 0]) / s_val
            y = 0.25 * s_val
            z = (R_target[1, 2] + R_target[2, 1]) / s_val
        else:
            s_val = 2.0 * np.sqrt(1.0 + R_target[2, 2] - R_target[0, 0] - R_target[1, 1])
            w = (R_target[1, 0] - R_target[0, 1]) / s_val
            x = (R_target[0, 2] + R_target[2, 0]) / s_val
            y = (R_target[1, 2] + R_target[2, 1]) / s_val
            z = 0.25 * s_val

        self.data.mocap_quat[self.mocap_idx] = np.array([w, x, y, z])

    # ==================================================================
    # 主控制循环（单步）
    # ==================================================================

    def control_step(self):
        """每步仿真调用一次，执行完整的 5 层控制链路"""

        # === 第1层：姿态补偿 ===
        w_comp, roll, pitch = self.compute_attitude_compensation()

        # === 次级任务：小球驻留 ===
        w_ball = np.zeros(3)
        ball_offset = np.zeros(3)
        if self.enable_ball:
            w_ball, ball_offset = self.compute_ball_centering()

        # === 第0层：平板期望速度旋量（世界坐标系） ===
        # 线速度：平板在全局坐标系中可缓慢前进
        # v_plate_des = np.array([0.02 * np.sin(0.3 * self.data.time), 0.0, 0.0])
        v_plate_des = np.array([0.0, 0.0, 0.0])  # 静止模式

        # 角速度 = 姿态补偿（硬约束）+ 小球驻留（软约束）
        w_plate_des = w_comp + 0.15 * w_ball  # 小球驻留仅占 15%

        # 强制：Z 轴禁止旋转
        w_plate_des[2] = 0.0

        # === 第2层：平板速度 → TCP 速度变换 ===
        twist_left  = self.plate_twist_to_tcp_velocity(
            v_plate_des, w_plate_des, self.tcp_left_id)
        twist_right = self.plate_twist_to_tcp_velocity(
            v_plate_des, w_plate_des, self.tcp_right_id)

        # === 第3层：雅可比 + DLS 伪逆 ===
        J_left  = self.compute_arm_jacobian(self.tcp_left_id,  self.left_joint_ids)
        J_right = self.compute_arm_jacobian(self.tcp_right_id, self.right_joint_ids)

        dq_left  = self.solve_ik_dls(J_left,  twist_left)
        dq_right = self.solve_ik_dls(J_right, twist_right)

        # 速度平滑（一阶低通滤波）
        alpha = 0.7
        dq_left  = alpha * dq_left  + (1 - alpha) * self.prev_dq_left
        dq_right = alpha * dq_right + (1 - alpha) * self.prev_dq_right
        self.prev_dq_left  = dq_left
        self.prev_dq_right = dq_right

        # === 第4层：积分 + 限位 ===
        q_cur_left  = self.get_current_joint_positions('left')
        q_cur_right = self.get_current_joint_positions('right')

        q_target_left  = self.integrate_and_clamp(dq_left,  q_cur_left,  self.left_joint_ranges)
        q_target_right = self.integrate_and_clamp(dq_right, q_cur_right, self.right_joint_ranges)

        # === 第5层：写入执行器 ===
        self.write_arm_actuators(q_target_left, q_target_right)
        self.write_static_joints()

        # === 更新平板 mocap 位姿 ===
        self.update_plate_mocap()

        # === 日志输出 ===
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            plate_z = self.data.xmat[self.plate_id].reshape(3, 3)[:, 2]
            level_err = np.degrees(np.arccos(np.clip(plate_z[2], -1, 1)))
            print(f"[t={self.data.time:5.2f}s] "
                  f"roll={np.degrees(roll):+5.1f}° "
                  f"pitch={np.degrees(pitch):+5.1f}° "
                  f"| level_err={level_err:5.2f}° "
                  f"ball_offset=({ball_offset[0]:+5.3f}, {ball_offset[1]:+5.3f})")

        return dq_left, dq_right, roll, pitch

# =====================================================================
# 仿真运行器
# =====================================================================

class SimulationRunner:
    """MuJoCo 仿真运行封装"""

    def __init__(self, xml_path, args):
        self.xml_path = xml_path
        self.args = args
        self.model = None
        self.data = None
        self.controller = None

    def load(self):
        """加载 MJCF 模型并设置初始状态"""
        print(f"[Load] 加载场景文件: {self.xml_path}")
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"MJCF 文件不存在: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # ---- 在 Python 侧设置初始 qpos ----
        self.data.qpos[:] = 0.0

        def set_free_body_qpos(body_name, pos, quat_wxyz=(1, 0, 0, 0)):
            """设置 free body 的初始位姿"""
            body_id = mujoco.mj_name2id(self.model,
                                        mujoco.mjtObj.mjOBJ_BODY, body_name)
            jnt_id = self.model.body_jntadr[body_id]
            start = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[start:start + 3] = pos
            self.data.qpos[start + 3:start + 7] = quat_wxyz

        # 平板绕 Z 旋转 90°：长边沿 Y 正对机器人，短边在左右
        qz90 = 0.707107
        plate_half_long = 0.21   # 与 XML size 一致（旋转后沿 Y）
        finger_reach = 0.225     # Joint7 → 手指中点 (local -Y)
        edge_clearance = 0.008   # 手指中点在短边外侧，避免穿模

        # 水平夹持：手指开合沿世界 Z，接近方向沿 ±Y（无俯仰）
        # 左臂: X→+Z, Y→+Y, Z→-X
        R_left = np.array([[0.0, 0.0, -1.0],
                           [0.0, 1.0,  0.0],
                           [1.0, 0.0,  0.0]])

        tcp_left_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Joint7_L")
        tcp_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Joint7_R")
        left_joint_ids  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                            f"left_arm_joint_{i}") for i in range(1, 8)]
        right_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                            f"right_arm_joint_{i}") for i in range(1, 8)]

        # 已标定近水平种子（倾角约 0.4°）
        q_left_seed = np.array([0.9691, -0.72, 0.144, -0.88, 1.5063, -0.4957, 1.5708])

        plate_nominal = np.array([0.35, 0.0, 0.98])
        finger_left = plate_nominal + np.array(
            [0.0, +(plate_half_long + edge_clearance), 0.0])
        j7_left_des = finger_left - R_left @ np.array([0.0, -finger_reach, 0.0])

        print("[Load] 求解左臂 6D IK (水平短边外侧夹持)...")
        q_left_init = self._ik_solve_arm_6d(
            tcp_left_id, left_joint_ids, j7_left_des, R_left,
            q0=q_left_seed, w_pos=0.35, w_rot=1.4,
            tol_pos=0.012, tol_rot=0.10, n_iter=2000)
        print(f"[Load]  左臂关节角: {np.round(q_left_init, 3)}")

        # 右臂严格镜像，保证对称与等高
        q_right_init = np.array([
            -q_left_init[0],  q_left_init[1], -q_left_init[2],
             q_left_init[3], -q_left_init[4],  q_left_init[5],
            -q_left_init[6],
        ])
        for k, jid in enumerate(right_joint_ids):
            lo, hi = self.model.jnt_range[jid]
            q = q_right_init[k]
            if lo < hi:
                q = np.clip(q, lo, hi)
            self.data.qpos[self.model.jnt_qposadr[jid]] = q
            q_right_init[k] = q
        mujoco.mj_forward(self.model, self.data)
        print(f"[Load]  右臂镜像角: {np.round(q_right_init, 3)}")

        def finger_midpoint(side):
            names = (('leftfinger1_Link', 'leftfinger2_Link') if side == 'left'
                     else ('rightfinger1_Link', 'rightfinger2_Link'))
            f1 = self.data.xpos[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, names[0])]
            f2 = self.data.xpos[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, names[1])]
            return 0.5 * (f1 + f2)

        mid_l = finger_midpoint('left')
        mid_r = finger_midpoint('right')
        tcp_left_pos  = self.data.xpos[tcp_left_id].copy()
        tcp_right_pos = self.data.xpos[tcp_right_id].copy()
        R_l = self.data.xmat[tcp_left_id].reshape(3, 3)
        tilt_deg = float(np.degrees(np.arcsin(np.clip(R_l[2, 1], -1, 1))))

        plate_center_x = 0.5 * (mid_l[0] + mid_r[0])
        plate_center_y = 0.0
        plate_surface_z = 0.5 * (mid_l[2] + mid_r[2])
        finger_span_y = 0.5 * (abs(mid_l[1]) + abs(mid_r[1]))

        print(f"\n[Load] IK 结果:")
        print(f"  左 Joint7: ({tcp_left_pos[0]:.3f}, {tcp_left_pos[1]:.3f}, {tcp_left_pos[2]:.3f})")
        print(f"  右 Joint7: ({tcp_right_pos[0]:.3f}, {tcp_right_pos[1]:.3f}, {tcp_right_pos[2]:.3f})")
        print(f"  左手指中点: ({mid_l[0]:.3f}, {mid_l[1]:.3f}, {mid_l[2]:.3f})")
        print(f"  右手指中点: ({mid_r[0]:.3f}, {mid_r[1]:.3f}, {mid_r[2]:.3f})")
        print(f"  TCP高度差: {abs(tcp_left_pos[2]-tcp_right_pos[2])*1000:.1f} mm")
        print(f"  夹爪倾角: {tilt_deg:+.2f}° (0=水平)")
        print(f"  接近方向: {np.round(mid_l - tcp_left_pos, 3)}")
        print(f"  平板中心: ({plate_center_x:.3f}, {plate_center_y:.3f}, {plate_surface_z:.3f})")
        print(f"  手指|Y|={finger_span_y:.3f}, 板半长={plate_half_long:.3f}, "
              f"外侧余量={finger_span_y - plate_half_long:.3f} m")

        set_free_body_qpos("plate", (plate_center_x, plate_center_y, plate_surface_z),
                           (qz90, 0, 0, qz90))
        set_free_body_qpos("cup", (plate_center_x + 0.04, 0.06, plate_surface_z + 0.06))
        set_free_body_qpos("water_mass", (plate_center_x + 0.04, 0.06, plate_surface_z + 0.085))
        set_free_body_qpos("ball", (plate_center_x - 0.04, -0.06, plate_surface_z + 0.03))

        for i, act_id in enumerate(range(12, 19)):
            self.data.ctrl[act_id] = q_left_init[i]
        for i, act_id in enumerate(range(20, 27)):
            self.data.ctrl[act_id] = q_right_init[i]
        self.data.ctrl[ACT_LEFT_FINGER] = 0.010
        self.data.ctrl[ACT_RIGHT_FINGER] = 0.010

        mujoco.mj_forward(self.model, self.data)

        print(f"\n[Load] 模型加载成功")
        print(f"  nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
        p_pos = self.data.xpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'plate')]
        print(f"  平板初始位置: ({p_pos[0]:.3f}, {p_pos[1]:.3f}, {p_pos[2]:.3f})")
        print(f"  夹持模式: 左右短边水平外侧夹持（防穿模）")

        self.controller = DualArmPlateController(self.model, self.data, self.args)
        self.controller.plate_desired_pos = np.array(
            [plate_center_x, plate_center_y, plate_surface_z])
        self.controller.finger_grip = 0.010

    def step(self):
        """执行单步仿真"""
        self.controller.control_step()
        mujoco.mj_step(self.model, self.data)

    def run_headless(self):
        """无 GUI 模式运行"""
        print(f"[Run] 无GUI模式，仿真时长 {self.args.duration}s")
        total_steps = int(self.args.duration / self.model.opt.timestep)

        t_start = time.time()
        for step in range(total_steps):
            self.step()
            if step % 500 == 0:
                elapsed = time.time() - t_start
                progress = 100.0 * step / total_steps
                print(f"[Progress] {progress:5.1f}%  elapsed={elapsed:.1f}s")

        elapsed = time.time() - t_start
        print(f"[Done] 仿真完成，实际用时 {elapsed:.1f}s，"
              f"RTF = {self.args.duration / elapsed:.2f}")

    def run_gui(self):
        """带 GUI 可视化模式运行"""
        print(f"[Run] GUI可视化模式，按 ESC 退出")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 设置相机视角
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = [0.3, 0.0, 0.7]

            while viewer.is_running():
                step_start = time.time()
                self.step()
                viewer.sync()

                # 帧率控制
                elapsed = time.time() - step_start
                if elapsed < self.model.opt.timestep:
                    time.sleep(self.model.opt.timestep - elapsed)

    def run_step_mode(self):
        """逐步调试模式（按 Enter 前进一帧）"""
        print(f"[Run] 逐步调试模式")
        print(f"  Enter → 前进一步 | q → 退出 | i → 打印状态")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = [0.3, 0.0, 0.7]

            viewer.sync()
            step_idx = 0

            while viewer.is_running():
                try:
                    cmd = input(f"[Step {step_idx}] 命令: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break

                if cmd == 'q' or cmd == 'quit':
                    break
                elif cmd == 'i' or cmd == 'info':
                    self._print_state()
                    viewer.sync()
                    continue
                elif cmd == '':
                    pass  # 按 Enter 就前进
                else:
                    print(f"  未知命令: {cmd}")
                    viewer.sync()
                    continue

                self.step()
                step_idx += 1
                viewer.sync()

    def _print_state(self):
        """打印当前仿真状态"""
        ctrl = self.controller
        data = self.data

        roll, pitch = ctrl.get_chassis_orientation()
        plate_pos = data.xpos[ctrl.plate_id]
        ball_pos = data.xpos[ctrl.ball_id]
        offset = ball_pos - plate_pos

        plate_z = data.xmat[ctrl.plate_id].reshape(3, 3)[:, 2]
        level_err = np.degrees(np.arccos(np.clip(plate_z[2], -1, 1)))

        print(f"\n  === 仿真状态 (t={data.time:.3f}s) ===")
        print(f"  底盘: roll={np.degrees(roll):+.2f}°  pitch={np.degrees(pitch):+.2f}°")
        print(f"  平板: pos=({plate_pos[0]:.3f},{plate_pos[1]:.3f},{plate_pos[2]:.3f})")
        print(f"        水平误差={level_err:.3f}°")
        print(f"  小球: offset=({offset[0]:+.4f},{offset[1]:+.4f},{offset[2]:+.4f})")
        print(f"  左臂 qpos: {ctrl.get_current_joint_positions('left')}")
        print(f"  右臂 qpos: {ctrl.get_current_joint_positions('right')}")
        print()

    def _ik_solve_arm_6d(self, tcp_body_id, joint_ids, target_pos, target_R,
                          q0=None, n_iter=1500, w_pos=1.0, w_rot=0.35,
                          tol_pos=0.008, tol_rot=0.08):
        """
        6D 数值 IK：位置 + 姿态（DLS）

        手指在 Joint7 local -Y 方向伸出约 0.225m，开合主要沿 local X。
        侧向夹持时令 local X≈世界 Z，local Y 指向板外侧。
        """
        def set_q(q):
            for k, jid in enumerate(joint_ids):
                self.data.qpos[self.model.jnt_qposadr[jid]] = q[k]

        def get_q():
            return np.array([self.data.qpos[self.model.jnt_qposadr[j]]
                             for j in joint_ids])

        def rot_error(R_cur, R_des):
            R_err = R_des @ R_cur.T
            return 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])

        if q0 is not None:
            set_q(q0)
        else:
            set_q(np.zeros(7))

        best_cost = float('inf')
        best_q = get_q()

        for i in range(n_iter):
            mujoco.mj_forward(self.model, self.data)
            p = self.data.xpos[tcp_body_id].copy()
            R = self.data.xmat[tcp_body_id].reshape(3, 3).copy()

            e_pos = target_pos - p
            e_rot = rot_error(R, target_R)
            pos_n = np.linalg.norm(e_pos)
            rot_n = np.linalg.norm(e_rot)
            cost = pos_n + 0.25 * rot_n

            if cost < best_cost:
                best_cost = cost
                best_q = get_q()

            if pos_n < tol_pos and rot_n < tol_rot:
                print(f"  [IK6] ✓ 第{i}次: pos={pos_n*1000:.1f}mm, rot={rot_n:.3f}rad")
                return get_q()

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, tcp_body_id)

            J = np.zeros((6, len(joint_ids)))
            for k, jid in enumerate(joint_ids):
                dof = self.model.jnt_dofadr[jid]
                J[:3, k] = jacp[:, dof]
                J[3:, k] = jacr[:, dof]

            e = np.concatenate([w_pos * e_pos, w_rot * e_rot])
            Jw = J.copy()
            Jw[:3] *= w_pos
            Jw[3:] *= w_rot
            lam = 0.08 * max(1.0, pos_n / 0.05)
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam**2 * np.eye(6), e)
            dq = np.clip(dq, -0.25, 0.25)

            q = get_q() + dq
            for k, jid in enumerate(joint_ids):
                lo, hi = self.model.jnt_range[jid]
                if lo < hi:
                    q[k] = np.clip(q[k], lo, hi)
            set_q(q)
        else:
            set_q(best_q)
            mujoco.mj_forward(self.model, self.data)
            p = self.data.xpos[tcp_body_id]
            R = self.data.xmat[tcp_body_id].reshape(3, 3)
            pos_n = np.linalg.norm(target_pos - p)
            rot_n = np.linalg.norm(rot_error(R, target_R))
            print(f"  [IK6] ⚠ 未完全收敛: pos={pos_n*1000:.1f}mm, rot={rot_n:.3f}rad "
                  f"(已取最优)")

        return get_q()

    def run(self):
        """根据参数选择运行模式"""
        self.load()

        if self.args.step:
            self.run_step_mode()
        elif self.args.gui:
            self.run_gui()
        else:
            self.run_headless()

# =====================================================================
# 入口
# =====================================================================

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(
        description="轮式移动双臂机器人协同稳载仿真系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dual_arm_controller.py                          # 无GUI模式运行30秒
  python dual_arm_controller.py --gui                    # GUI可视化模式
  python dual_arm_controller.py --gui --step             # 逐步调试模式
  python dual_arm_controller.py --gui --no-ball          # 仅验证姿态补偿
  python dual_arm_controller.py --kp-attitude 10.0       # 高增益快速收敛
  python dual_arm_controller.py --gui --duration 60      # 长时间运行
        """)

    parser.add_argument('--gui', action='store_true',
                        help='启用 MuJoCo 可视化渲染')
    parser.add_argument('--step', action='store_true',
                        help='逐步仿真模式（按 Enter 前进）')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='仿真总时长（秒，无GUI模式）')
    parser.add_argument('--kp-attitude', type=float, default=5.0,
                        help='姿态补偿比例增益')
    parser.add_argument('--kp-ball', type=float, default=2.0,
                        help='小球驻留比例增益')
    parser.add_argument('--no-ball', action='store_true',
                        help='禁用小球驻留任务')
    parser.add_argument('--xml', type=str,
                        default='scene_dual_arm_plate.xml',
                        help='MJCF 场景文件路径')

    return parser.parse_args()


def main():
    args = parse_args()

    # 查找 MJCF 文件
    xml_path = args.xml
    if not os.path.isabs(xml_path):
        # 尝试多个可能的位置
        candidates = [
            os.path.join(os.path.dirname(__file__), xml_path),
            xml_path,
        ]
        for cand in candidates:
            if os.path.exists(cand):
                xml_path = os.path.abspath(cand)
                break
        else:
            print(f"[Error] 找不到 MJCF 文件: {args.xml}")
            print(f"  搜索路径: {candidates}")
            sys.exit(1)

    print("=" * 60)
    print("  轮式移动双臂机器人协同稳载仿真系统")
    print("  BalanceDual-Arm v1.0")
    print("=" * 60)
    print(f"  MJCF:    {xml_path}")
    print(f"  GUI:     {args.gui}")
    print(f"  Step:    {args.step}")
    print(f"  Kp_att:  {args.kp_attitude}")
    print(f"  Kp_ball: {args.kp_ball}")
    print(f"  Ball:    {'启用' if not args.no_ball else '禁用'}")
    print("=" * 60)

    runner = SimulationRunner(xml_path, args)
    runner.run()


if __name__ == '__main__':
    main()
