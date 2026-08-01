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
  第5层：重力补偿 + PD 力矩
         τ = g(q) + Kp·(q_des−q) + Kd·(dq_des−dq)  → data.ctrl

仿真规则：
  - 全部关节使用 motor 力矩型执行器
  - 禁止直接写 qpos 赋值控制机器人（仅初始化允许）
  - 禁止纯速度开环控制
  - 重力 g(q)：mj_forward(q, dq=0) 后读 qfrc_bias（同 anyverse WBC 验证）
  - 默认 --hold：锁定初始关节角保持，先验证动力学稳定

依赖：
  pip install mujoco numpy scipy

运行：
  python dual_arm_controller.py --duration 20   # hold 初始姿态 20s（默认）
  python dual_arm_controller.py --gui          # 带GUI可视化
  python dual_arm_controller.py --balance      # 启用稳水平任务

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
        self.hold_mode = not getattr(args, 'balance', False)  # 默认 hold
        self.finger_grip = 0.010                # 手指夹紧位置 (m)

        # 力矩 PD 增益（参考 anyverse gravity_pd）
        # τ = g(q) + Kp·e + Kd·ė
        self.kp_arm = np.array([80.0, 80.0, 60.0, 60.0, 30.0, 30.0, 25.0])
        self.kd_arm = np.array([10.0, 10.0,  8.0,  8.0,  3.0,  3.0,  2.5])
        self.ki_arm = np.zeros(7)
        self.i_err_left = np.zeros(7)
        self.i_err_right = np.zeros(7)
        self.i_err_limit = 0.2
        self.kp_leg = 150.0
        self.kd_leg = 20.0
        self.kp_trunk = 200.0
        self.kd_trunk = 25.0
        self.kp_head = 40.0
        self.kd_head = 4.0
        self.kp_finger = 50.0
        self.kd_finger = 3.0

        # ---- 运动学 ID 映射 ----
        self._init_kinematics()

        # ---- 状态变量 ----
        self.step_count = 0
        self.plate_desired_pos = np.array([0.35, 0.0, 0.98])  # 平板期望位置
        self.prev_dq_left = np.zeros(7)   # 上一帧左臂关节速度
        self.prev_dq_right = np.zeros(7)  # 上一帧右臂关节速度
        # 全部执行器保持目标（初始化后锁定）
        self.hold_q_targets = {}
        self.q_des_left = np.zeros(7)
        self.q_des_right = np.zeros(7)
        self.q_init_left = np.zeros(7)
        self.q_init_right = np.zeros(7)

        # 场景浮体初始位姿（HOLD 时钉住可视化）
        self.scene_free_qpos = {}  # name -> (qadr, qpos7)
        self.plate_hold_yaw = np.pi / 2  # 长边正对机器人（Rz90）

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

        # 执行器 → 关节 DOF 地址（motor gear=1 时 ctrl 对应关节力矩）
        self.act_dofadr = np.zeros(model.nu, dtype=int)
        self.act_jnt_ids = np.zeros(model.nu, dtype=int)
        for a in range(model.nu):
            # 传动链第一个关节
            jnt_id = model.actuator_trnid[a, 0]
            self.act_jnt_ids[a] = jnt_id
            self.act_dofadr[a] = model.jnt_dofadr[jnt_id]

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

        print(f"[Init] 左臂关节 ID: {self.left_joint_ids}")
        print(f"[Init] 右臂关节 ID: {self.right_joint_ids}")
        print(f"[Init] 左臂执行器 ID: {self.left_act_ids}")
        print(f"[Init] 右臂执行器 ID: {self.right_act_ids}")
        print(f"[Init] 执行器模式: motor 力矩控制 (τ = g(q) + PD)")

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

    def get_current_joint_velocities(self, side='left'):
        """读取当前关节速度 qvel"""
        ids = self.left_joint_ids if side == 'left' else self.right_joint_ids
        return np.array([self.data.qvel[self.model.jnt_dofadr[j]] for j in ids])

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
    # 第5层：重力补偿 + PD → data.ctrl
    # 参考 anyverse mujoco_wbc_validation / gravity_pd_controller：
    #   τ = g(q) + Kp·(q_des−q) + Kd·(dq_des−dq)
    #   g(q) = qfrc_bias |_{dq=0}  （mj_forward 后读取）
    # ==================================================================

    def compute_gravity_torques(self):
        """
        用 MuJoCo 求纯重力力矩 g(q)。

        临时将 qvel 置零后 mj_forward，此时 qfrc_bias = g(q)；
        再恢复原状态并重新 forward，避免污染仿真。
        """
        model, data = self.model, self.data
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()

        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        g = data.qfrc_bias.copy()

        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_forward(model, data)
        return g

    def _hold_kp_kd(self, act_id):
        """按执行器分组返回 (Kp, Kd)"""
        if ACT_LEGS_START <= act_id < ACT_TRUNK_START:
            return self.kp_leg, self.kd_leg
        if ACT_TRUNK_START <= act_id < ACT_LEFT_ARM_START:
            return self.kp_trunk, self.kd_trunk
        if act_id >= ACT_HEAD_START:
            return self.kp_head, self.kd_head
        if act_id in (ACT_LEFT_FINGER, ACT_RIGHT_FINGER):
            return self.kp_finger, self.kd_finger
        return self.kp_leg, self.kd_leg

    def write_torque_ctrl(self, q_des_left, dq_des_left, q_des_right, dq_des_right):
        """
        计算力矩并写入 motor 执行器 ctrl：

          τ = g(q) + Kp·e + Kd·ė
          data.ctrl = clip(τ)
        """
        model, data = self.model, self.data

        q_cur_l = self.get_current_joint_positions('left')
        q_cur_r = self.get_current_joint_positions('right')
        dq_cur_l = self.get_current_joint_velocities('left')
        dq_cur_r = self.get_current_joint_velocities('right')

        g = self.compute_gravity_torques()

        def set_ctrl(act_id, tau):
            lo, hi = model.actuator_ctrlrange[act_id]
            data.ctrl[act_id] = float(np.clip(tau, lo, hi))

        # 积分项：消除接触/weld 外力引起的稳态偏差（仅 hold 或始终启用）
        self.i_err_left += (q_des_left - q_cur_l) * self.dt
        self.i_err_right += (q_des_right - q_cur_r) * self.dt
        self.i_err_left = np.clip(self.i_err_left, -self.i_err_limit, self.i_err_limit)
        self.i_err_right = np.clip(self.i_err_right, -self.i_err_limit, self.i_err_limit)

        for i, act_id in enumerate(self.left_act_ids):
            dof = int(self.act_dofadr[act_id])
            e = q_des_left[i] - q_cur_l[i]
            de = dq_des_left[i] - dq_cur_l[i]
            tau = (g[dof] + self.kp_arm[i] * e + self.kd_arm[i] * de
                   + self.ki_arm[i] * self.i_err_left[i])
            set_ctrl(act_id, tau)

        for i, act_id in enumerate(self.right_act_ids):
            dof = int(self.act_dofadr[act_id])
            e = q_des_right[i] - q_cur_r[i]
            de = dq_des_right[i] - dq_cur_r[i]
            tau = (g[dof] + self.kp_arm[i] * e + self.kd_arm[i] * de
                   + self.ki_arm[i] * self.i_err_right[i])
            set_ctrl(act_id, tau)

        for act_id in list(range(ACT_LEGS_START, ACT_LEFT_ARM_START)) + \
                      list(range(ACT_HEAD_START, model.nu)):
            dof = int(self.act_dofadr[act_id])
            jid = int(self.act_jnt_ids[act_id])
            qadr = model.jnt_qposadr[jid]
            q_des = self.hold_q_targets.get(act_id, 0.0)
            e = q_des - data.qpos[qadr]
            de = 0.0 - data.qvel[dof]
            kp, kd = self._hold_kp_kd(act_id)
            set_ctrl(act_id, g[dof] + kp * e + kd * de)

        for act_id in (ACT_LEFT_FINGER, ACT_RIGHT_FINGER):
            dof = int(self.act_dofadr[act_id])
            jid = int(self.act_jnt_ids[act_id])
            qadr = model.jnt_qposadr[jid]
            e = self.finger_grip - data.qpos[qadr]
            de = 0.0 - data.qvel[dof]
            set_ctrl(act_id, g[dof] + self.kp_finger * e + self.kd_finger * de)

    def lock_hold_targets(self):
        """初始化完成后，锁定全部关节当前角度作为保持目标"""
        for act_id in list(range(ACT_LEGS_START, ACT_LEFT_ARM_START)) + \
                      list(range(ACT_HEAD_START, self.model.nu)):
            jid = int(self.act_jnt_ids[act_id])
            qadr = self.model.jnt_qposadr[jid]
            self.hold_q_targets[act_id] = float(self.data.qpos[qadr])

        self.q_des_left = self.get_current_joint_positions('left').copy()
        self.q_des_right = self.get_current_joint_positions('right').copy()
        self.q_init_left = self.q_des_left.copy()
        self.q_init_right = self.q_des_right.copy()
        print(f"[Hold] 锁定初始姿态 "
              f"左臂={np.round(self.q_des_left, 3)}  "
              f"右臂={np.round(self.q_des_right, 3)}")

        if self.hold_mode:
            self._prepare_hold_scene()

    def _prepare_hold_scene(self):
        """
        HOLD 场景准备：
          - 关闭接触（初始姿态网格自穿透会产生极大接触力）
          - 保留平板 mocap-weld，保持 Rz90° 可视化
          - 杯/水/球钉在初始位姿（无接触时否则会掉落）
        """
        model, data = self.model, self.data

        # 仅关接触，保留 equality（平板 weld + 手指镜像）
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        # 记录并钉住场景浮体当前位姿（load 已摆好 Rz90 板 + 杯/球）
        self.scene_free_qpos = {}
        for name in ("plate", "cup", "water_mass", "ball"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            jid = model.body_jntadr[bid]
            qadr = int(model.jnt_qposadr[jid])
            self.scene_free_qpos[name] = (qadr, data.qpos[qadr:qadr + 7].copy())

        # mocap 对齐到期望位姿，yaw 固定 90°
        self.plate_hold_yaw = np.pi / 2
        self._set_plate_mocap_yaw(self.plate_hold_yaw)
        # 同步物理板姿态到 mocap（weld 收敛前先对齐，避免“转回去”）
        if "plate" in self.scene_free_qpos:
            qadr, q7 = self.scene_free_qpos["plate"]
            q7 = q7.copy()
            q7[:3] = self.plate_desired_pos
            s2 = np.sin(self.plate_hold_yaw / 2.0)
            c2 = np.cos(self.plate_hold_yaw / 2.0)
            q7[3:] = [c2, 0.0, 0.0, s2]  # wxyz, Rz(yaw)
            data.qpos[qadr:qadr + 7] = q7
            self.scene_free_qpos["plate"] = (qadr, q7.copy())

        self.finger_grip = 0.010
        for fname, val in (('leftfinger1_joint', 0.010), ('leftfinger2_joint', -0.010),
                           ('rightfinger1_joint', 0.010), ('rightfinger2_joint', -0.010)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = val

        mujoco.mj_forward(model, data)
        print("[Hold] 已禁用接触；板/杯/球保留在初始位姿（板 Rz90°）")

    def _set_plate_mocap_yaw(self, yaw):
        """设置平板 mocap 位姿：水平 + 指定 yaw"""
        cy, sy = np.cos(yaw), np.sin(yaw)
        # Rz(yaw) → quat wxyz
        self.data.mocap_pos[self.mocap_idx] = self.plate_desired_pos
        self.data.mocap_quat[self.mocap_idx] = np.array([
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
        ])

    def _pin_scene_objects(self):
        """每步钉住杯/水/球（无接触时防坠落）；板由 mocap weld 驱动"""
        data = self.data
        for name, (qadr, q7) in self.scene_free_qpos.items():
            if name == "plate":
                continue  # 板跟 mocap
            data.qpos[qadr:qadr + 7] = q7
            # 清速度
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            dadr = self.model.jnt_dofadr[self.model.body_jntadr[bid]]
            data.qvel[dadr:dadr + 6] = 0.0

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

        # === Hold 模式：锁定初始关节角，仅重力补偿 + PD ===
        if self.hold_mode:
            self._set_plate_mocap_yaw(self.plate_hold_yaw)
            self._pin_scene_objects()
            self.write_torque_ctrl(
                self.q_des_left, np.zeros(7),
                self.q_des_right, np.zeros(7))

            self.step_count += 1
            if self.step_count % self.log_interval == 0:
                roll, pitch = self.get_chassis_orientation()
                q_l = self.get_current_joint_positions('left')
                q_r = self.get_current_joint_positions('right')
                err_l = np.linalg.norm(q_l - self.q_init_left)
                err_r = np.linalg.norm(q_r - self.q_init_right)
                print(f"[HOLD t={self.data.time:5.2f}s] "
                      f"roll={np.degrees(roll):+5.2f}° "
                      f"pitch={np.degrees(pitch):+5.2f}° "
                      f"| ‖Δq_L‖={err_l:.4f} ‖Δq_R‖={err_r:.4f}")
            return

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

        # === 第5层：重力补偿 + PD 力矩 → data.ctrl ===
        self.write_torque_ctrl(q_target_left, dq_left, q_target_right, dq_right)

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

        # 手指初始开合写到 qpos（motor 控制力矩，初始化用状态）
        for fname, val in (('leftfinger1_joint', 0.010), ('leftfinger2_joint', -0.010),
                           ('rightfinger1_joint', 0.010), ('rightfinger2_joint', -0.010)):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, fname)
            self.data.qpos[self.model.jnt_qposadr[jid]] = val

        # motor 执行器：ctrl 为力矩，初始化为 0，首帧 control_step 再写入
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        print(f"\n[Load] 模型加载成功")
        print(f"  nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}")
        p_pos = self.data.xpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'plate')]
        print(f"  平板初始位置: ({p_pos[0]:.3f}, {p_pos[1]:.3f}, {p_pos[2]:.3f})")
        print(f"  夹持模式: 左右短边水平外侧夹持（防穿模）")
        print(f"  执行器: motor 力矩控制 (τ = g(q) + PD)")
        mode = "HOLD 初始姿态" if not self.args.balance else "BALANCE 稳水平"
        print(f"  控制模式: {mode}")

        self.controller = DualArmPlateController(self.model, self.data, self.args)
        self.controller.plate_desired_pos = np.array(
            [plate_center_x, plate_center_y, plate_surface_z])
        self.controller.finger_grip = 0.010
        self.controller.lock_hold_targets()
        # 首帧写入平衡力矩，避免 mj_step 前零力矩导致塌陷
        self.controller.write_torque_ctrl(
            q_left_init, np.zeros(7), q_right_init, np.zeros(7))

    def step(self):
        """执行单步仿真"""
        self.controller.control_step()
        mujoco.mj_step(self.model, self.data)

    def run_headless(self):
        """无 GUI 模式运行"""
        mode = "HOLD" if self.controller.hold_mode else "BALANCE"
        print(f"[Run] 无GUI模式 [{mode}]，仿真时长 {self.args.duration}s")
        total_steps = int(self.args.duration / self.model.opt.timestep)

        # 稳定性统计
        max_err_l = 0.0
        max_err_r = 0.0
        max_pitch = 0.0

        t_start = time.time()
        for step in range(total_steps):
            self.step()
            if step % 100 == 0:
                q_l = self.controller.get_current_joint_positions('left')
                q_r = self.controller.get_current_joint_positions('right')
                err_l = float(np.linalg.norm(q_l - self.controller.q_init_left))
                err_r = float(np.linalg.norm(q_r - self.controller.q_init_right))
                _, pitch = self.controller.get_chassis_orientation()
                max_err_l = max(max_err_l, err_l)
                max_err_r = max(max_err_r, err_r)
                max_pitch = max(max_pitch, abs(float(pitch)))

            if step % 500 == 0:
                elapsed = time.time() - t_start
                progress = 100.0 * step / total_steps
                print(f"[Progress] {progress:5.1f}%  elapsed={elapsed:.1f}s  "
                      f"‖Δq_L‖={err_l:.4f} ‖Δq_R‖={err_r:.4f} "
                      f"|pitch|={np.degrees(abs(pitch)):.2f}°")

        elapsed = time.time() - t_start
        print(f"[Done] 仿真完成，实际用时 {elapsed:.1f}s，"
              f"RTF = {self.args.duration / elapsed:.2f}")
        print(f"[Stability] max‖Δq_L‖={max_err_l:.4f}  max‖Δq_R‖={max_err_r:.4f}  "
              f"max|pitch|={np.degrees(max_pitch):.2f}°")

        # 终态
        self._print_state()
        ok = (max_err_l < 0.05 and max_err_r < 0.05 and max_pitch < np.radians(3.0))
        print(f"[Verdict] {'PASS 初始姿态保持稳定' if ok else 'FAIL 漂移过大，需调增益'}")
        return ok

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
  python dual_arm_controller.py --duration 20            # HOLD 初始姿态 20s（默认）
  python dual_arm_controller.py --gui                    # GUI 可视化 HOLD
  python dual_arm_controller.py --balance --gui          # 启用稳水平任务
  python dual_arm_controller.py --gui --step             # 逐步调试
  python dual_arm_controller.py --balance --no-ball      # 仅姿态补偿
        """)

    parser.add_argument('--gui', action='store_true',
                        help='启用 MuJoCo 可视化渲染')
    parser.add_argument('--step', action='store_true',
                        help='逐步仿真模式（按 Enter 前进）')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='仿真总时长（秒，无GUI模式，默认20）')
    parser.add_argument('--balance', action='store_true',
                        help='启用稳水平/小球任务（默认关闭，仅 HOLD 初始姿态）')
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
    print(f"  Mode:    {'BALANCE' if args.balance else 'HOLD'}")
    print(f"  Duration:{args.duration}s")
    print(f"  Kp_att:  {args.kp_attitude}")
    print(f"  Kp_ball: {args.kp_ball}")
    print(f"  Ball:    {'启用' if not args.no_ball else '禁用'}")
    print("=" * 60)

    runner = SimulationRunner(xml_path, args)
    runner.run()


if __name__ == '__main__':
    main()
