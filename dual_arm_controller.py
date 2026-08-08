#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轮式移动双臂机器人协同稳载控制系统
===============================================
项目：BalanceDual-Arm

模式：
  HOLD  — 锁定初始关节角，重力补偿 + PD；夹爪夹持平板（无 weld）
  CARRY — 夹爪夹持 + 双臂速度前馈沿 +X 慢移（无 mocap 硬焊）

控制链路（CARRY）：
  平板期望速度 → TCP 旋量 → DLS 伪逆 → 积分限位
  → τ = g(q) + Kp·e + Kd·ė → data.ctrl

运行：
  python dual_arm_controller.py --gui           # 推荐：有界面
  python dual_arm_controller.py --hold --gui
  python dual_arm_controller.py                 # 无界面 CARRY + CSV

版本：v1.3
日期：2026-08-08
"""

import mujoco
import mujoco.viewer
import numpy as np
import argparse
import os
import sys
import time
import csv

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
# CSV 日志
# =====================================================================

class CsvLogger:
    """短时运行全量必要数据记录，便于复盘调参"""

    HEADER = [
        't',
        'roll_deg', 'pitch_deg',
        'plate_x', 'plate_y', 'plate_z',
        'plate_des_x', 'plate_des_y', 'plate_des_z',
        'plate_err_xy', 'plate_err_z',
        'plate_roll_deg', 'plate_pitch_deg', 'plate_yaw_deg', 'level_err_deg',
        'ball_ox', 'ball_oy', 'ball_oz',
        'cup_x', 'cup_y', 'cup_z',
        'grip_contacts', 'ncon',
        'tcp_Lx', 'tcp_Ly', 'tcp_Lz',
        'tcp_Rx', 'tcp_Ry', 'tcp_Rz',
        'qL1', 'qL2', 'qL3', 'qL4', 'qL5', 'qL6', 'qL7',
        'qR1', 'qR2', 'qR3', 'qR4', 'qR5', 'qR6', 'qR7',
        'dqL1', 'dqL2', 'dqL3', 'dqL4', 'dqL5', 'dqL6', 'dqL7',
        'dqR1', 'dqR2', 'dqR3', 'dqR4', 'dqR5', 'dqR6', 'dqR7',
        'tauL1', 'tauL2', 'tauL3', 'tauL4', 'tauL5', 'tauL6', 'tauL7',
        'tauR1', 'tauR2', 'tauR3', 'tauR4', 'tauR5', 'tauR6', 'tauR7',
        'dq_cmd_L_norm', 'dq_cmd_R_norm',
        'tau_L_norm', 'tau_R_norm',
        'tcp_err_L', 'tcp_err_R',
    ]

    def __init__(self, path, log_every=1):
        self.path = path
        self.log_every = max(1, int(log_every))
        self.rows = []
        self.step_i = 0
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(self.path, 'w', newline='') as f:
            csv.writer(f).writerow(self.HEADER)
        print(f"[CSV] 记录到 {self.path} (every {self.log_every} step)")

    def maybe_log(self, ctrl, dq_left=None, dq_right=None, tcp_err_L=0.0, tcp_err_R=0.0):
        self.step_i += 1
        if (self.step_i - 1) % self.log_every != 0:
            return
        model, data = ctrl.model, ctrl.data
        roll, pitch = ctrl.get_chassis_orientation()
        plate = data.xpos[ctrl.plate_id]
        R = data.xmat[ctrl.plate_id].reshape(3, 3)
        # ZYX 近似：从旋转矩阵取 roll/pitch/yaw
        plate_yaw = np.arctan2(R[1, 0], R[0, 0])
        plate_pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        plate_roll = np.arctan2(R[2, 1], R[2, 2])
        level_err = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        des = ctrl.plate_desired_pos
        ball = data.xpos[ctrl.ball_id]
        cup = data.xpos[ctrl.cup_id]
        qL = ctrl.get_current_joint_positions('left')
        qR = ctrl.get_current_joint_positions('right')
        dqL = ctrl.get_current_joint_velocities('left')
        dqR = ctrl.get_current_joint_velocities('right')
        tauL = np.array([data.ctrl[a] for a in ctrl.left_act_ids], dtype=float)
        tauR = np.array([data.ctrl[a] for a in ctrl.right_act_ids], dtype=float)
        tcpL = data.xpos[ctrl.tcp_left_id]
        tcpR = data.xpos[ctrl.tcp_right_id]
        dq_left = np.zeros(7) if dq_left is None else dq_left
        dq_right = np.zeros(7) if dq_right is None else dq_right
        row = [
            float(data.time),
            float(np.degrees(roll)), float(np.degrees(pitch)),
            float(plate[0]), float(plate[1]), float(plate[2]),
            float(des[0]), float(des[1]), float(des[2]),
            float(np.linalg.norm(plate[:2] - des[:2])),
            float(plate[2] - des[2]),
            float(np.degrees(plate_roll)), float(np.degrees(plate_pitch)),
            float(np.degrees(plate_yaw)), float(level_err),
            float(ball[0] - plate[0]), float(ball[1] - plate[1]), float(ball[2] - plate[2]),
            float(cup[0]), float(cup[1]), float(cup[2]),
            int(ctrl._count_finger_plate_contacts()), int(data.ncon),
            float(tcpL[0]), float(tcpL[1]), float(tcpL[2]),
            float(tcpR[0]), float(tcpR[1]), float(tcpR[2]),
            *map(float, qL), *map(float, qR),
            *map(float, dqL), *map(float, dqR),
            *map(float, tauL), *map(float, tauR),
            float(np.linalg.norm(dq_left)), float(np.linalg.norm(dq_right)),
            float(np.linalg.norm(tauL)), float(np.linalg.norm(tauR)),
            float(tcp_err_L), float(tcp_err_R),
        ]
        self.rows.append(row)
        if len(self.rows) >= 50:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerows(self.rows)
        self.rows.clear()

    def close(self):
        self.flush()
        print(f"[CSV] 已保存 {self.path}")


def summarize_csv(path):
    """打印 CSV 关键指标，辅助找问题"""
    if not os.path.exists(path):
        print(f"[CSV] 文件不存在: {path}")
        return
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("[CSV] 空文件")
        return
    def col(name):
        return np.array([float(r[name]) for r in rows])

    t = col('t')
    print("\n========== CSV 复盘 ==========")
    print(f"  样本数={len(rows)}  t=[{t[0]:.3f},{t[-1]:.3f}]s")
    print(f"  level_err: max={col('level_err_deg').max():.3f}°  "
          f"mean={col('level_err_deg').mean():.3f}°")
    print(f"  plate_err_xy: max={col('plate_err_xy').max()*1000:.1f}mm  "
          f"final={col('plate_err_xy')[-1]*1000:.1f}mm")
    print(f"  plate_err_z:  max={abs(col('plate_err_z')).max()*1000:.1f}mm  "
          f"final={col('plate_err_z')[-1]*1000:.1f}mm")
    print(f"  plate_x: {col('plate_x')[0]:.3f} → {col('plate_x')[-1]:.3f} "
          f"(Δ={col('plate_x')[-1]-col('plate_x')[0]:.3f}m)")
    print(f"  grip_contacts: min={col('grip_contacts').min():.0f}  "
          f"mean={col('grip_contacts').mean():.1f}")
    print(f"  |pitch|: max={np.abs(col('pitch_deg')).max():.2f}°")
    ball_r = np.sqrt(col('ball_ox')**2 + col('ball_oy')**2)
    print(f"  ball_radial: max={ball_r.max()*1000:.1f}mm  final={ball_r[-1]*1000:.1f}mm  (仅观测)")
    print(f"  ‖τ_L‖ max={col('tau_L_norm').max():.1f}  ‖τ_R‖ max={col('tau_R_norm').max():.1f}")
    print(f"  ‖dq_cmd_L‖ max={col('dq_cmd_L_norm').max():.3f}  "
          f"‖dq_cmd_R‖ max={col('dq_cmd_R_norm').max():.3f}")
    if 'tcp_err_L' in rows[0]:
        print(f"  tcp_err: Lmax={col('tcp_err_L').max()*1000:.1f}mm  "
              f"Rmax={col('tcp_err_R').max()*1000:.1f}mm")
    # 夹爪夹持：握持接触与板跟踪均参与判据
    issues = []
    if col('grip_contacts').min() < 2 and (col('grip_contacts') < 2).mean() > 0.05:
        issues.append("握持接触丢失(grip_contacts<2 超过5%样本)")
    if col('level_err_deg').max() > 15.0:
        issues.append("平板倾角过大(>15°)")
    if col('plate_err_xy').max() > 0.08:
        issues.append("平板水平跟踪误差>80mm")
    if np.abs(col('pitch_deg')).max() > 10.0:
        issues.append("躯干pitch过大(>10°)")
    if 'tcp_err_L' in rows[0] and max(col('tcp_err_L').max(), col('tcp_err_R').max()) > 0.10:
        issues.append("TCP跟踪误差>100mm")
    if 'plate_z' in rows[0] and col('plate_z').min() < 0.85:
        issues.append("平板掉落(z<0.85)")
    if 'plate_err_z' in rows[0] and abs(col('plate_err_z')).max() > 0.10:
        issues.append("平板高度误差>100mm")
    dx = float(col('plate_x')[-1] - col('plate_x')[0])
    dx_des = float(col('plate_des_x')[-1] - col('plate_des_x')[0]) if 'plate_des_x' in rows[0] else dx
    if dx_des > 0.015 and dx < 0.5 * dx_des:
        issues.append(f"实际平移不足(Δx={dx:.3f}m < 0.5×指令{dx_des:.3f}m)")
    if issues:
        print("  ⚠ 发现问题:")
        for s in issues:
            print(f"    - {s}")
    else:
        print("  ✓ 未触发硬告警阈值")
    print("==============================\n")
    return issues

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
        self.dt = model.opt.timestep
        self.dls_lambda = 0.05
        self.carry_mode = not bool(getattr(args, 'hold', False))
        self.hold_mode = not self.carry_mode
        self.finger_grip = 0.012

        # 搬运轨迹：期望板心沿 +X（夹爪拖动，无 weld）
        self.carry_hold_s = float(getattr(args, 'carry_hold', 0.5))
        self.carry_move_s = float(getattr(args, 'carry_move', 6.0))
        self.carry_distance = float(getattr(args, 'carry_dist', 0.06))
        self.carry_vx = self.carry_distance / max(self.carry_move_s, 1e-3)
        self.plate_start_pos = None

        # 力矩 PD：τ = g(q) + Kp·e + Kd·ė
        self.kp_arm = np.array([80.0, 80.0, 60.0, 60.0, 30.0, 30.0, 25.0])
        self.kd_arm = np.array([10.0, 10.0,  8.0,  8.0,  3.0,  3.0,  2.5])
        self.ki_arm = np.zeros(7)
        self.i_err_left = np.zeros(7)
        self.i_err_right = np.zeros(7)
        self.i_err_limit = 0.2
        self.kp_leg = 400.0
        self.kd_leg = 40.0
        self.kp_trunk = 600.0
        self.kd_trunk = 60.0
        self.kp_head = 40.0
        self.kd_head = 4.0
        self.kp_finger = 300.0
        self.kd_finger = 12.0
        self.kp_tcp_pos = 6.0
        self.kp_tcp_rot = 4.0

        # ---- 运动学 ID 映射 ----
        self._init_kinematics()

        # ---- 状态变量 ----
        self.step_count = 0
        self.plate_desired_pos = np.array([0.35, 0.0, 0.98])
        self.prev_dq_left = np.zeros(7)
        self.prev_dq_right = np.zeros(7)
        self.hold_q_targets = {}
        self.q_des_left = np.zeros(7)
        self.q_des_right = np.zeros(7)
        self.q_init_left = np.zeros(7)
        self.q_init_right = np.zeros(7)

        self.scene_free_qpos = {}
        self.plate_hold_yaw = np.pi / 2
        self.grasp_off_L = np.zeros(3)
        self.grasp_off_R = np.zeros(3)
        self.grasp_R_L = np.eye(3)
        self.grasp_R_R = np.eye(3)

        # CSV
        self.csv_logger = None
        self._last_dq_left = np.zeros(7)
        self._last_dq_right = np.zeros(7)

        # ---- 日志 ----
        self.log_interval = max(1, int(0.5 / self.dt))

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
        self.cup_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
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
    # 平板速度旋量 → TCP 速度变换
    # ==================================================================

    def plate_twist_to_tcp_velocity(self, v_plate, w_plate, tcp_body_id,
                                    grasp_off=None):
        """将平板中心速度旋量变换为 TCP 速度。"""
        if grasp_off is not None:
            R_p = self.data.xmat[self.plate_id].reshape(3, 3)
            r_tcp = R_p @ grasp_off
        else:
            plate_pos = self.data.xpos[self.plate_id]
            r_tcp = self.data.xpos[tcp_body_id] - plate_pos
        v_tcp = v_plate + np.cross(w_plate, r_tcp)
        return np.concatenate([v_tcp, w_plate.copy()])

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

        self._configure_grasp_collisions()
        self._prepare_hold_scene()

    def _configure_grasp_collisions(self):
        """关闭机器人网格自碰；保留手爪↔板/杯/球接触。"""
        model = self.model
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

        finger_bodies = {
            'leftfinger1_Link', 'leftfinger2_Link',
            'rightfinger1_Link', 'rightfinger2_Link',
        }
        grasp_bodies = {'plate', 'cup', 'ball', 'water_mass'} | finger_bodies

        n_off = 0
        for gid in range(model.ngeom):
            bid = int(model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ''

            if bname in grasp_bodies:
                model.geom_contype[gid] = 1
                model.geom_conaffinity[gid] = 1
                model.geom_friction[gid] = [2.5, 0.01, 0.001]
                continue

            if bname == 'plate_mocap':
                model.geom_contype[gid] = 0
                model.geom_conaffinity[gid] = 0
                continue

            if bid == 0 or bname in ('', 'world'):
                if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                    model.geom_contype[gid] = 0
                    model.geom_conaffinity[gid] = 0
                    n_off += 1
                continue

            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0
            n_off += 1

        for gid in range(model.ngeom):
            bid = int(model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ''
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            if bname == 'plate' or 'pad' in gname or 'finger' in bname:
                if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
                    continue
                model.geom_friction[gid] = [8.0, 0.2, 0.02]
                model.geom_solref[gid] = [0.003, 1.0]
                model.geom_condim[gid] = 4
        print(f"[Collision] 关闭 {n_off} 个机器人网格；保留手爪↔板接触")

    def _disable_plate_mocap_weld(self):
        """确保平板不被 mocap 焊接（夹爪夹持模式）。"""
        eq_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "plate_mocap_weld")
        if eq_id < 0:
            print("[Grasp] plate_mocap_weld 未定义（已取消硬焊）")
            return
        if hasattr(self.data, "eq_active"):
            self.data.eq_active[eq_id] = 0
        if hasattr(self.model, "eq_active0"):
            self.model.eq_active0[eq_id] = 0
        print("[Grasp] 已禁用 plate_mocap_weld，改用夹爪夹持")

    def _prepare_hold_scene(self):
        """板 Rz90，手指夹紧，记录夹持相对位姿（无 mocap 硬焊）"""
        model, data = self.model, self.data
        self._disable_plate_mocap_weld()

        self.scene_free_qpos = {}
        for name in ("plate", "cup", "water_mass", "ball"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            jid = model.body_jntadr[bid]
            qadr = int(model.jnt_qposadr[jid])
            self.scene_free_qpos[name] = (qadr, data.qpos[qadr:qadr + 7].copy())

        self.plate_hold_yaw = np.pi / 2
        # 仅摆初始姿态；之后靠夹爪，不驱动 mocap 焊
        if "plate" in self.scene_free_qpos:
            qadr, q7 = self.scene_free_qpos["plate"]
            q7 = q7.copy()
            q7[:3] = self.plate_desired_pos
            s2 = np.sin(self.plate_hold_yaw / 2.0)
            c2 = np.cos(self.plate_hold_yaw / 2.0)
            q7[3:] = [c2, 0.0, 0.0, s2]
            data.qpos[qadr:qadr + 7] = q7
            self.scene_free_qpos["plate"] = (qadr, q7.copy())

        gval = self.finger_grip
        for fname, val in (('leftfinger1_joint', gval), ('leftfinger2_joint', -gval),
                           ('rightfinger1_joint', gval), ('rightfinger2_joint', -gval)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = val

        mujoco.mj_forward(model, data)
        n_fp = self._count_finger_plate_contacts()

        # 记录夹持相对位姿（板坐标系）：搬运时 TCP 伺服到该相对位姿
        R_p = data.xmat[self.plate_id].reshape(3, 3).copy()
        p_p = data.xpos[self.plate_id].copy()
        p_L = data.xpos[self.tcp_left_id].copy()
        p_R = data.xpos[self.tcp_right_id].copy()
        R_L = data.xmat[self.tcp_left_id].reshape(3, 3).copy()
        R_R = data.xmat[self.tcp_right_id].reshape(3, 3).copy()
        self.grasp_off_L = R_p.T @ (p_L - p_p)
        self.grasp_off_R = R_p.T @ (p_R - p_p)
        self.grasp_R_L = R_p.T @ R_L
        self.grasp_R_R = R_p.T @ R_R
        print(f"[Hold] 板 Rz90°；手指夹紧；finger-plate 接触对={n_fp}")
        print(f"[Grasp] off_L={np.round(self.grasp_off_L, 3)}  "
              f"off_R={np.round(self.grasp_off_R, 3)}")

    def _ik_q_des(self, tcp_body_id, joint_ids, joint_ranges, p_des, R_des,
                  q0, n_iter=6):
        """临时改 qpos 做几步位置IK，返回 q_des 后恢复仿真状态"""
        model, data = self.model, self.data
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()

        def set_q(q):
            for k, jid in enumerate(joint_ids):
                lo, hi = joint_ranges[k]
                qq = q[k]
                if lo < hi:
                    qq = np.clip(qq, lo, hi)
                data.qpos[model.jnt_qposadr[jid]] = qq

        def get_q():
            return np.array([data.qpos[model.jnt_qposadr[j]] for j in joint_ids])

        q = np.array(q0, dtype=float)
        set_q(q)
        for _ in range(n_iter):
            mujoco.mj_forward(model, data)
            p = data.xpos[tcp_body_id]
            R = data.xmat[tcp_body_id].reshape(3, 3)
            e_pos = p_des - p
            R_err = R_des @ R.T
            e_rot = 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])
            if np.linalg.norm(e_pos) < 0.002 and np.linalg.norm(e_rot) < 0.02:
                break
            twist = np.concatenate([e_pos, 0.4 * e_rot])
            J = self.compute_arm_jacobian(tcp_body_id, joint_ids)
            dq = self.solve_ik_dls(J, twist)
            dq = np.clip(dq, -0.2, 0.2)
            q = q + dq
            set_q(q)

        q_out = get_q()
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_forward(model, data)
        return q_out

    def _tcp_pose_servo(self, tcp_body_id, grasp_off, grasp_R_rel):
        """相对实际板的 TCP 位姿误差 → 速度旋量修正。"""
        data = self.data
        R_p = data.xmat[self.plate_id].reshape(3, 3)
        p_p = data.xpos[self.plate_id]
        p_des = p_p + R_p @ grasp_off
        R_des = R_p @ grasp_R_rel

        p = data.xpos[tcp_body_id]
        R = data.xmat[tcp_body_id].reshape(3, 3)
        e_pos = p_des - p
        R_err = R_des @ R.T
        e_rot = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        twist = np.zeros(6)
        twist[:3] = self.kp_tcp_pos * e_pos
        twist[3:] = self.kp_tcp_rot * e_rot
        return twist, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))

    def _count_finger_plate_contacts(self):
        """统计手指与平板之间的接触数量"""
        model, data = self.model, self.data
        plate_id = self.plate_id
        finger_ids = set()
        for name in ('leftfinger1_Link', 'leftfinger2_Link',
                     'rightfinger1_Link', 'rightfinger2_Link'):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                finger_ids.add(bid)
        n = 0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if (b1 == plate_id and b2 in finger_ids) or \
               (b2 == plate_id and b1 in finger_ids):
                n += 1
        return n

    def _set_plate_mocap_yaw(self, yaw):
        """设置平板 mocap 位姿：水平 + 指定 yaw"""
        self.data.mocap_pos[self.mocap_idx] = self.plate_desired_pos
        self.data.mocap_quat[self.mocap_idx] = np.array([
            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)
        ])

    # ==================================================================
    # 平板 Mocap 更新
    # ==================================================================

    def update_plate_mocap(self, yaw=None):
        """
        更新平板 mocap：绝对水平 + 指定/当前 yaw。
        搬运时强制 yaw=plate_hold_yaw(90°)，避免板转回去。
        """
        if yaw is None:
            plate_xmat = self.data.xmat[self.plate_id].reshape(3, 3).copy()
            yaw = np.arctan2(plate_xmat[1, 0], plate_xmat[0, 0])

        self._set_plate_mocap_yaw(yaw)

    def _update_carry_plate_target(self):
        """先静止再沿 +X 匀速平移 plate_desired_pos（mocap 跟随）。"""
        if self.plate_start_pos is None:
            self.plate_start_pos = self.plate_desired_pos.copy()

        t = float(self.data.time)
        if t <= self.carry_hold_s:
            self.plate_desired_pos[:] = self.plate_start_pos
            return np.zeros(3)

        t_move = t - self.carry_hold_s
        if t_move >= self.carry_move_s:
            self.plate_desired_pos[:] = self.plate_start_pos + np.array(
                [self.carry_distance, 0.0, 0.0])
            return np.zeros(3)

        self.plate_desired_pos[:] = self.plate_start_pos + np.array(
            [self.carry_vx * t_move, 0.0, 0.0])
        return np.array([self.carry_vx, 0.0, 0.0])

    # ==================================================================
    # 主控制循环（单步）
    # ==================================================================

    def control_step(self):
        """每步仿真：HOLD 保持 或 CARRY 夹爪夹持 + 双臂跟踪"""

        if self.hold_mode:
            self.write_torque_ctrl(
                self.q_des_left, np.zeros(7),
                self.q_des_right, np.zeros(7))
            self._last_dq_left[:] = 0.0
            self._last_dq_right[:] = 0.0
            if self.csv_logger is not None:
                self.csv_logger.maybe_log(self)

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
                      f"| ‖Δq_L‖={err_l:.4f} ‖Δq_R‖={err_r:.4f} "
                      f"grip={self._count_finger_plate_contacts()}")
            return

        # === CARRY：夹爪夹持 + 手臂速度前馈（无 mocap 硬焊） ===
        v_plate_des = self._update_carry_plate_target()
        w_plate_des = np.zeros(3)
        plate = self.data.xpos[self.plate_id]
        R_p = self.data.xmat[self.plate_id].reshape(3, 3)
        epos_L = float(np.linalg.norm(
            plate + R_p @ self.grasp_off_L - self.data.xpos[self.tcp_left_id]))
        epos_R = float(np.linalg.norm(
            plate + R_p @ self.grasp_off_R - self.data.xpos[self.tcp_right_id]))

        twist_left = self.plate_twist_to_tcp_velocity(
            v_plate_des, w_plate_des, self.tcp_left_id, self.grasp_off_L)
        twist_right = self.plate_twist_to_tcp_velocity(
            v_plate_des, w_plate_des, self.tcp_right_id, self.grasp_off_R)
        servo_L, _, _ = self._tcp_pose_servo(
            self.tcp_left_id, self.grasp_off_L, self.grasp_R_L)
        servo_R, _, _ = self._tcp_pose_servo(
            self.tcp_right_id, self.grasp_off_R, self.grasp_R_R)
        twist_left = twist_left + servo_L
        twist_right = twist_right + servo_R

        J_left = self.compute_arm_jacobian(self.tcp_left_id, self.left_joint_ids)
        J_right = self.compute_arm_jacobian(self.tcp_right_id, self.right_joint_ids)
        dq_left = self.solve_ik_dls(J_left, twist_left)
        dq_right = self.solve_ik_dls(J_right, twist_right)
        alpha = 0.7
        dq_left = alpha * dq_left + (1 - alpha) * self.prev_dq_left
        dq_right = alpha * dq_right + (1 - alpha) * self.prev_dq_right
        self.prev_dq_left = dq_left.copy()
        self.prev_dq_right = dq_right.copy()

        self.q_des_left = self.integrate_and_clamp(
            dq_left, self.q_des_left, self.left_joint_ranges)
        self.q_des_right = self.integrate_and_clamp(
            dq_right, self.q_des_right, self.right_joint_ranges)

        self.write_torque_ctrl(
            self.q_des_left, dq_left, self.q_des_right, dq_right)

        if self.csv_logger is not None:
            self.csv_logger.maybe_log(self, dq_left, dq_right, epos_L, epos_R)

        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            roll, pitch = self.get_chassis_orientation()
            plate_z = R_p[:, 2]
            level_err = np.degrees(np.arccos(np.clip(plate_z[2], -1, 1)))
            print(f"[CARRY t={self.data.time:5.2f}s] "
                  f"roll={np.degrees(roll):+5.1f}° pitch={np.degrees(pitch):+5.1f}° "
                  f"| level={level_err:5.2f}° grip={self._count_finger_plate_contacts()} "
                  f"plate=({plate[0]:.3f},{plate[1]:.3f},{plate[2]:.3f}) "
                  f"des_x={self.plate_desired_pos[0]:.3f} "
                  f"eTCP=({epos_L*1000:.1f}/{epos_R*1000:.1f}mm)")

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

        self.controller = DualArmPlateController(self.model, self.data, self.args)
        self.controller.plate_desired_pos = np.array(
            [plate_center_x, plate_center_y, plate_surface_z])
        self.controller.finger_grip = 0.010
        self.controller.lock_hold_targets()

        mode = "HOLD" if self.controller.hold_mode else "CARRY"
        print(f"  控制模式: {mode}")

        csv_path = getattr(self.args, 'csv', None)
        if csv_path:
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(os.path.dirname(self.xml_path), csv_path)
            every = max(1, int(getattr(self.args, 'csv_every', 5)))
            self.controller.csv_logger = CsvLogger(csv_path, log_every=every)

        # 首帧写入平衡力矩，避免 mj_step 前零力矩导致塌陷
        self.controller.write_torque_ctrl(
            q_left_init, np.zeros(7), q_right_init, np.zeros(7))

    def step(self):
        """执行单步仿真"""
        self.controller.control_step()
        mujoco.mj_step(self.model, self.data)

    def run_headless(self):
        """无 GUI 模式运行"""
        mode = "HOLD" if self.controller.hold_mode else "CARRY"
        print(f"[Run] 无GUI模式 [{mode}]，仿真时长 {self.args.duration}s")
        total_steps = int(self.args.duration / self.model.opt.timestep)

        max_err_l = 0.0
        max_err_r = 0.0
        max_pitch = 0.0
        err_l = err_r = 0.0
        pitch = 0.0

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
                progress = 100.0 * step / max(total_steps, 1)
                print(f"[Progress] {progress:5.1f}%  elapsed={elapsed:.1f}s  "
                      f"‖Δq_L‖={err_l:.4f} ‖Δq_R‖={err_r:.4f} "
                      f"|pitch|={np.degrees(abs(pitch)):.2f}°")

        elapsed = time.time() - t_start
        print(f"[Done] 仿真完成，实际用时 {elapsed:.1f}s，"
              f"RTF = {self.args.duration / max(elapsed, 1e-6):.2f}")
        print(f"[Stability] max‖Δq_L‖={max_err_l:.4f}  max‖Δq_R‖={max_err_r:.4f}  "
              f"max|pitch|={np.degrees(max_pitch):.2f}°")

        if self.controller.csv_logger is not None:
            self.controller.csv_logger.close()
            issues = summarize_csv(self.controller.csv_logger.path)
        else:
            issues = []

        self._print_state()
        if self.controller.hold_mode:
            ok = (max_err_l < 0.05 and max_err_r < 0.05 and max_pitch < np.radians(5.0))
        else:
            # CARRY：手臂应跟板移动，不以初始 Δq 判失败；看 CSV 板跟踪与躯干
            ok = max_pitch < np.radians(10.0)
        if issues:
            ok = False
        print(f"[Verdict] {'PASS' if ok else 'FAIL'} "
              f"({'CARRY' if not self.controller.hold_mode else 'HOLD'})")
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
        description="双臂协同稳载仿真：HOLD / CARRY（夹爪夹持，无 mocap 硬焊）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dual_arm_controller.py --gui                 # 有界面 CARRY（推荐）
  python dual_arm_controller.py --hold --gui
  python dual_arm_controller.py --duration 8          # 无界面
        """)

    parser.add_argument('--gui', action='store_true',
                        help='启用 MuJoCo 可视化')
    parser.add_argument('--step', action='store_true',
                        help='逐步仿真（按 Enter 前进）')
    parser.add_argument('--duration', type=float, default=7.5,
                        help='仿真时长(s)')
    parser.add_argument('--hold', action='store_true',
                        help='HOLD：仅保持初始姿态')
    parser.add_argument('--carry-hold', type=float, default=0.5,
                        help='搬运前静止时间(s)')
    parser.add_argument('--carry-move', type=float, default=6.0,
                        help='平移持续时间(s)')
    parser.add_argument('--carry-dist', type=float, default=0.06,
                        help='沿+X平移距离(m)')
    parser.add_argument('--csv', type=str, default='logs/carry_last.csv',
                        help='CSV 日志路径')
    parser.add_argument('--csv-every', type=int, default=5,
                        help='每隔多少仿真步写一行CSV')
    parser.add_argument('--xml', type=str,
                        default='scene_dual_arm_plate.xml',
                        help='MJCF 场景文件路径')
    return parser.parse_args()


def main():
    args = parse_args()

    xml_path = args.xml
    if not os.path.isabs(xml_path):
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

    mode = 'HOLD' if args.hold else 'CARRY'
    print("=" * 60)
    print("  轮式移动双臂机器人协同稳载仿真系统")
    print("  BalanceDual-Arm v1.3")
    print("=" * 60)
    print(f"  MJCF:    {xml_path}")
    print(f"  GUI:     {args.gui}")
    print(f"  Mode:    {mode}")
    print(f"  Duration:{args.duration}s")
    print(f"  CSV:     {args.csv}")
    if mode == 'CARRY':
        print(f"  Carry:   hold={args.carry_hold}s move={args.carry_move}s "
              f"dist={args.carry_dist}m  [grip-only]")
    print("=" * 60)

    runner = SimulationRunner(xml_path, args)
    runner.run()


if __name__ == '__main__':
    main()
