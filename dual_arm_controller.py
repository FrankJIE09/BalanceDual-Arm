#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轮式移动双臂机器人协同稳载控制系统
===============================================
项目：BalanceDual-Arm

模式：
  HOLD  — 世界系锁定初始法兰 TCP 位姿（DLS 逆解）+ 手指力偏置
  CARRY — 期望板位姿误差(板系)→板速→TCP→DLS

控制链路：
  （板旋量）+ TCP 位姿误差 → DLS 伪逆 → 积分限位
  → τ = g(q) + Kp·e + Kd·ė → data.ctrl
  手指：τ = g + Kp·e + Kd·ė + bias_close

运行：
  python dual_arm_controller.py                 # 读 config.yaml
  python dual_arm_controller.py --config my.yaml
  # sim.keyboard=true 时 GUI 下可用 qawsedrftgyh 遥操期望板 6D

版本：v1.3
日期：2026-08-09
"""

import mujoco
import mujoco.viewer
import numpy as np
import argparse
import os
import sys
import time
import csv
from types import SimpleNamespace

try:
    import yaml
except ImportError as e:
    raise SystemExit("需要 PyYAML：pip install pyyaml") from e

try:
    from pynput import keyboard as pynput_keyboard
except ImportError as e:
    pynput_keyboard = None
    _PYNPUT_IMPORT_ERROR = e
else:
    _PYNPUT_IMPORT_ERROR = None

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
# 键盘遥操（pynput）：期望板体 6 自由度
#   q/a ±X   w/s ±Y   e/d ±Z   r/f ±roll   t/g ±pitch   y/h ±yaw
# =====================================================================

class PlateKeyboardTeleop:
    """按住键产生板体系线速度 / 角速度指令。"""

    # char -> (vx, vy, vz, wx, wy, wz) 单位方向
    _DIR = {
        'q': ( 1, 0, 0, 0, 0, 0),
        'a': (-1, 0, 0, 0, 0, 0),
        'w': ( 0, 1, 0, 0, 0, 0),
        's': ( 0,-1, 0, 0, 0, 0),
        'e': ( 0, 0, 1, 0, 0, 0),
        'd': ( 0, 0,-1, 0, 0, 0),
        'r': ( 0, 0, 0, 1, 0, 0),
        'f': ( 0, 0, 0,-1, 0, 0),
        't': ( 0, 0, 0, 0, 1, 0),
        'g': ( 0, 0, 0, 0,-1, 0),
        'y': ( 0, 0, 0, 0, 0, 1),
        'h': ( 0, 0, 0, 0, 0,-1),
    }

    def __init__(self, v_lin=0.08, w_ang=0.4):
        if pynput_keyboard is None:
            raise RuntimeError(
                f"键盘模式需要 pynput：pip install pynput "
                f"(import error: {_PYNPUT_IMPORT_ERROR})")
        self.v_lin = float(v_lin)
        self.w_ang = float(w_ang)
        self._pressed = set()
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    @staticmethod
    def help_text():
        return (
            "键盘遥操（期望板体坐标）：\n"
            "  q/a ±X   w/s ±Y   e/d ±Z\n"
            "  r/f ±roll   t/g ±pitch   y/h ±yaw\n"
            "  （按住持续运动；关 GUI 窗口退出）"
        )

    def _key_char(self, key):
        try:
            ch = key.char
        except AttributeError:
            return None
        if ch is None:
            return None
        return ch.lower()

    def _on_press(self, key):
        ch = self._key_char(key)
        if ch in self._DIR:
            self._pressed.add(ch)

    def _on_release(self, key):
        ch = self._key_char(key)
        if ch in self._DIR:
            self._pressed.discard(ch)

    def body_twist(self):
        """返回板体系 (v[3], w[3])，m/s 与 rad/s。"""
        v = np.zeros(3)
        w = np.zeros(3)
        for ch in list(self._pressed):
            d = self._DIR.get(ch)
            if d is None:
                continue
            v += self.v_lin * np.array(d[:3], dtype=float)
            w += self.w_ang * np.array(d[3:], dtype=float)
        return v, w

    def stop(self):
        try:
            self._listener.stop()
        except Exception:
            pass


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
        'q_fL1', 'q_fL2', 'q_fR1', 'q_fR2',
        'tau_fL', 'tau_fR', 'e_fL', 'e_fR',
        'cf_fL', 'cf_fR',
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
        tcpL = data.site_xpos[ctrl.tcp_left_id]
        tcpR = data.site_xpos[ctrl.tcp_right_id]
        dq_left = np.zeros(7) if dq_left is None else dq_left
        dq_right = np.zeros(7) if dq_right is None else dq_right
        fd = ctrl.get_finger_debug()
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
            int(fd['n_contact']), int(data.ncon),
            float(fd['q_L1']), float(fd['q_L2']), float(fd['q_R1']), float(fd['q_R2']),
            float(fd['tau_L']), float(fd['tau_R']), float(fd['e_L']), float(fd['e_R']),
            float(fd['cf_L']), float(fd['cf_R']),
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
          f"mean={col('grip_contacts').mean():.1f}  (仅观测，mocap-weld 不依赖摩擦)")
    print(f"  |pitch|: max={np.abs(col('pitch_deg')).max():.2f}°")
    ball_r = np.sqrt(col('ball_ox')**2 + col('ball_oy')**2)
    print(f"  ball_radial: max={ball_r.max()*1000:.1f}mm  final={ball_r[-1]*1000:.1f}mm  (仅观测)")
    if 'tau_fL' in rows[0]:
        print(f"  finger τ: L=[{col('tau_fL').min():+.1f},{col('tau_fL').max():+.1f}]N  "
              f"R=[{col('tau_fR').min():+.1f},{col('tau_fR').max():+.1f}]N")
        print(f"  finger q1: L=[{col('q_fL1').min()*1000:.1f},{col('q_fL1').max()*1000:.1f}]mm  "
              f"R=[{col('q_fR1').min()*1000:.1f},{col('q_fR1').max()*1000:.1f}]mm")
        print(f"  finger e:  Lmax={abs(col('e_fL')).max()*1000:.1f}mm  "
              f"Rmax={abs(col('e_fR')).max()*1000:.1f}mm")
        print(f"  finger cf: Lmax={col('cf_fL').max():.1f}N  Rmax={col('cf_fR').max():.1f}N")
    print(f"  ‖τ_L‖ max={col('tau_L_norm').max():.1f}  ‖τ_R‖ max={col('tau_R_norm').max():.1f}")
    print(f"  ‖dq_cmd_L‖ max={col('dq_cmd_L_norm').max():.3f}  "
          f"‖dq_cmd_R‖ max={col('dq_cmd_R_norm').max():.3f}")
    if 'tcp_err_L' in rows[0]:
        print(f"  tcp_err: Lmax={col('tcp_err_L').max()*1000:.1f}mm  "
              f"Rmax={col('tcp_err_R').max()*1000:.1f}mm")
    # mocap-weld：判据看板跟踪与躯干稳定，不看摩擦握持/小球驻留
    issues = []
    if col('level_err_deg').max() > 8.0:
        issues.append("平板倾角过大(>8°)")
    if col('plate_err_xy').max() > 0.05:
        issues.append("平板水平跟踪误差>50mm")
    if np.abs(col('pitch_deg')).max() > 8.0:
        issues.append("躯干pitch过大(>8°)")
    if 'tcp_err_L' in rows[0] and max(col('tcp_err_L').max(), col('tcp_err_R').max()) > 0.08:
        issues.append("TCP跟踪误差>80mm")
    if 'plate_z' in rows[0] and col('plate_z').min() < 0.85:
        issues.append("平板掉落(z<0.85)")
    if 'plate_err_z' in rows[0] and abs(col('plate_err_z')).max() > 0.08:
        issues.append("平板高度误差>80mm")
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

    def __init__(self, model, data, cfg):
        """
        初始化控制器

        Args:
            model: MuJoCo MjModel
            data:   MuJoCo MjData
            cfg:    YAML 配置（SimpleNamespace）
        """
        self.model = model
        self.data = data
        self.cfg = cfg
        ctrl = cfg.control
        carry = cfg.carry
        sim = cfg.sim

        # ---- 控制参数 ----
        self.dt = model.opt.timestep
        self.dls_lambda = float(ctrl.dls_lambda)
        mode = str(sim.mode).strip().lower()
        self.hold_mode = mode in ('hold',)
        self.carry_mode = not self.hold_mode
        self.finger_grip = 0.0

        # 初始位姿 / 期望位姿分开；速度由误差反馈产生
        self.carry_settle_s = float(getattr(carry, 'settle_s',
                                            getattr(carry, 'hold_s', 0.5)))
        self.carry_move_s = float(carry.move_s)
        self.init_rpy, self.target_delta_body, self.target_rpy = (
            self._parse_carry_poses(carry))
        # 兼容旧字段：场景初始 yaw
        self.plate_hold_yaw = float(self.init_rpy[2])
        self.plate_start_pos = None
        self._plate_des_prev = None
        # 姿态轨迹在四元数上 slerp（配置仍用 rpy 读写）
        self.init_quat = self._rpy_to_quat(self.init_rpy)
        self.target_quat = self._rpy_to_quat(self.target_rpy)
        self.plate_desired_quat = self.init_quat.copy()

        # 键盘遥操（GUI + sim.keyboard）；偏移叠在轨迹目标上
        self.keyboard_enabled = bool(getattr(sim, 'keyboard', False))
        teleop_cfg = getattr(carry, 'teleop', None)
        self._teleop_v_lin = float(getattr(teleop_cfg, 'v_lin', 0.08)
                                   if teleop_cfg is not None else 0.08)
        self._teleop_w_ang = float(getattr(teleop_cfg, 'w_ang', 0.4)
                                   if teleop_cfg is not None else 0.4)
        self.teleop = None
        self._kb_pos_off = np.zeros(3)
        self._kb_R_off = np.eye(3)

        # 力矩 PD：τ = g(q) + Kp·e + Kd·ė
        self.kp_arm = np.asarray(ctrl.kp_arm, dtype=float)
        self.kd_arm = np.asarray(ctrl.kd_arm, dtype=float)
        self.ki_arm = np.asarray(ctrl.ki_arm, dtype=float)
        self.i_err_left = np.zeros(7)
        self.i_err_right = np.zeros(7)
        self.i_err_limit = float(ctrl.i_err_limit)
        self.kp_leg = float(ctrl.kp_leg)
        self.kd_leg = float(ctrl.kd_leg)
        self.kp_trunk = float(ctrl.kp_trunk)
        self.kd_trunk = float(ctrl.kd_trunk)
        self.kp_head = float(ctrl.kp_head)
        self.kd_head = float(ctrl.kd_head)
        self.kp_finger = float(ctrl.kp_finger)
        self.kd_finger = float(ctrl.kd_finger)
        self.finger_close_bias = float(ctrl.finger_close_bias)
        self.kp_tcp_pos = float(ctrl.kp_tcp_pos)
        self.kp_tcp_rot = float(ctrl.kp_tcp_rot)
        # 板位姿误差 → 板体系期望速度（外环）
        self.kp_plate_pos = float(getattr(ctrl, 'kp_plate_pos', 2.0))
        self.kp_plate_rot = float(getattr(ctrl, 'kp_plate_rot', 2.0))
        self.v_plate_body_limit = float(getattr(ctrl, 'v_plate_body_limit', 0.3))
        self.w_plate_body_limit = float(getattr(ctrl, 'w_plate_body_limit', 1.0))

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
        self.grasp_off_L = np.zeros(3)
        self.grasp_off_R = np.zeros(3)
        self.grasp_R_L = np.eye(3)
        self.grasp_R_R = np.eye(3)
        # HOLD：世界系 TCP 位姿锁定目标
        self.tcp_hold_pos_L = np.zeros(3)
        self.tcp_hold_pos_R = np.zeros(3)
        self.tcp_hold_R_L = np.eye(3)
        self.tcp_hold_R_R = np.eye(3)

        # CSV
        self.csv_logger = None
        self._last_dq_left = np.zeros(7)
        self._last_dq_right = np.zeros(7)

        # 手指诊断缓存（write_torque_ctrl 写入）
        self._finger_dbg = {
            'tau_L': 0.0, 'tau_R': 0.0,
            'tau_raw_L': 0.0, 'tau_raw_R': 0.0,
            'e_L': 0.0, 'e_R': 0.0,
            'g_L': 0.0, 'g_R': 0.0,
            'bias_L': 0.0, 'bias_R': 0.0,
            'clipped_L': False, 'clipped_R': False,
        }

        # ---- 日志 ----
        log_s = float(getattr(sim, 'log_interval_s', 0.5))
        self.log_interval = max(1, int(log_s / self.dt))

    @staticmethod
    def _parse_rpy(ns, default_rpy):
        """从 {rpy:[r,p,y]} 或 {yaw:} 解析世界系 RPY。"""
        rpy = np.asarray(default_rpy, dtype=float).reshape(3).copy()
        if ns is None:
            return rpy
        if hasattr(ns, 'rpy'):
            return np.asarray(ns.rpy, dtype=float).reshape(3)
        if hasattr(ns, 'yaw'):
            rpy[2] = float(ns.yaw)
        return rpy

    @staticmethod
    def _parse_carry_poses(carry):
        """
        解析 init_pose / target_pose。

        init_pose.rpy：场景初始姿态
        target_pose.{delta_body, rpy}：控制期望终点
        兼容旧键 target_delta_body / target_yaw。
        """
        default_rpy = np.array([0.0, 0.0, np.pi / 2])
        init_rpy = DualArmPlateController._parse_rpy(
            getattr(carry, 'init_pose', None), default_rpy)
        if (getattr(carry, 'init_pose', None) is None
                and hasattr(carry, 'target_yaw')):
            # 旧配置：target_yaw 曾兼作初始 yaw
            init_rpy = np.array([0.0, 0.0, float(carry.target_yaw)])

        delta = np.zeros(3)
        tp = getattr(carry, 'target_pose', None)
        if tp is not None:
            if hasattr(tp, 'delta_body'):
                delta = np.asarray(tp.delta_body, dtype=float).reshape(3)
            # 未写 target rpy 时默认与初始姿态相同（只移位置）
            target_rpy = DualArmPlateController._parse_rpy(tp, init_rpy)
            return init_rpy, delta, target_rpy

        if hasattr(carry, 'target_delta_body'):
            delta = np.asarray(carry.target_delta_body, dtype=float).reshape(3)
        elif hasattr(carry, 'dist'):
            dist = float(carry.dist)
            v_dir = np.asarray(getattr(carry, 'v_body_dir', [1.0, 0.0, 0.0]),
                               dtype=float)
            n = float(np.linalg.norm(v_dir))
            v_dir = v_dir / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
            delta = v_dir * dist
        target_rpy = init_rpy.copy()
        if hasattr(carry, 'target_yaw'):
            target_rpy = np.array([0.0, 0.0, float(carry.target_yaw)])
        return init_rpy, delta, target_rpy

    @staticmethod
    def _rpy_to_R(rpy):
        """世界系 RPY → 旋转矩阵，R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
        roll, pitch, yaw = [float(x) for x in rpy]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, cr, -sr],
                       [0.0, sr,  cr]])
        Ry = np.array([[cp, 0.0, sp],
                       [0.0, 1.0, 0.0],
                       [-sp, 0.0, cp]])
        Rz = np.array([[cy, -sy, 0.0],
                       [sy,  cy, 0.0],
                       [0.0, 0.0, 1.0]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _R_to_quat_wxyz(R):
        """旋转矩阵 → MuJoCo quat [w,x,y,z]。"""
        R = np.asarray(R, dtype=float).reshape(3, 3)
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0.0:
            s = 0.5 / np.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([w, x, y, z], dtype=float)
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _rpy_to_quat(rpy):
        """世界系 RPY → MuJoCo quat [w,x,y,z]。"""
        return DualArmPlateController._R_to_quat_wxyz(
            DualArmPlateController._rpy_to_R(rpy))

    @staticmethod
    def _quat_to_R(q):
        """MuJoCo quat [w,x,y,z] → 旋转矩阵。"""
        q = np.asarray(q, dtype=float).reshape(4)
        n = float(np.linalg.norm(q))
        if n < 1e-12:
            return np.eye(3)
        w, x, y, z = q / n
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    @staticmethod
    def _R_to_rpy(R):
        """旋转矩阵 → 世界系 RPY（与 _rpy_to_R 约定一致，仅用于日志）。"""
        R = np.asarray(R, dtype=float).reshape(3, 3)
        pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        cp = float(np.cos(pitch))
        if abs(cp) < 1e-8:
            # gimbal lock：roll=0，yaw 从 R[0:2,1] 取
            roll = 0.0
            yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
        else:
            roll = float(np.arctan2(R[2, 1], R[2, 2]))
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return np.array([roll, pitch, yaw])

    @staticmethod
    def _quat_slerp(q0, q1, alpha):
        """单位四元数球面线性插值，α∈[0,1]。"""
        q0 = np.asarray(q0, dtype=float).reshape(4)
        q1 = np.asarray(q1, dtype=float).reshape(4)
        n0 = float(np.linalg.norm(q0))
        n1 = float(np.linalg.norm(q1))
        q0 = q0 / n0 if n0 > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
        q1 = q1 / n1 if n1 > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
        dot = float(np.dot(q0, q1))
        # 走短弧
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + alpha * (q1 - q0)
            n = float(np.linalg.norm(q))
            return q / n if n > 1e-12 else q0.copy()
        theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sin_0 = float(np.sin(theta_0))
        theta = theta_0 * float(alpha)
        s0 = float(np.sin(theta_0 - theta)) / sin_0
        s1 = float(np.sin(theta)) / sin_0
        q = s0 * q0 + s1 * q1
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-12 else q0.copy()

    @staticmethod
    def _skew(v):
        x, y, z = [float(a) for a in v]
        return np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ])

    @staticmethod
    def _exp_so3(w_dt):
        """旋转向量 → SO(3)（Rodrigues）。"""
        w_dt = np.asarray(w_dt, dtype=float).reshape(3)
        th = float(np.linalg.norm(w_dt))
        if th < 1e-12:
            return np.eye(3)
        k = w_dt / th
        K = DualArmPlateController._skew(k)
        return (np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K))

    def enable_keyboard_teleop(self):
        """启动 pynput 监听（仅 GUI 调用）。"""
        if not self.keyboard_enabled:
            return False
        if self.teleop is not None:
            return True
        self.teleop = PlateKeyboardTeleop(
            v_lin=self._teleop_v_lin, w_ang=self._teleop_w_ang)
        print(PlateKeyboardTeleop.help_text())
        print(f"  teleop v_lin={self._teleop_v_lin:.3f}m/s  "
              f"w_ang={self._teleop_w_ang:.3f}rad/s")
        return True

    def disable_keyboard_teleop(self):
        if self.teleop is not None:
            self.teleop.stop()
            self.teleop = None

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

        # TCP site：法兰盘中心（scene 中 tcp_L / tcp_R）
        self.tcp_left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_L")
        self.tcp_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_R")
        if self.tcp_left_id < 0 or self.tcp_right_id < 0:
            raise RuntimeError("缺少 TCP site：需要 scene 中定义 tcp_L / tcp_R（法兰中心）")
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
        print(f"[Init] 执行器模式: motor 力矩控制 (τ = g(q) + PD；关节 armature=折算惯量)")

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

    def get_tcp_world_pose(self, tcp_site_id):
        """获取法兰 TCP site 在世界坐标系中的位置和朝向"""
        pos = self.data.site_xpos[tcp_site_id].copy()
        xmat = self.data.site_xmat[tcp_site_id].reshape(3, 3).copy()
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

    def plate_body_twist_to_world(self, v_body, w_body, R_p=None):
        """板体坐标系旋量 → 世界系：^{w}v=R^{p}v，^{w}ω=R^{p}ω。"""
        if R_p is None:
            R_p = self.data.xmat[self.plate_id].reshape(3, 3)
        v_world = R_p @ np.asarray(v_body, dtype=float)
        w_world = R_p @ np.asarray(w_body, dtype=float)
        return v_world, w_world

    def plate_twist_to_tcp_velocity(self, v_plate_world, w_plate_world, tcp_site_id,
                                    grasp_off=None, plate_R=None):
        """
        世界系平板旋量 → 世界系 TCP 旋量（刚体速度公式）。

          v_tcp = v_plate + ω_plate × r
          ω_tcp = ω_plate
          r = plate_R @ grasp_off（默认用真实板姿态）
        """
        if grasp_off is not None:
            if plate_R is None:
                plate_R = self.data.xmat[self.plate_id].reshape(3, 3)
            r_tcp = plate_R @ grasp_off
        else:
            plate_pos = self.data.xpos[self.plate_id]
            r_tcp = self.data.site_xpos[tcp_site_id] - plate_pos
        v_tcp = v_plate_world + np.cross(w_plate_world, r_tcp)
        return np.concatenate([v_tcp, np.asarray(w_plate_world, dtype=float)])

    # ==================================================================
    # 第3层：雅可比矩阵计算
    # ==================================================================

    def compute_arm_jacobian(self, tcp_site_id, joint_ids):
        """
        计算单臂法兰 TCP site 的几何雅可比矩阵 J ∈ R^(6×7)

        使用 mj_jacSite 计算世界坐标系下的平移和旋转雅可比。

        Args:
            tcp_site_id: TCP site ID（法兰中心）
            joint_ids:   关节 ID 列表

        Returns:
            J: 6×7 雅可比矩阵 (3行平移 + 3行旋转)
        """
        n_joints = len(joint_ids)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, tcp_site_id)

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

        for act_id, side in ((ACT_LEFT_FINGER, 'L'), (ACT_RIGHT_FINGER, 'R')):
            dof = int(self.act_dofadr[act_id])
            jid = int(self.act_jnt_ids[act_id])
            qadr = model.jnt_qposadr[jid]
            e = self.finger_grip - data.qpos[qadr]
            de = 0.0 - data.qvel[dof]
            g_f = float(g[dof])
            bias = float(self.finger_close_bias)
            tau_raw = (g_f + self.kp_finger * e + self.kd_finger * de + bias)
            lo, hi = model.actuator_ctrlrange[act_id]
            tau_cmd = float(np.clip(tau_raw, lo, hi))
            data.ctrl[act_id] = tau_cmd
            self._finger_dbg[f'tau_{side}'] = tau_cmd
            self._finger_dbg[f'tau_raw_{side}'] = float(tau_raw)
            self._finger_dbg[f'e_{side}'] = float(e)
            self._finger_dbg[f'g_{side}'] = g_f
            self._finger_dbg[f'bias_{side}'] = bias
            self._finger_dbg[f'clipped_{side}'] = bool(tau_raw < lo or tau_raw > hi)

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
                model.geom_friction[gid] = [5.0, 0.1, 0.01]
                model.geom_solref[gid] = [0.003, 1.0]
                model.geom_condim[gid] = 4
        print(f"[Collision] 关闭 {n_off} 个机器人网格；保留手爪↔板接触")

    def _prepare_hold_scene(self):
        """板 Rz90 + mocap，手指夹紧，记录夹持相对位姿"""
        model, data = self.model, self.data

        self.scene_free_qpos = {}
        for name in ("plate", "cup", "water_mass", "ball"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            jid = model.body_jntadr[bid]
            qadr = int(model.jnt_qposadr[jid])
            self.scene_free_qpos[name] = (qadr, data.qpos[qadr:qadr + 7].copy())

        self._set_plate_mocap_pose(self._rpy_to_R(self.init_rpy))
        if "plate" in self.scene_free_qpos:
            qadr, q7 = self.scene_free_qpos["plate"]
            q7 = q7.copy()
            q7[:3] = self.plate_desired_pos
            q7[3:] = self._R_to_quat_wxyz(self._rpy_to_R(self.init_rpy))
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
        p_L = data.site_xpos[self.tcp_left_id].copy()
        p_R = data.site_xpos[self.tcp_right_id].copy()
        R_L = data.site_xmat[self.tcp_left_id].reshape(3, 3).copy()
        R_R = data.site_xmat[self.tcp_right_id].reshape(3, 3).copy()
        self.grasp_off_L = R_p.T @ (p_L - p_p)
        self.grasp_off_R = R_p.T @ (p_R - p_p)
        self.grasp_R_L = R_p.T @ R_L
        self.grasp_R_R = R_p.T @ R_R
        # 世界系末端锁定目标（HOLD 用）
        self.tcp_hold_pos_L = p_L.copy()
        self.tcp_hold_pos_R = p_R.copy()
        self.tcp_hold_R_L = R_L.copy()
        self.tcp_hold_R_R = R_R.copy()
        print(f"[Hold] 板 Rz90°；手指夹紧；finger-plate 接触对={n_fp}")
        print(f"[Grasp] off_L={np.round(self.grasp_off_L, 3)}  "
              f"off_R={np.round(self.grasp_off_R, 3)}")
        print(f"[TCP-Hold] L={np.round(self.tcp_hold_pos_L, 3)}  "
              f"R={np.round(self.tcp_hold_pos_R, 3)}")
        print(f"[Finger] q_des={self.finger_grip*1000:.1f}mm  "
              f"Kp={self.kp_finger:.0f} Kd={self.kd_finger:.0f}  "
              f"bias={self.finger_close_bias:+.1f}N(正=合拢)  "
              f"joint1 range=[0,23]mm  (slide, ctrl=力N)")
        self._print_finger_debug('INIT-F')

    def _ik_q_des(self, tcp_site_id, joint_ids, joint_ranges, p_des, R_des,
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
            p = data.site_xpos[tcp_site_id]
            R = data.site_xmat[tcp_site_id].reshape(3, 3)
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
            J = self.compute_arm_jacobian(tcp_site_id, joint_ids)
            dq = self.solve_ik_dls(J, twist)
            dq = np.clip(dq, -0.2, 0.2)
            q = q + dq
            set_q(q)

        q_out = get_q()
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        mujoco.mj_forward(model, data)
        return q_out

    def _tcp_pose_servo(self, tcp_site_id, grasp_off, grasp_R_rel,
                        plate_pos=None, plate_R=None):
        """
        相对板的 TCP 位姿误差 → 速度旋量修正。
        plate_pos/R 默认用真实板；CARRY 应传入期望板，避免追着掉落的板飞。
        """
        data = self.data
        if plate_R is None:
            plate_R = data.xmat[self.plate_id].reshape(3, 3)
        if plate_pos is None:
            plate_pos = data.xpos[self.plate_id]
        p_des = plate_pos + plate_R @ grasp_off
        R_des = plate_R @ grasp_R_rel

        p = data.site_xpos[tcp_site_id]
        R = data.site_xmat[tcp_site_id].reshape(3, 3)
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

    def _finger_qpos_pair(self, side):
        """返回 (finger1_q, finger2_q, finger1_dq, jnt_range_f1)"""
        model, data = self.model, self.data
        prefix = 'left' if side == 'left' else 'right'
        out = []
        for suffix in ('finger1_joint', 'finger2_joint'):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{prefix}{suffix}')
            qadr = int(model.jnt_qposadr[jid])
            out.append(float(data.qpos[qadr]))
        jid1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{prefix}finger1_joint')
        dq = float(data.qvel[int(model.jnt_dofadr[jid1])])
        lo, hi = model.jnt_range[jid1]
        return out[0], out[1], dq, float(lo), float(hi)

    def _finger_plate_contact_force(self, side):
        """手指-板接触法向力之和 (N)，按左右手分别统计"""
        model, data = self.model, self.data
        prefix = 'left' if side == 'left' else 'right'
        finger_ids = set()
        for name in (f'{prefix}finger1_Link', f'{prefix}finger2_Link'):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                finger_ids.add(bid)
        force6 = np.zeros(6)
        total = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if not ((b1 == self.plate_id and b2 in finger_ids) or
                    (b2 == self.plate_id and b1 in finger_ids)):
                continue
            mujoco.mj_contactForce(model, data, i, force6)
            total += abs(float(force6[0]))
        return total

    def get_finger_debug(self):
        """汇总左右手指位置/力矩/接触，供打印与 CSV"""
        q_L1, q_L2, dq_L, lo_L, hi_L = self._finger_qpos_pair('left')
        q_R1, q_R2, dq_R, lo_R, hi_R = self._finger_qpos_pair('right')
        # pad 世界坐标（便于看是否贴板）
        pad_pos = {}
        for name in ('leftfinger1_Link', 'leftfinger2_Link',
                     'rightfinger1_Link', 'rightfinger2_Link'):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            pad_pos[name] = self.data.xpos[bid].copy() if bid >= 0 else np.zeros(3)
        plate = self.data.xpos[self.plate_id]
        return {
            'q_des': float(self.finger_grip),
            'kp': float(self.kp_finger), 'kd': float(self.kd_finger),
            'bias': float(self.finger_close_bias),
            'q_L1': q_L1, 'q_L2': q_L2, 'dq_L': dq_L, 'lo_L': lo_L, 'hi_L': hi_L,
            'q_R1': q_R1, 'q_R2': q_R2, 'dq_R': dq_R, 'lo_R': lo_R, 'hi_R': hi_R,
            'e_L': float(self._finger_dbg['e_L']),
            'e_R': float(self._finger_dbg['e_R']),
            'tau_L': float(self._finger_dbg['tau_L']),
            'tau_R': float(self._finger_dbg['tau_R']),
            'tau_raw_L': float(self._finger_dbg['tau_raw_L']),
            'tau_raw_R': float(self._finger_dbg['tau_raw_R']),
            'g_L': float(self._finger_dbg['g_L']),
            'g_R': float(self._finger_dbg['g_R']),
            'clipped_L': bool(self._finger_dbg['clipped_L']),
            'clipped_R': bool(self._finger_dbg['clipped_R']),
            'cf_L': self._finger_plate_contact_force('left'),
            'cf_R': self._finger_plate_contact_force('right'),
            'n_contact': self._count_finger_plate_contacts(),
            'pad_pos': pad_pos,
            'plate': plate.copy(),
            'plate_z': float(plate[2]),
        }

    def _print_finger_debug(self, tag='FINGER'):
        """周期打印手指力矩与位置，便于排查夹不住"""
        d = self.get_finger_debug()
        gap_L = abs(d['q_L1'] - d['q_L2'])  # 因镜像，约 2*|q1|
        gap_R = abs(d['q_R1'] - d['q_R2'])
        clip_L = 'CLIP' if d['clipped_L'] else 'ok'
        clip_R = 'CLIP' if d['clipped_R'] else 'ok'
        print(
            f"[{tag} t={self.data.time:5.2f}s] "
            f"q_des={d['q_des']*1000:.1f}mm  "
            f"Kp={d['kp']:.0f} Kd={d['kd']:.0f} bias={d['bias']:+.1f}N | "
            f"ncon_grip={d['n_contact']} plate_z={d['plate_z']:.3f}"
        )
        print(
            f"  L: q1={d['q_L1']*1000:+6.1f}mm q2={d['q_L2']*1000:+6.1f}mm "
            f"(range[{d['lo_L']*1000:.0f},{d['hi_L']*1000:.0f}]mm) "
            f"e={d['e_L']*1000:+6.1f}mm dq={d['dq_L']:+.3f} "
            f"τ={d['tau_L']:+7.2f}N (raw={d['tau_raw_L']:+.2f} g={d['g_L']:+.2f} {clip_L}) "
            f"cf={d['cf_L']:.1f}N gap≈{gap_L*1000:.1f}mm"
        )
        print(
            f"  R: q1={d['q_R1']*1000:+6.1f}mm q2={d['q_R2']*1000:+6.1f}mm "
            f"(range[{d['lo_R']*1000:.0f},{d['hi_R']*1000:.0f}]mm) "
            f"e={d['e_R']*1000:+6.1f}mm dq={d['dq_R']:+.3f} "
            f"τ={d['tau_R']:+7.2f}N (raw={d['tau_raw_R']:+.2f} g={d['g_R']:+.2f} {clip_R}) "
            f"cf={d['cf_R']:.1f}N gap≈{gap_R*1000:.1f}mm"
        )
        # 指尖相对板中心（世界系）
        p = d['plate']
        for name, key in (('L1', 'leftfinger1_Link'), ('L2', 'leftfinger2_Link'),
                          ('R1', 'rightfinger1_Link'), ('R2', 'rightfinger2_Link')):
            x = d['pad_pos'][key]
            print(f"  pad_{name}=({x[0]:.3f},{x[1]:.3f},{x[2]:.3f}) "
                  f"Δplate=({x[0]-p[0]:+.3f},{x[1]-p[1]:+.3f},{x[2]-p[2]:+.3f})")

    def _set_plate_mocap_pose(self, R=None):
        """设置平板 mocap 位姿：与 target_pose 期望姿态一致。"""
        if R is None:
            R = self._desired_plate_rotation()
        self.data.mocap_pos[self.mocap_idx] = self.plate_desired_pos
        self.data.mocap_quat[self.mocap_idx] = self._R_to_quat_wxyz(R)

    # ==================================================================
    # 平板 Mocap 更新
    # ==================================================================

    def update_plate_mocap(self, yaw=None, R=None):
        """
        更新平板 mocap 到当前期望板位姿。
        R 默认取当前插值后的期望姿态；yaw 仅兼容旧调用。
        """
        if R is None:
            if yaw is not None:
                R = self._rpy_to_R([0.0, 0.0, float(yaw)])
            else:
                R = self._desired_plate_rotation()
        self._set_plate_mocap_pose(R)

    def _carry_alpha(self):
        """settle→move 插值系数 α∈[0,1]。"""
        t = float(self.data.time)
        if t <= self.carry_settle_s:
            return 0.0
        t_move = t - self.carry_settle_s
        if self.carry_move_s <= 1e-9 or t_move >= self.carry_move_s:
            return 1.0
        return t_move / self.carry_move_s

    def _desired_plate_rotation(self):
        """
        当前期望板姿态：init→target 四元数 slerp 后转 R。
        HOLD / settle 时为初始姿态。
        """
        return self._quat_to_R(self.plate_desired_quat)

    def _rot_error_vec(self, R_cur, R_des):
        R_err = R_des @ R_cur.T
        return 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])

    def _update_plate_target_pose(self):
        """
        更新期望板位姿（位置 + 姿态）。

        settle：保持 init；move：
          p = p0 + α · (R_target @ delta_body)
          q = slerp(q_init, q_target, α)
        若启用键盘遥操：在轨迹目标上叠加板体系 6D 增量。
        """
        if self.plate_start_pos is None:
            self.plate_start_pos = self.plate_desired_pos.copy()
            self._plate_des_prev = self.plate_desired_pos.copy()
            self.plate_desired_quat = self.init_quat.copy()
            self._kb_pos_off[:] = 0.0
            self._kb_R_off[:] = np.eye(3)

        alpha = self._carry_alpha()
        R_end = self._quat_to_R(self.target_quat)
        delta_w = R_end @ self.target_delta_body
        traj_p = self.plate_start_pos + alpha * delta_w
        traj_q = self._quat_slerp(self.init_quat, self.target_quat, alpha)
        R_traj = self._quat_to_R(traj_q)

        if self.teleop is not None:
            v_b, w_b = self.teleop.body_twist()
            R_des = R_traj @ self._kb_R_off
            self._kb_pos_off += R_des @ v_b * self.dt
            dR = self._exp_so3(w_b * self.dt)
            self._kb_R_off = self._kb_R_off @ dR
            # 数值正交化
            u, _, vh = np.linalg.svd(self._kb_R_off)
            self._kb_R_off = u @ vh
            if np.linalg.det(self._kb_R_off) < 0:
                u[:, -1] *= -1.0
                self._kb_R_off = u @ vh

        self.plate_desired_pos[:] = traj_p + self._kb_pos_off
        self.plate_desired_quat[:] = self._R_to_quat_wxyz(
            R_traj @ self._kb_R_off)

    def _plate_twist_des_world(self):
        """
        target_pose 误差（板体系）→ 期望板速度 → 世界系旋量。

          更新 p_des 后：
          e_p^b = R_pᵀ (p_des - p)
          ^{p}v = Kp_pos e_p^b + R_pᵀ ṗ_des   (目标运动前馈)
          ^{p}ω = Kp_rot e_R^b
        """
        self._update_plate_target_pose()
        R_p = self.data.xmat[self.plate_id].reshape(3, 3)
        p_p = self.data.xpos[self.plate_id]
        R_des = self._desired_plate_rotation()
        p_des = self.plate_desired_pos

        if self._plate_des_prev is None:
            self._plate_des_prev = p_des.copy()
        p_dot_des = (p_des - self._plate_des_prev) / max(self.dt, 1e-9)
        self._plate_des_prev = p_des.copy()
        v_ff_body = R_p.T @ p_dot_des

        e_pos_body = R_p.T @ (p_des - p_p)
        e_rot_world = self._rot_error_vec(R_p, R_des)
        e_rot_body = R_p.T @ e_rot_world

        v_body = self.kp_plate_pos * e_pos_body + v_ff_body
        w_body = self.kp_plate_rot * e_rot_body

        vn = float(np.linalg.norm(v_body))
        if vn > self.v_plate_body_limit:
            v_body *= self.v_plate_body_limit / vn
        wn = float(np.linalg.norm(w_body))
        if wn > self.w_plate_body_limit:
            w_body *= self.w_plate_body_limit / wn

        return self.plate_body_twist_to_world(v_body, w_body, R_p)

    # ==================================================================
    # 主控制循环（单步）
    # ==================================================================

    def _tcp_world_pose_servo(self, tcp_site_id, p_des, R_des):
        """世界系 TCP 位姿误差 → 速度旋量（HOLD 末端锁定）。"""
        data = self.data
        p = data.site_xpos[tcp_site_id]
        R = data.site_xmat[tcp_site_id].reshape(3, 3)
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

    def _arm_tcp_track_step(self, v_plate_world, w_plate_world):
        """
        CARRY：世界系板旋量 → 左右 TCP + 相对**期望板**位姿伺服
        → DLS(J_world) → 更新 q_des。
        """
        R_des = self._desired_plate_rotation()
        p_des = self.plate_desired_pos
        epos_L = float(np.linalg.norm(
            p_des + R_des @ self.grasp_off_L - self.data.site_xpos[self.tcp_left_id]))
        epos_R = float(np.linalg.norm(
            p_des + R_des @ self.grasp_off_R - self.data.site_xpos[self.tcp_right_id]))

        # 刚体速度映射用期望板姿态下的杠杆臂（与期望夹持一致）
        twist_left = self.plate_twist_to_tcp_velocity(
            v_plate_world, w_plate_world, self.tcp_left_id,
            self.grasp_off_L, plate_R=R_des)
        twist_right = self.plate_twist_to_tcp_velocity(
            v_plate_world, w_plate_world, self.tcp_right_id,
            self.grasp_off_R, plate_R=R_des)
        servo_L, _, _ = self._tcp_pose_servo(
            self.tcp_left_id, self.grasp_off_L, self.grasp_R_L,
            plate_pos=p_des, plate_R=R_des)
        servo_R, _, _ = self._tcp_pose_servo(
            self.tcp_right_id, self.grasp_off_R, self.grasp_R_R,
            plate_pos=p_des, plate_R=R_des)
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
        return dq_left, dq_right, epos_L, epos_R

    def _arm_tcp_hold_world_step(self):
        """HOLD：世界系锁定初始 TCP 位姿 → DLS → 更新 q_des。"""
        twist_left, epos_L, _ = self._tcp_world_pose_servo(
            self.tcp_left_id, self.tcp_hold_pos_L, self.tcp_hold_R_L)
        twist_right, epos_R, _ = self._tcp_world_pose_servo(
            self.tcp_right_id, self.tcp_hold_pos_R, self.tcp_hold_R_R)

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
        return dq_left, dq_right, epos_L, epos_R

    def control_step(self):
        """每步仿真：HOLD 世界系 TCP 锁定，或 CARRY 跟板平移"""

        if self.hold_mode:
            dq_left, dq_right, epos_L, epos_R = self._arm_tcp_hold_world_step()
            self.write_torque_ctrl(
                self.q_des_left, dq_left, self.q_des_right, dq_right)
            self.update_plate_mocap()
            self._last_dq_left[:] = dq_left
            self._last_dq_right[:] = dq_right
            if self.csv_logger is not None:
                self.csv_logger.maybe_log(
                    self, dq_left, dq_right, epos_L, epos_R)

            self.step_count += 1
            if self.step_count % self.log_interval == 0:
                roll, pitch = self.get_chassis_orientation()
                plate = self.data.xpos[self.plate_id]
                R_p = self.data.xmat[self.plate_id].reshape(3, 3)
                level_err = np.degrees(np.arccos(np.clip(R_p[2, 2], -1, 1)))
                print(f"[HOLD t={self.data.time:5.2f}s] "
                      f"roll={np.degrees(roll):+5.2f}° "
                      f"pitch={np.degrees(pitch):+5.2f}° "
                      f"| level={level_err:5.2f}° grip={self._count_finger_plate_contacts()} "
                      f"plate_z={plate[2]:.3f} "
                      f"eTCP=({epos_L*1000:.1f}/{epos_R*1000:.1f}mm)")
                self._print_finger_debug('HOLD-F')
            return

        # === CARRY：期望板误差(板系)→板速→TCP→DLS ===
        v_plate_w, w_plate_w = self._plate_twist_des_world()
        dq_left, dq_right, epos_L, epos_R = self._arm_tcp_track_step(
            v_plate_w, w_plate_w)

        self.write_torque_ctrl(
            self.q_des_left, dq_left, self.q_des_right, dq_right)
        self.update_plate_mocap()

        if self.csv_logger is not None:
            self.csv_logger.maybe_log(self, dq_left, dq_right, epos_L, epos_R)

        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            roll, pitch = self.get_chassis_orientation()
            plate = self.data.xpos[self.plate_id]
            R_p = self.data.xmat[self.plate_id].reshape(3, 3)
            level_err = np.degrees(np.arccos(np.clip(R_p[2, 2], -1, 1)))
            p0 = self.plate_start_pos if self.plate_start_pos is not None \
                else self.plate_desired_pos
            p_des = self.plate_desired_pos
            delta_w = self._rpy_to_R(self.target_rpy) @ self.target_delta_body
            dp_des = p_des - p0
            e_plate = p_des - plate
            alpha = self._carry_alpha()
            if alpha <= 0.0:
                phase = 'settle'
            elif alpha < 1.0:
                phase = 'move'
            else:
                phase = 'done'
            t = float(self.data.time)
            print(f"[CARRY t={t:5.2f}s {phase} α={alpha:.2f}] "
                  f"roll={np.degrees(roll):+5.1f}° pitch={np.degrees(pitch):+5.1f}° "
                  f"| level={level_err:5.2f}° grip={self._count_finger_plate_contacts()}")
            print(f"  init_rpy={np.round(self.init_rpy, 4).tolist()}  "
                  f"target_rpy={np.round(self.target_rpy, 4).tolist()}  "
                  f"rpy_des={np.round(self._R_to_rpy(self._desired_plate_rotation()), 4).tolist()}")
            print(f"  q_des={np.round(self.plate_desired_quat, 4).tolist()}")
            print(f"  delta_body={np.round(self.target_delta_body, 4).tolist()}  "
                  f"delta_world={np.round(delta_w, 4).tolist()}")
            print(f"  p0=({p0[0]:.4f},{p0[1]:.4f},{p0[2]:.4f})  "
                  f"p_des=({p_des[0]:.4f},{p_des[1]:.4f},{p_des[2]:.4f})  "
                  f"Δp_des={np.round(dp_des, 4).tolist()}")
            print(f"  z0={p0[2]:.4f}  z_des={p_des[2]:.4f}  z_act={plate[2]:.4f}  "
                  f"Δz_des={(p_des[2]-p0[2])*1000:.1f}mm  "
                  f"Δz_act={(plate[2]-p0[2])*1000:.1f}mm  "
                  f"e_z={(p_des[2]-plate[2])*1000:.1f}mm")
            print(f"  p_act=({plate[0]:.4f},{plate[1]:.4f},{plate[2]:.4f})  "
                  f"e_plate={np.round(e_plate, 4).tolist()}  "
                  f"eTCP=({epos_L*1000:.1f}/{epos_R*1000:.1f}mm)")
            self._print_finger_debug('CARRY-F')
# =====================================================================
# 仿真运行器
# =====================================================================

class SimulationRunner:
    """MuJoCo 仿真运行封装"""

    def __init__(self, xml_path, cfg):
        self.xml_path = xml_path
        self.cfg = cfg
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
        edge_clearance = 0.008   # 手指中点在短边外侧，避免穿模

        # Joint7 期望姿态（水平短边夹持），再映射到法兰 TCP
        # 左臂 Joint7: X→+Z, Y→+Y, Z→-X
        R_j7_left = np.array([[0.0, 0.0, -1.0],
                              [0.0, 1.0,  0.0],
                              [1.0, 0.0,  0.0]])
        # 法兰相对 Joint7 的固定旋转（与 flange geom / tcp site quat 一致）
        qw, qx, qy, qz = 0.000563312, 0.000562864, 0.707388, -0.706825
        R_j7_flange = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])
        R_tcp_left = R_j7_left @ R_j7_flange
        # 手指中点在法兰系约 +Z 0.1305m（由 URDF 几何推得）
        finger_in_tcp = np.array([0.0, 0.0, 0.1305])

        tcp_left_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_L")
        tcp_right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_R")
        if tcp_left_id < 0 or tcp_right_id < 0:
            raise RuntimeError("scene 缺少 tcp_L / tcp_R site（法兰中心）")
        left_joint_ids  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                            f"left_arm_joint_{i}") for i in range(1, 8)]
        right_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                            f"right_arm_joint_{i}") for i in range(1, 8)]

        # 已标定近水平种子（倾角约 0.4°）
        q_left_seed = np.array([0.9691, -0.72, 0.144, -0.88, 1.5063, -0.4957, 1.5708])

        plate_nominal = np.array([0.35, 0.0, 0.98])
        finger_left = plate_nominal + np.array(
            [0.0, +(plate_half_long + edge_clearance), 0.0])
        tcp_left_des = finger_left - R_tcp_left @ finger_in_tcp

        print("[Load] 求解左臂 6D IK (法兰 TCP / 水平短边外侧夹持)...")
        q_left_init = self._ik_solve_arm_6d(
            tcp_left_id, left_joint_ids, tcp_left_des, R_tcp_left,
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
        tcp_left_pos  = self.data.site_xpos[tcp_left_id].copy()
        tcp_right_pos = self.data.site_xpos[tcp_right_id].copy()
        R_l = self.data.site_xmat[tcp_left_id].reshape(3, 3)
        # 法兰接近轴(+Z)的世界 Z 分量：0=水平接近
        tilt_deg = float(np.degrees(np.arcsin(np.clip(R_l[2, 2], -1, 1))))

        plate_center_x = 0.5 * (mid_l[0] + mid_r[0])
        plate_center_y = 0.0
        plate_surface_z = 0.5 * (mid_l[2] + mid_r[2])
        finger_span_y = 0.5 * (abs(mid_l[1]) + abs(mid_r[1]))

        print(f"\n[Load] IK 结果:")
        print(f"  左法兰TCP: ({tcp_left_pos[0]:.3f}, {tcp_left_pos[1]:.3f}, {tcp_left_pos[2]:.3f})")
        print(f"  右法兰TCP: ({tcp_right_pos[0]:.3f}, {tcp_right_pos[1]:.3f}, {tcp_right_pos[2]:.3f})")
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

        # 手指初始开合：由板厚决定（MuJoCo box size 为半尺寸）
        # finger2 = -finger1，两侧各开半厚 → 开口约等于全厚
        plate_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, 'plate_geom')
        plate_half_thick = float(self.model.geom_size[plate_gid][2])
        plate_thick = 2.0 * plate_half_thick
        jid_f1 = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 'leftfinger1_joint')
        lo_f, hi_f = self.model.jnt_range[jid_f1]
        finger_q0 = float(np.clip(plate_half_thick, max(lo_f, 0.0), hi_f))
        for fname, val in (('leftfinger1_joint', finger_q0),
                           ('leftfinger2_joint', -finger_q0),
                           ('rightfinger1_joint', finger_q0),
                           ('rightfinger2_joint', -finger_q0)):
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
        print(f"  板厚={plate_thick*1000:.1f}mm → 手指初始 qpos="
              f"±{finger_q0*1000:.1f}mm")
        print(f"  夹持模式: 左右短边水平外侧夹持（防穿模）")
        print(f"  执行器: motor 力矩控制 (τ = g(q) + PD)")

        self.controller = DualArmPlateController(self.model, self.data, self.cfg)
        self.controller.plate_desired_pos = np.array(
            [plate_center_x, plate_center_y, plate_surface_z])
        self.controller.finger_grip = finger_q0  # 与初始 qpos 一致，偏置再往里夹
        self.controller.lock_hold_targets()

        mode = "HOLD" if self.controller.hold_mode else "CARRY"
        print(f"  控制模式: {mode}")

        csv_path = getattr(self.cfg.sim, 'csv', None)
        if csv_path:
            if not os.path.isabs(csv_path):
                csv_path = os.path.join(os.path.dirname(self.xml_path), csv_path)
            every = max(1, int(getattr(self.cfg.sim, 'csv_every', 5)))
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
        duration = float(self.cfg.sim.duration)
        print(f"[Run] 无GUI模式 [{mode}]，仿真时长 {duration}s")
        total_steps = int(duration / self.model.opt.timestep)

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
              f"RTF = {duration / max(elapsed, 1e-6):.2f}")
        print(f"[Stability] max‖Δq_L‖={max_err_l:.4f}  max‖Δq_R‖={max_err_r:.4f}  "
              f"max|pitch|={np.degrees(max_pitch):.2f}°")

        if self.controller.csv_logger is not None:
            self.controller.csv_logger.close()
            issues = summarize_csv(self.controller.csv_logger.path)
        else:
            issues = []

        self._print_state()
        if self.controller.hold_mode:
            # 末端锁定后关节会跟板微调，不以初始 Δq 判失败
            ok = max_pitch < np.radians(8.0)
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
        print(f"[Run] GUI可视化模式，按 ESC / 关窗口退出")
        self.controller.enable_keyboard_teleop()

        try:
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
        finally:
            self.controller.disable_keyboard_teleop()

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

    def _ik_solve_arm_6d(self, tcp_site_id, joint_ids, target_pos, target_R,
                          q0=None, n_iter=1500, w_pos=1.0, w_rot=0.35,
                          tol_pos=0.008, tol_rot=0.08):
        """
        6D 数值 IK：位置 + 姿态（DLS），末端为法兰 TCP site。

        手指中点约在法兰 local +Z 0.1305m；开合主要沿法兰 local X。
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
            p = self.data.site_xpos[tcp_site_id].copy()
            R = self.data.site_xmat[tcp_site_id].reshape(3, 3).copy()

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
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, tcp_site_id)

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
            p = self.data.site_xpos[tcp_site_id]
            R = self.data.site_xmat[tcp_site_id].reshape(3, 3)
            pos_n = np.linalg.norm(target_pos - p)
            rot_n = np.linalg.norm(rot_error(R, target_R))
            print(f"  [IK6] ⚠ 未完全收敛: pos={pos_n*1000:.1f}mm, rot={rot_n:.3f}rad "
                  f"(已取最优)")

        return get_q()

    def run(self):
        """根据配置选择运行模式"""
        self.load()

        if bool(self.cfg.sim.step):
            self.run_step_mode()
        elif bool(self.cfg.sim.gui):
            self.run_gui()
        else:
            self.run_headless()

# =====================================================================
# 入口
# =====================================================================

def _dict_to_ns(obj):
    """递归把 dict 转成 SimpleNamespace，便于 cfg.sim.gui 访问。"""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_ns(v) for v in obj]
    return obj


def load_config(path):
    """从 YAML 加载配置。"""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件格式错误（需要映射）: {path}")
    for key in ('sim', 'carry', 'control'):
        if key not in raw:
            raise KeyError(f"配置缺少 '{key}' 段: {path}")
    cfg = _dict_to_ns(raw)
    cfg._config_path = path
    return cfg


def parse_args():
    """仅解析配置文件路径；其余参数全部在 YAML 中。"""
    parser = argparse.ArgumentParser(
        description="双臂协同稳载仿真：参数由 YAML 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dual_arm_controller.py
  python dual_arm_controller.py --config config.yaml
        """)
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'config.yaml')
    parser.add_argument('--config', type=str, default=default_cfg,
                        help='YAML 配置文件路径')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    xml_path = cfg.sim.xml
    if not os.path.isabs(xml_path):
        candidates = [
            os.path.join(os.path.dirname(__file__), xml_path),
            os.path.join(os.path.dirname(cfg._config_path), xml_path),
            xml_path,
        ]
        for cand in candidates:
            if os.path.exists(cand):
                xml_path = os.path.abspath(cand)
                break
        else:
            print(f"[Error] 找不到 MJCF 文件: {cfg.sim.xml}")
            print(f"  搜索路径: {candidates}")
            sys.exit(1)

    mode = str(cfg.sim.mode).strip().upper()
    print("=" * 60)
    print("  轮式移动双臂机器人协同稳载仿真系统")
    print("  BalanceDual-Arm v1.3")
    print("=" * 60)
    print(f"  Config:  {cfg._config_path}")
    print(f"  MJCF:    {xml_path}")
    print(f"  GUI:     {bool(cfg.sim.gui)}")
    print(f"  Keyboard:{bool(getattr(cfg.sim, 'keyboard', False))}")
    print(f"  Mode:    {mode}")
    print(f"  Duration:{float(cfg.sim.duration)}s")
    print(f"  CSV:     {cfg.sim.csv}")
    if mode == 'CARRY':
        ip = getattr(cfg.carry, 'init_pose', None)
        tp = getattr(cfg.carry, 'target_pose', None)
        init_rpy = list(ip.rpy) if ip is not None and hasattr(ip, 'rpy') \
            else [0.0, 0.0, float(getattr(cfg.carry, 'target_yaw', np.pi / 2))]
        if tp is not None:
            delta = list(tp.delta_body)
            target_rpy = list(tp.rpy) if hasattr(tp, 'rpy') else list(init_rpy)
        else:
            delta = list(getattr(cfg.carry, 'target_delta_body', [0, 0, 0]))
            target_rpy = [0.0, 0.0, float(getattr(cfg.carry, 'target_yaw', np.pi / 2))]
        print(f"  Carry:   settle={cfg.carry.settle_s}s move={cfg.carry.move_s}s")
        print(f"  init_pose.rpy={init_rpy}")
        print(f"  target_pose.delta_body={delta} rpy={target_rpy}")
    print("=" * 60)

    runner = SimulationRunner(xml_path, cfg)
    runner.run()


if __name__ == '__main__':
    main()
