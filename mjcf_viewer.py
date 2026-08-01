#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KITT1_5V3 双足双臂机器人 MJCF 模型查看器
支持加载 MJCF 文件并提供交互式关节控制

机器人结构:
- 腿部: 8 个铰链关节 (ZQ, ZQL, YQ, YQL, ZH, ZHL, YH, YHL)
- 躯干: 4 个铰链关节 (trunk_joint_1~4)
- 左臂: 7 个铰链关节 (left_arm_joint_1~7)
- 右臂: 7 个铰链关节 (right_arm_joint_1~7)
- 头部: 2 个铰链关节 (head_joint_1~2)
- 手指: 4 个滑动关节 (leftfinger1~2, rightfinger1~2)
- 执行器类型: position (位置伺服，ctrl 为目标角度)

依赖:
- mujoco
- numpy
"""

import mujoco
import mujoco.viewer
import numpy as np
import argparse
import os
import sys


class MJCFViewer:
    """MJCF模型查看器类"""

    def __init__(self, mjcf_path):
        """
        初始化查看器

        Args:
            mjcf_path (str): MJCF文件路径
        """
        self.mjcf_path = mjcf_path
        self.model = None
        self.data = None
        self.viewer = None
        self.running = False

        # 关节控制参数
        self.joint_targets = {}
        self.joint_velocities = {}
        self.control_mode = 'position'  # 'position' 或 'velocity'

        # 执行器映射：joint_name → actuator_index
        self.actuator_map = {}

        # 场景物体（带 freejoint 的非机器人体）
        self.scene_bodies = []

        # 是否有 keyframe（场景文件常有）
        self.has_keyframe = False

        print(f"🤖 初始化 KITT1_5V3 双足双臂机器人 MJCF 查看器")
        print(f"📁 文件路径: {mjcf_path}")

    def load_model(self):
        """加载MJCF模型"""
        try:
            if not os.path.exists(self.mjcf_path):
                raise FileNotFoundError(f"MJCF文件不存在: {self.mjcf_path}")

            print("🔄 正在加载MJCF模型...")
            self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
            self.data = mujoco.MjData(self.model)

            print("✅ 模型加载成功!")

            # 初始化关节分组（必须在 print_model_info 之前）
            self._init_joint_groups()

            # 构建 actuator 映射 + 检测场景物体 + 检测 keyframe
            self._init_actuator_map()
            self._init_scene_info()

            self.print_model_info()
            self.print_joint_index_mapping()

            # 初始化关节目标位置
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name:
                    self.joint_targets[joint_name] = 0.0
                    self.joint_velocities[joint_name] = 0.0

            # 如果有 keyframe，用 keyframe 初始化 data
            if self.has_keyframe:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
                print("✅ 已从 keyframe[0] 初始化初始位姿")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

    def _init_joint_groups(self):
        """按关节名称前缀自动分组"""
        self.joint_groups = {
            "腿部": [],      # ZQ, ZQL, YQ, YQL, ZH, ZHL, YH, YHL
            "躯干": [],      # trunk_joint_*
            "左臂": [],      # left_arm_joint_*
            "右臂": [],      # right_arm_joint_*
            "头部": [],      # head_joint_*
            "手指": [],      # leftfinger*, rightfinger*
            "其他": [],
        }

        leg_prefixes = ("ZQ", "ZQL", "YQ", "YQL", "ZH", "ZHL", "YH", "YHL")

        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not joint_name:
                continue

            if joint_name.startswith("trunk_joint_"):
                self.joint_groups["躯干"].append((i, joint_name))
            elif joint_name.startswith("left_arm_joint_"):
                self.joint_groups["左臂"].append((i, joint_name))
            elif joint_name.startswith("right_arm_joint_"):
                self.joint_groups["右臂"].append((i, joint_name))
            elif joint_name.startswith("head_joint_"):
                self.joint_groups["头部"].append((i, joint_name))
            elif joint_name.startswith("leftfinger") or joint_name.startswith("rightfinger"):
                self.joint_groups["手指"].append((i, joint_name))
            elif joint_name in leg_prefixes:
                self.joint_groups["腿部"].append((i, joint_name))
            else:
                self.joint_groups["其他"].append((i, joint_name))

    def _init_actuator_map(self):
        """构建 joint_name → actuator_index 的映射"""
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name is None:
                continue
            # 尝试从执行器名推断关节名
            joint_id = self.model.actuator_trnid[i][0] if hasattr(self.model, 'actuator_trnid') else -1
            # 直接用 actuator 的 target joint name
            try:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else None
            except Exception:
                joint_name = None
            if joint_name:
                self.actuator_map[joint_name] = i

        if self.actuator_map:
            print(f"🔗 已构建 {len(self.actuator_map)} 个 actuator→joint 映射")

    def _init_scene_info(self):
        """检测场景物体（带 freejoint 的非机器人身体）和 keyframe"""
        # 检测 freejoint body（场景物体）
        robot_body_names = {
            "ZQ_Link", "ZQL_Link", "YQ_Link", "YQL_Link",
            "ZH_Link", "ZHL_Link", "YH_Link", "YHL_Link",
            "Trunk1", "Trunk2", "Trunk3", "Trunk4",
            "Joint1_L", "Joint2_L", "Joint3_L", "Joint4_L",
            "Joint5_L", "Joint6_L", "Joint7_L",
            "Joint1_R", "Joint2_R", "Joint3_R", "Joint4_R",
            "Joint5_R", "Joint6_R", "Joint7_R",
            "Head1", "Head2",
            "leftfinger1_Link", "leftfinger2_Link",
            "rightfinger1_Link", "rightfinger2_Link",
        }

        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name is None or body_name == "world":
                continue
            if body_name in robot_body_names:
                continue
            # 检查是否有 freejoint
            jnt_adr = self.model.body_jntadr[i]
            jnt_num = self.model.body_jntnum[i]
            has_free = False
            for j in range(jnt_num):
                if self.model.jnt_type[jnt_adr + j] == 0:  # freejoint
                    has_free = True
                    break
            if has_free:
                self.scene_bodies.append((i, body_name))

        # 检测 keyframe
        self.has_keyframe = self.model.nkey > 0
        if self.has_keyframe:
            print(f"🎬 检测到 {self.model.nkey} 个 keyframe")
        if self.scene_bodies:
            print(f"📦 检测到 {len(self.scene_bodies)} 个场景物体: "
                  f"{', '.join(name for _, name in self.scene_bodies)}")

    def print_model_info(self):
        """打印模型信息"""
        print("\n📊 模型信息:")
        print(f"  🔗 刚体数量: {self.model.nbody}")
        print(f"  🔄 关节数量: {self.model.njnt}")
        print(f"  ⚙️  自由度: {self.model.nv}")
        print(f"  🎮 执行器数量: {self.model.nu}")
        if getattr(self.model, "neq", 0) > 0:
            print(f"  🔗 等式约束 (equality): {self.model.neq}")
        if self.scene_bodies:
            print(f"  📦 场景物体: {len(self.scene_bodies)} 个")
        if hasattr(self.model, "nsensor") and self.model.nsensor > 0:
            print(f"  📡 传感器数量: {self.model.nsensor}")
        if self.has_keyframe:
            print(f"  🎬 Keyframe 数量: {self.model.nkey}")

        # 打印关节信息（按分组显示）
        if self.model.njnt > 0:
            # 关节类型映射
            type_map = {0: "自由", 1: "球形", 2: "滑动", 3: "铰链", 4: "螺旋"}

            print(f"\n🔄 关节列表 (共 {self.model.njnt} 个):")
            for group_name, joints in self.joint_groups.items():
                if not joints:
                    continue
                print(f"\n  📌 {group_name} ({len(joints)} 个):")
                for joint_idx, joint_name in joints:
                    joint_type = self.model.jnt_type[joint_idx]
                    joint_range = self.model.jnt_range[joint_idx]
                    type_name = type_map.get(joint_type, f"未知({joint_type})")

                    if joint_range[0] == joint_range[1]:
                        range_str = "无限制"
                    else:
                        range_str = f"[{joint_range[0]:.2f}, {joint_range[1]:.2f}]"

                    print(f"    {joint_idx + 1:2d}. {joint_name:<24} 类型: {type_name:<6} 范围: {range_str}")

        # 打印执行器信息
        if self.model.nu > 0:
            print(f"\n⚙️  执行器列表:")
            for i in range(self.model.nu):
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if actuator_name:
                    print(f"    {i + 1:2d}. {actuator_name}")

    def print_joint_index_mapping(self):
        """打印关节名称 -> MuJoCo 内部索引的完整对照表"""
        type_map = {0: "free", 1: "ball", 2: "slide", 3: "hinge", 4: "screw"}

        print("\n" + "=" * 72)
        print("  📋 关节名称 → qpos 索引 完整对照表")
        print("=" * 72)
        print(f"  {'索引':>5s} │ {'关节名称':<26s} │ {'类型':<8s} │ {'范围'}")
        print("  " + "─" * 5 + "─┼─" + "─" * 26 + "─┼─" + "─" * 8 + "─┼─" + "─" * 22)

        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not joint_name:
                continue
            jtype = type_map.get(self.model.jnt_type[i], f"?({self.model.jnt_type[i]})")
            jrange = self.model.jnt_range[i]
            if jrange[0] == jrange[1]:
                range_str = "unlimited"
            else:
                range_str = f"[{jrange[0]:.4f}, {jrange[1]:.4f}]"
            print(f"  {i:5d} │ {joint_name:<26s} │ {jtype:<8s} │ {range_str}")

        print("=" * 72)

        # qvel 索引（hinge/slide 与 qpos 一一对应）
        print(f"\n  💡 qpos[{0}:{self.model.njnt}] 和 qvel[{0}:{self.model.nv}] 均按此顺序")
        print(f"  💡 自由度为 {self.model.nv}，执行器数量为 {self.model.nu}")

    def save_joint_mapping(self, filepath=None):
        """将关节索引对照表保存为 JSON 文件"""
        import json

        if filepath is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, "joint_mapping.json")

        type_map = {0: "free", 1: "ball", 2: "slide", 3: "hinge", 4: "screw"}

        mapping = {
            "model": "KITT1_5V3_dual_arm",
            "total_joints": self.model.njnt,
            "total_actuators": self.model.nu,
            "nv": self.model.nv,
            "joints": []
        }

        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not joint_name:
                continue
            jrange = self.model.jnt_range[i]
            entry = {
                "qpos_index": i,
                "name": joint_name,
                "type": type_map.get(self.model.jnt_type[i], f"unknown_{self.model.jnt_type[i]}"),
                "range": [float(jrange[0]), float(jrange[1])],
            }
            mapping["joints"].append(entry)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        print(f"✅ 关节对照表已保存到: {filepath}")

    def start_viewer(self):
        """启动可视化查看器"""
        print("\n🚀 启动 KITT1_5V3 双足双臂机器人可视化查看器...")
        print("\n📝 控制说明:")
        print("  🖱️  鼠标左键拖拽: 旋转视角")
        print("  🖱️  鼠标右键拖拽: 平移视角")
        print("  🖱️  鼠标滚轮: 缩放")
        print("  ⌨️  空格键: 暂停/继续仿真")
        print("  ⌨️  Ctrl+R: 重置仿真")
        print("  ⌨️  Tab: 显示/隐藏帮助")
        print("  ⌨️  ESC: 退出程序")
        print("\n🎮 关节控制:")
        print("  通过 position 执行器 ctrl 控制关节，不直接写 qpos。")
        print("  命令格式: <关节名称> <角度>")
        print("  例如: LA1_act -0.5   或   left_arm_joint_1 -0.5")
        print("  输入 'list' 查看所有关节及分组")
        print("  输入 'index' 查看关节名称→索引对照表")
        print("  输入 'info' 查看当前仿真状态")
        print("  输入 'actors' 查看场景物体状态")
        print("  输入 'save' 保存对照表到 JSON 文件")
        print("  输入 'help' 查看所有命令")
        print("  输入 'reset' 复位全部关节")
        print("  输入 'quit' 退出程序")

        self.running = True

        # 启动控制线程
        # control_thread = threading.Thread(target=self.control_loop, daemon=True)
        # control_thread.start()

        # 启动查看器
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer

            while self.running and viewer.is_running():
                # 将 joint_targets 同步到对应的 position 执行器 ctrl
                for joint_name, target in self.joint_targets.items():
                    act_idx = self.actuator_map.get(joint_name)
                    if act_idx is not None and act_idx < self.model.nu:
                        self.data.qpos[:] =0

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

    def control_loop(self):
        """关节控制循环"""
        print(f"\n💡 关节控制已启动 (在当前终端输入命令)")
        self.print_control_help()

        while self.running:
            try:
                command = input("🎮 > ").strip()
                if not command:
                    continue

                if command.lower() == 'quit':
                    self.running = False
                    break
                elif command.lower() == 'help':
                    self.print_control_help()
                elif command.lower() == 'list':
                    self.list_joints()
                elif command.lower() == 'index':
                    self.print_joint_index_mapping()
                elif command.lower() == 'save':
                    self.save_joint_mapping()
                elif command.lower() == 'reset':
                    self.reset_joints()
                elif command.lower() == 'info':
                    self.print_current_state()
                elif command.lower() == 'actors':
                    self.list_scene_bodies()
                elif command.lower().startswith('mode'):
                    self.set_control_mode(command)
                else:
                    self.parse_joint_command(command)

            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
            except Exception as e:
                print(f"❌ 命令错误: {e}")

    def print_control_help(self):
        """打印控制帮助"""
        print("\n🎮 关节控制命令:")
        print("  <关节名> <值>     - 设置关节目标位置 (通过 ctrl)")
        print("  list             - 列出所有关节（按分组）+ 场景物体")
        print("  index            - 显示关节名称→索引对照表")
        print("  save             - 保存关节对照表到 JSON 文件")
        print("  reset            - 重置所有关节到零位（含 keyframe 复位）")
        print("  info             - 显示当前仿真状态")
        print("  actors           - 显示场景物体状态")
        print("  mode <position|velocity> - 切换控制模式")
        print("  help             - 显示此帮助")
        print("  quit             - 退出程序")
        print(f"\n当前控制模式: {self.control_mode}")
        if self.actuator_map:
            print(f"已关联执行器: {len(self.actuator_map)} 个")
        if self.scene_bodies:
            print(f"场景物体: {', '.join(name for _, name in self.scene_bodies)}")

    def list_joints(self):
        """列出所有关节（按分组显示）"""
        print(f"\n🔄 关节列表 (共 {self.model.njnt} 个):")
        for group_name, joints in self.joint_groups.items():
            if not joints:
                continue
            print(f"\n  📌 {group_name} ({len(joints)} 个):")
            for joint_idx, joint_name in joints:
                current_pos = self.data.qpos[joint_idx] if joint_idx < len(self.data.qpos) else 0.0
                current_vel = self.data.qvel[joint_idx] if joint_idx < len(self.data.qvel) else 0.0
                target = self.joint_targets.get(joint_name, 0.0)

                joint_range = self.model.jnt_range[joint_idx]
                if joint_range[0] == joint_range[1]:
                    range_str = "无限制"
                else:
                    range_str = f"[{joint_range[0]:.2f}, {joint_range[1]:.2f}]"

                # 显示 ctrl 值（如果有关联执行器）
                act_idx = self.actuator_map.get(joint_name)
                ctrl_val = self.data.ctrl[act_idx] if act_idx is not None and act_idx < self.model.nu else None
                ctrl_str = f" ctrl:{ctrl_val:8.3f}" if ctrl_val is not None else ""

                print(
                    f"  {joint_idx + 1:2d}. {joint_name:<24} 位置: {current_pos:8.3f} 速度: {current_vel:8.3f} 目标: {target:8.3f}{ctrl_str} 范围: {range_str}")

        # 显示场景物体（自由刚体）
        if self.scene_bodies:
            print(f"\n📦 场景物体 ({len(self.scene_bodies)} 个):")
            for body_idx, body_name in self.scene_bodies:
                jnt_adr = self.model.body_jntadr[body_idx]
                # freejoint 占 7 个 qpos (3 pos + 4 quat)
                pos = self.data.qpos[jnt_adr:jnt_adr + 3]
                quat = self.data.qpos[jnt_adr + 3:jnt_adr + 7]
                print(f"  {body_name:<20} pos: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]"
                      f"  quat: [{quat[0]:6.3f}, {quat[1]:6.3f}, {quat[2]:6.3f}, {quat[3]:6.3f}]")

    def list_scene_bodies(self):
        """单独显示场景物体状态"""
        if not self.scene_bodies:
            print("📦 当前模型无场景物体")
            return
        print(f"\n📦 场景物体状态:")
        for body_idx, body_name in self.scene_bodies:
            jnt_adr = self.model.body_jntadr[body_idx]
            pos = self.data.qpos[jnt_adr:jnt_adr + 3]
            quat = self.data.qpos[jnt_adr + 3:jnt_adr + 7]
            print(f"  {body_name:<20} pos: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]"
                  f"  quat: [{quat[0]:6.3f}, {quat[1]:6.3f}, {quat[2]:6.3f}, {quat[3]:6.3f}]")

    def reset_joints(self):
        """重置所有关节"""
        print("🔄 重置所有关节...")
        for joint_name in self.joint_targets:
            self.joint_targets[joint_name] = 0.0
            self.joint_velocities[joint_name] = 0.0

        # 清零所有 ctrl
        self.data.ctrl[:] = 0.0

        # 根据是否有 keyframe 选择重置方式
        if self.has_keyframe:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        print("✅ 关节已重置，ctrl 全部归零" + (" (已从 keyframe 恢复)" if self.has_keyframe else ""))

    def print_current_state(self):
        """打印当前状态"""
        print(f"\n📊 当前状态:")
        print(f"  ⏰ 仿真时间: {self.data.time:.3f}s")
        print(f"  🎮 控制模式: {self.control_mode}")
        print(f"  🔄 关节数量: {self.model.njnt}")
        print(f"  🎮 执行器数量: {self.model.nu}")
        if self.scene_bodies:
            print(f"  📦 场景物体: {len(self.scene_bodies)} 个")

        if self.model.njnt > 0:
            print(f"  📈 关节状态摘要:")
            positions = self.data.qpos[:self.model.njnt]
            velocities = self.data.qvel[:self.model.njnt] if self.model.njnt <= len(self.data.qvel) else [
                                                                                                             0.0] * self.model.njnt

            print(f"    位置范围: [{np.min(positions):.3f}, {np.max(positions):.3f}]")
            print(f"    速度范围: [{np.min(velocities):.3f}, {np.max(velocities):.3f}]")

        if self.model.nu > 0:
            ctrl_min, ctrl_max = np.min(self.data.ctrl), np.max(self.data.ctrl)
            print(f"  ctrl 范围: [{ctrl_min:.3f}, {ctrl_max:.3f}]")

    def set_control_mode(self, command):
        """设置控制模式"""
        parts = command.split()
        if len(parts) != 2:
            print("❌ 用法: mode <position|velocity>")
            return

        mode = parts[1].lower()
        if mode in ['position', 'velocity']:
            self.control_mode = mode
            print(f"✅ 控制模式已切换到: {mode}")
        else:
            print("❌ 无效模式，请使用 'position' 或 'velocity'")

    def parse_joint_command(self, command):
        """解析关节控制命令，写入 data.ctrl（position 执行器）"""
        parts = command.split()
        if len(parts) != 2:
            print("❌ 用法: <关节名称> <数值>")
            return

        joint_name, value_str = parts

        try:
            value = float(value_str)
        except ValueError:
            print(f"❌ 无效数值: {value_str}")
            return

        # 先尝试作为关节名查找
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            # 再尝试作为执行器名查找
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if act_id >= 0:
                # 从执行器反查关节名
                joint_id_found = self.model.actuator_trnid[act_id][0] if hasattr(self.model, 'actuator_trnid') and self.model.actuator_trnid.shape[1] > 0 else -1
                if joint_id_found >= 0:
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id_found)
                    joint_id = joint_id_found
            if joint_id < 0:
                print(f"❌ 未找到关节或执行器: {joint_name}")
                print("💡 使用 'list' 命令查看可用关节")
                return

        joint_range = self.model.jnt_range[joint_id]
        if joint_range[0] != joint_range[1]:
            if value < joint_range[0] or value > joint_range[1]:
                print(f"⚠️  值 {value:.3f} 超出关节范围 [{joint_range[0]:.3f}, {joint_range[1]:.3f}]")

        self.joint_targets[joint_name] = value
        # 优先通过 ctrl 写入（position 执行器），回退到直接写 qpos
        act_idx = self.actuator_map.get(joint_name)
        if act_idx is not None and act_idx < self.model.nu:
            self.data.ctrl[act_idx] = value
            print(f"✅ 设置关节 {joint_name} (ACT[{act_idx}]) ctrl: {value:.3f}")
        elif joint_id < len(self.data.qpos):
            self.data.qpos[joint_id] = value
            print(f"✅ 设置关节 {joint_name} qpos: {value:.3f}")
        else:
            print(f"⚠️  无法控制关节 {joint_name}: 无执行器且索引越界")


def find_mjcf_files(directory=None):
    """查找可用的MJCF文件（搜索项目根目录及子目录）"""
    mjcf_files = set()

    if directory is None:
        # 搜索项目根目录下的 .xml 文件（如 scene_dual_arm_plate.xml）
        for file in os.listdir("."):
            if file.endswith('.xml') and os.path.isfile(file):
                mjcf_files.add(os.path.join(".", file))
        # 同时搜索子目录（KITT 模型文件夹等）
        search_dirs = ["KITT1_5V3robot_dual_dahuan_urdf", "mjcf_models"]
        for d in search_dirs:
            if os.path.isdir(d):
                for root, dirs, files in os.walk(d):
                    for file in files:
                        if file.endswith('.xml'):
                            mjcf_files.add(os.path.join(root, file))
    else:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.xml'):
                    mjcf_files.add(os.path.join(root, file))

    return sorted(mjcf_files)


def list_available_models():
    """列出可用的模型"""
    mjcf_files = find_mjcf_files()

    if not mjcf_files:
        print("❌ 未找到MJCF文件")
        print("💡 请确保 KITT1_5V3robot_dual_dahuan_urdf 目录中存在 .xml 文件")
        return None

    print("📁 可用的MJCF模型:")
    for i, file in enumerate(mjcf_files, 1):
        basename = os.path.basename(file)
        size = os.path.getsize(file)
        size_str = f"{size / 1024:.1f}KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f}MB"
        print(f"  {i:2d}. {basename:<20} ({size_str})")

    return mjcf_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MJCF模型查看器")
    parser.add_argument("model", nargs="?", help="MJCF文件路径或模型名称")
    parser.add_argument("--list", "-l", action="store_true", help="列出可用模型")
    parser.add_argument("--directory", "-d", default=None, help="MJCF文件目录（默认自动搜索 KITT1_5V3 模型）")

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    mjcf_path = None

    if args.model:
        # 检查是否为完整路径
        if os.path.exists(args.model):
            mjcf_path = args.model
        else:
            # 在 KITT 模型目录中查找
            search_dirs = [args.directory] if args.directory else ["KITT1_5V3robot_dual_dahuan_urdf", "mjcf_models", "."]
            for d in search_dirs:
                if not os.path.exists(d):
                    continue
                potential_path = os.path.join(d, args.model)
                if os.path.exists(potential_path):
                    mjcf_path = potential_path
                    break
                if os.path.exists(potential_path + ".xml"):
                    mjcf_path = potential_path + ".xml"
                    break
            else:
                print(f"❌ 未找到模型文件: {args.model}")
                mjcf_files = list_available_models()
                return
    else:
        # 交互式选择
        mjcf_files = list_available_models()
        if not mjcf_files:
            return

        try:
            choice = input(f"\n请选择模型 (1-{len(mjcf_files)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(mjcf_files):
                mjcf_path = mjcf_files[index]
            else:
                print("❌ 无效选择")
                return
        except (ValueError, KeyboardInterrupt):
            print("❌ 操作取消")
            return

    if not mjcf_path:
        print("❌ 未指定模型文件")
        return

    # 启动查看器
    try:
        viewer = MJCFViewer(mjcf_path)
        viewer.load_model()
        viewer.start_viewer()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"❌ 程序错误: {e}")


if __name__ == "__main__":
    main()
