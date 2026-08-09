# BalanceDual-Arm

轮式移动双臂机器人协同稳载仿真（MuJoCo）：双臂夹持平板，力矩控制跟踪期望板位姿。

<p align="center">
  <img src="images/image.png" alt="双臂协同稳载初始夹持姿态" width="720"/>
</p>

## 功能概览

| 模式 | 说明 |
|------|------|
| **HOLD** | 世界系锁定初始法兰 TCP，手指持续夹紧 |
| **CARRY** | 期望板位姿误差 → 板体系速度 → TCP → DLS → 关节力矩 PD |

当前物理设定：**无 `plate_mocap_weld`**，平板靠夹爪摩擦夹持；执行器为 **`motor` 力矩模式**（非位置执行器）。关节带 `armature` 近似减速器折算惯量；**刚度在软件 PD**（`config.yaml` 的 `kp`/`kd`）。

## 环境

- Python 3.8+
- Conda 环境示例：`wbc_validation`
- 依赖见 [`requirements.txt`](requirements.txt)

```bash
conda activate wbc_validation
pip install -r requirements.txt
```

## 快速运行

```bash
cd /path/to/BalanceDual-Arm
conda activate wbc_validation
export DISPLAY="${DISPLAY:-:1}"

# 默认读取 config.yaml（GUI + CARRY + 键盘遥操）
python dual_arm_controller.py

# 指定配置
python dual_arm_controller.py --config config.yaml
```

几乎所有参数在 **`config.yaml`**；命令行仅 `--config`。

## 目录结构

```
BalanceDual-Arm/
├── config.yaml                 # 仿真 / CARRY / 控制参数
├── dual_arm_controller.py      # 主程序（控制 + GUI）
├── scene_dual_arm_plate.xml    # MJCF 场景（机器人 + 板 + 杯/球）
├── requirements.txt
├── mjcf_viewer.py              # 场景查看器
├── KITT1_5V3robot_dual_dahuan_urdf/   # 网格与本体资源
├── logs/                       # CSV 运行日志
├── images/
└── .cursor/skills/balance-dual-arm-sim-gui/   # GUI 仿真 skill
```

## 控制链路（CARRY）

```
init_pose / target_pose（YAML）
    │  settle → move：位置插值 + 姿态四元数 slerp
    │  可选键盘：板体系 6D 增量叠在轨迹目标上
    ▼
期望板位姿误差（板系）→ v_body, ω_body（限幅）
    ▼
板旋量 → 左右 TCP 速度 + 相对期望板的 TCP 位姿伺服
    ▼
DLS(J) → q̇ → 积分限位得到 q_des
    ▼
τ = g(q) + Kp·(q_des−q) + Kd·(q̇_des−q̇) → data.ctrl（motor）
手指：τ = g + PD + finger_close_bias
```

- TCP site：`tcp_L` / `tcp_R`（法兰中心）
- 末端雅可比：`mj_jacSite`

## 配置说明（`config.yaml`）

### `sim`

| 键 | 含义 |
|----|------|
| `gui` | 是否开 MuJoCo 被动查看器 |
| `mode` | `hold` / `carry` |
| `keyboard` | GUI 下启用 pynput 键盘遥操 |
| `duration` | **仅无 GUI** 时的仿真时长（s） |
| `csv` / `csv_every` | CSV 路径与采样间隔（步） |
| `log_interval_s` | 控制台日志间隔（s） |

GUI 按 `timestep=0.002` s（2 ms）实时 pacing。

### `carry`

| 键 | 含义 |
|----|------|
| `settle_s` | 目标保持初始位姿的时间 |
| `move_s` | 插值到终点的时间 |
| `init_pose.rpy` | 场景初始板姿态（世界系 RPY，rad） |
| `target_pose.delta_body` | 终点相对初始、在**期望板体**下的平移（m） |
| `target_pose.rpy` | 终点姿态（世界系 RPY） |
| `teleop.v_lin` | 键盘线速度（m/s） |
| `teleop.w_ang` | 键盘角速度（rad/s） |

姿态：`R = Rz(yaw)·Ry(pitch)·Rx(roll)`；轨迹姿态用**四元数 slerp**。

### `control`

关节力矩 PD、手指偏置、TCP / 板外环增益与速度限幅均在此段，按注释调参即可。

## 键盘遥操（`sim.keyboard: true`）

在**期望板体坐标系**下，按住键持续运动（速率见 `carry.teleop`）：

| 键 | 方向 |
|----|------|
| `q` / `a` | ±X |
| `w` / `s` | ±Y |
| `e` / `d` | ±Z |
| `r` / `f` | ±roll |
| `t` / `g` | ±pitch |
| `y` / `h` | ±yaw |

每仿真步增量 ≈ 速率 × `0.002`。键盘增量叠在 settle/move 轨迹目标之上。关 GUI 窗口退出。

## 场景与执行器要点

- `scene_dual_arm_plate.xml`：KITT1_5V3 双臂 + 平板 / 杯 / 球
- 执行器：全部 **`motor`**，`gear=1`，`ctrl` = 关节力矩
- 关节 class：`arm_prox` / `arm_mid` / `arm_dist` 等带 `armature`、`damping`、`frictionloss`
- 手指镜像：`equality` 约束；夹紧靠 PD + `finger_close_bias`

## 机器人自由度（摘要）

| 部位 | 数量 | 名称示例 |
|------|------|----------|
| 腿 | 8 | `ZQ`, `ZQL`, … |
| 躯干 | 4 | `trunk_joint_1`…`4` |
| 左/右臂 | 7+7 | `left/right_arm_joint_*` |
| 头 | 2 | `head_joint_*` |
| 手指 | 4（2 个电机 + 镜像） | `*finger*_joint` |

## CSV 与诊断

开启 `sim.csv` 后写入板位姿、水平误差、TCP、关节、手指力矩等，便于复盘。控制台按 `log_interval_s` 打印 CARRY 相位、`level`、板目标与实际高度等。

## 相关工具

```bash
# 仅查看 MJCF（不跑控制器）
python mjcf_viewer.py
```

Cursor 本库 GUI 仿真约定见：`.cursor/skills/balance-dual-arm-sim-gui/SKILL.md`（必须带 GUI，勿默认 headless）。

## 版本

当前主程序标注 **v1.3**（YAML 目标位姿 + 四元数 slerp + 键盘遥操 + 力矩 PD / armature）。
