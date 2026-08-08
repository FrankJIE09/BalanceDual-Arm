---
name: balance-dual-arm-sim-gui
description: >-
  Run BalanceDual-Arm MuJoCo dual-arm plate simulation with GUI only (never headless).
  Use in this repository when the user asks to run simulation, launch the controller,
  test HOLD/CARRY, show the viewer, or debug with visual feedback.
---

# BalanceDual-Arm 仿真（必须 GUI）

仅适用于本仓库 `/home/lenovo/Frank/code/BalanceDual-Arm`。

## 硬性规则

1. **每次运行仿真必须带 `--gui`**，禁止无界面 / headless 默认跑。
2. 不要主动加 `--duration` 做无窗口批跑来“代替”给用户看效果；用户要看窗口时就开 GUI。
3. 环境：`conda activate wbc_validation`（需已安装 `mujoco`）。
4. 工作目录：仓库根目录。
5. 当前物理设定：**无 `plate_mocap_weld`**，平板靠夹爪摩擦夹持。

## 启动命令

```bash
cd /home/lenovo/Frank/code/BalanceDual-Arm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wbc_validation
export DISPLAY="${DISPLAY:-:1}"

# 默认 CARRY（夹爪夹持 + 沿 +X 慢移）
python dual_arm_controller.py --gui

# 仅保持
python dual_arm_controller.py --hold --gui
```

GUI 为阻塞进程：用后台启动，并确认窗口已起来；用户关窗口后进程结束。

## 可选参数

| 参数 | 含义 | 默认 |
|------|------|------|
| `--carry-hold` | 搬运前静止 (s) | 0.5 |
| `--carry-move` | 平移持续 (s) | 6.0 |
| `--carry-dist` | +X 距离 (m) | 0.06 |
| `--csv` | CSV 路径 | `logs/carry_last.csv` |

示例：

```bash
python dual_arm_controller.py --gui --carry-dist 0.06 --csv logs/carry_gui.csv
```

## 不要做的事

- 不要用无 `--gui` 的命令冒充“已运行仿真给用户看”。
- 不要恢复 `plate_mocap_weld`，除非用户明确要求重新硬焊。
- 不要改用其他仓库的 anyverse / WBC GUI skill。
