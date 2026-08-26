# Semi-Taks-T1 与 LIBERO-plus 文件说明

本文档说明 Semi-Taks-T1 资产在本项目中的位置，以及各文件的用途。

## 资产目录

这些文件复制自原始 Taks 工程目录：

```text
/home/xhz/taks-controller-web/backend/assets/Semi_Taks_T1/
```

实际资产目录是：

```text
/home/xhz/lerobot/LIBERO-plus/libero/assets/robots/semi_taks_t1/
```

LIBERO-plus 还提供了以下兼容路径：

```text
/home/xhz/lerobot/LIBERO-plus/libero/libero/assets/robots/semi_taks_t1/
```

其中 `libero/libero/assets` 是指向 `../assets` 的符号链接，两处指向同一份文件。

资产目录中的文件：

- `Semi_Taks_T1.xml`：从 Taks 工程复制的完整 Semi-Taks-T1 MuJoCo 模型，保留左臂、右臂、腰部和头部。
- `scene_Semi_Taks_T1.xml`：原始完整机器人场景文件。
- `Semi_Taks_T1.urdf`：原始机器人 URDF，网格引用已改为本地 `meshes/` 路径。
- `meshes/`：机器人本体和 DM 夹爪的 STL 网格。
- `robot.xml`：LIBERO-plus 专用右臂模型，只保留 7 个右臂关节。
- `gripper.xml`：LIBERO-plus 专用双指夹爪模型，包含 2 个夹爪关节和 1 个夹爪执行器。
- `batch_rotate_stl.py`、`calculate_inertia_from_stl.py`、`compute_reach.py`、`update_inertia.py`、`update_inertia_all_files.py`：原始资产工具脚本。

## 生成脚本

```text
/home/xhz/lerobot/LIBERO-plus/benchmark_scripts/build_semi_taks_t1_libero.py
```

该脚本从 `Semi_Taks_T1.xml` 重新生成 `robot.xml` 和 `gripper.xml`。模型路径经过处理，可以被 MuJoCo 独立加载，也可以被 robosuite 合并到 LIBERO 环境中。

重新生成命令：

```bash
cd /home/xhz/lerobot
python3 LIBERO-plus/benchmark_scripts/build_semi_taks_t1_libero.py
```

## 机器人适配代码

- `LIBERO-plus/libero/libero/envs/robots/semi_taks_t1.py`：定义 `SemiTaksT1`、自定义夹爪以及桌面/地面任务使用的兼容别名。
- `LIBERO-plus/libero/libero/envs/__init__.py`：注册机器人和夹爪名称。
- `LIBERO-plus/libero/libero/envs/env_wrapper.py`：兼容 LIBERO-plus 对桌面任务的机器人名称处理。
- `src/lerobot/envs/libero.py`：向 LIBERO 环境传递可配置的机器人名称。
- `src/lerobot/envs/configs.py`：普通 `libero` 默认使用 `Panda`，`libero_plus` 默认使用 `SemiTaksT1`。

## 自由度和动作

Semi-Taks-T1 在 LIBERO-plus 中只启用右臂：

```text
right_shoulder_pitch
right_shoulder_roll
right_shoulder_yaw
right_elbow
right_wrist_roll
right_wrist_yaw
right_wrist_pitch
```

左臂、腰部和头部不作为自由关节控制。LIBERO 的动作接口仍然是 7 维：6 维末端位姿控制加 1 维夹爪控制。观测中的右臂关节为 7 维，双指夹爪 qpos 为 2 维。

## 使用配置

普通 LIBERO：

```text
--env.type=libero
```

默认机器人是 `Panda`。

LIBERO-plus：

```text
--env.type=libero_plus
```

默认机器人是 `SemiTaksT1`。如果直接使用底层 robosuite 环境，机器人名称应为 `SemiTaksT1`；LIBERO-plus 内部生成的 `MountedSemiTaksT1` 和 `OnTheGroundSemiTaksT1` 也已注册。

运行 LIBERO-plus 时需要让 Python 找到本地 fork：

```bash
cd /home/xhz/lerobot
PYTHONPATH=$PWD/LIBERO-plus:$PWD/LIBERO-plus/libero \
LIBERO_CONFIG_PATH=/home/xhz/.libero \
MUJOCO_GL=egl \
uv run python -c "from lerobot.envs.configs import LiberoPlusEnv; print(LiberoPlusEnv().robot)"
```

## 已验证内容

- `robot.xml` 可独立加载，模型维度为 `nq=7, nv=7, nu=7`。
- `gripper.xml` 可独立加载，模型维度为 `nq=2, nv=2, nu=1`。
- LIBERO-plus BDDL 环境可以 reset。
- `agentview` 和 `robot0_eye_in_hand` 图像可以生成。
- 右臂关节观测形状为 `(7,)`，夹爪观测形状为 `(2,)`。
- 7 维动作可以执行一步 MuJoCo/robosuite 仿真。

URDF 的 `check_urdf` 可以解析原始树结构；URDF 专用校验器提示的 `floating` 根关节和 `<mujoco>` 扩展属于原始 URDF 结构。LIBERO-plus 实际运行使用的是上述生成的 MuJoCo XML。
