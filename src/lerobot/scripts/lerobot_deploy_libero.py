#!/usr/bin/env python
"""LIBERO Panda 实时策略部署：终端输入 checkpoint、任务文本和 task id。"""

from __future__ import annotations

import argparse
import json
import os
import re
from itertools import count
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "glfw")

from lerobot.envs import (
    close_envs,
    make_env,
    make_env_config,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="策略 pretrained_model 目录或 Hub ID")
    p.add_argument("--vla-name", default=None, help="显式声明 VLA 类型（evo1/smolvla），用于和 checkpoint config.json 的 type 字段校验一致")
    p.add_argument("--task-text", help="语言任务；未提供时启动后在终端输入")
    p.add_argument("--suite", default="libero_object")
    p.add_argument(
        "--task-id",
        type=int,
        default=3,
        help="LIBERO task id，用于选择仿真场景",
    )
    p.add_argument("--robot", default="Panda")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vlm-model-name", default=None)
    p.add_argument("--n-action-steps", type=int, default=None, help="覆盖 checkpoint 的 action chunk 长度")
    p.add_argument("--postprocess-action-dim", type=int, default=7)
    p.add_argument("--gripper-threshold", type=float, default=0.0)
    p.add_argument("--gripper-below-threshold-value", type=float, default=-1.0)
    p.add_argument("--gripper-above-threshold-value", type=float, default=1.0)
    p.add_argument("--camera-name", default="agentview_image,robot0_eye_in_hand_image")
    p.add_argument("--render-camera", default="agentview")
    p.add_argument("--max-steps", type=int, default=0, help="最大执行步数；0 表示不限制")
    p.add_argument("--observation-width", type=int, default=None, help="覆盖 checkpoint 的图像宽度")
    p.add_argument("--observation-height", type=int, default=None, help="覆盖 checkpoint 的图像高度")
    return p.parse_args()


def _validate_task_text(task_name: str, task_text: str) -> None:
    match = re.match(r"pick_up_the_(.+?)_and_", task_name)
    if match is None:
        return
    target = match.group(1).replace("_", " ")
    normalized_text = " ".join(task_text.lower().replace("_", " ").split())
    if target not in normalized_text:
        raise ValueError(
            f"任务文本与 task id 不匹配：环境任务是 '{task_name}'，目标物体是 '{target}'，"
            f"但任务文本是 '{task_text}'。"
        )


def _resolve_observation_size(policy_cfg, width: int | None, height: int | None) -> tuple[int, int]:
    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("--observation-width 和 --observation-height 必须同时指定")
        return width, height
    image_features = policy_cfg.image_features
    if not image_features:
        raise ValueError("checkpoint 未声明视觉输入，请显式指定图像尺寸")
    shape = next(iter(image_features.values())).shape
    if len(shape) != 3:
        raise ValueError(f"checkpoint 图像特征维度无效: {shape}")
    _, image_height, image_width = shape
    return image_width, image_height


def main() -> None:
    args = _args()
    task_text = args.task_text or input("请输入文本任务: ").strip()
    if not task_text:
        raise ValueError("任务文本不能为空")
    task_id = args.task_id

    checkpoint = Path(args.checkpoint).expanduser()
    checkpoint_is_local = (
        checkpoint.is_dir()
        or checkpoint.is_absolute()
        or str(checkpoint).startswith(".")
        or "/" in str(checkpoint)
        or "\\" in str(checkpoint)
    )
    if checkpoint_is_local and not checkpoint.is_dir():
        raise FileNotFoundError(
            f"本地 checkpoint 目录不存在: {checkpoint}。"
            "80000 步的目录名通常是 checkpoints/080000/pretrained_model。"
        )
    checkpoint_ref = str(checkpoint.resolve()) if checkpoint.is_dir() else args.checkpoint

    from lerobot.configs.policies import PreTrainedConfig

    if args.vla_name is not None:
        ckpt_dir = checkpoint
        config_json = ckpt_dir / "config.json" if ckpt_dir.is_dir() else None
        if config_json is not None and config_json.exists():
            ckpt_type = json.loads(config_json.read_text()).get("type")
            if ckpt_type is not None and ckpt_type != args.vla_name:
                raise ValueError(
                    f"--vla-name='{args.vla_name}' 与 checkpoint config.json 的 type='{ckpt_type}' 不一致"
                )

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_ref)
    policy_cfg.pretrained_path = checkpoint_ref
    policy_cfg.device = args.device
    if args.vlm_model_name is not None and hasattr(policy_cfg, "vlm_model_name"):
        policy_cfg.vlm_model_name = args.vlm_model_name
    if hasattr(policy_cfg, "use_flash_attn"):
        policy_cfg.use_flash_attn = False
    if args.n_action_steps is not None and hasattr(policy_cfg, "n_action_steps"):
        policy_cfg.n_action_steps = args.n_action_steps
    if hasattr(policy_cfg, "postprocess_action_dim"):
        policy_cfg.postprocess_action_dim = args.postprocess_action_dim
    if hasattr(policy_cfg, "binarize_gripper"):
        policy_cfg.binarize_gripper = True
    if hasattr(policy_cfg, "gripper_threshold"):
        policy_cfg.gripper_threshold = args.gripper_threshold
    if hasattr(policy_cfg, "gripper_below_threshold_value"):
        policy_cfg.gripper_below_threshold_value = args.gripper_below_threshold_value
    if hasattr(policy_cfg, "gripper_above_threshold_value"):
        policy_cfg.gripper_above_threshold_value = args.gripper_above_threshold_value

    observation_width, observation_height = _resolve_observation_size(
        policy_cfg, args.observation_width, args.observation_height
    )
    env_cfg = make_env_config(
        "libero",
        task=args.suite,
        task_ids=[task_id],
        robot=args.robot,
        camera_name=args.camera_name,
        observation_width=observation_width,
        observation_height=observation_height,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        onscreen_renderer=True,
        render_camera=args.render_camera,
    )

    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)[args.suite][task_id]
    task_name = envs.call("task")[0]
    task_description = envs.call("task_description")[0]
    print(
        f"环境任务: task_id={task_id}, name={task_name}, environment_instruction={task_description}, "
        f"vla_instruction={task_text}"
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    policy.reset()
    obs, _ = envs.reset()
    try:
        envs.call("render")
        steps = count() if args.max_steps <= 0 else range(args.max_steps)
        for step in steps:
            policy_obs = preprocess_observation(obs)
            policy_obs["task"] = [task_text]
            policy_obs = env_pre(policy_obs)
            policy_obs = pre(policy_obs)
            policy_obs["task"] = [task_text]
            with torch.inference_mode():
                action = policy.select_action(policy_obs)
            action = post(action)
            action = env_post({ACTION: action})[ACTION].detach().cpu().numpy()
            obs, _, terminated, truncated, info = envs.step(action)
            envs.call("render")
            if bool(np.asarray(terminated).any() or np.asarray(truncated).any()):
                success = bool(np.asarray(info.get("is_success", False)).any())
                print(f"回合结束: step={step + 1}, success={success}")
                if success:
                    break
                obs, _ = envs.reset()
                policy.reset()
    except KeyboardInterrupt:
        print("收到 Ctrl+C，停止部署")
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()
