from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _metadata(file: h5py.File) -> tuple[str, str, dict]:
    data = file["data"]
    problem = json.loads(_text(data.attrs["problem_info"]))
    task = problem.get("language_instruction", "LIBERO task")
    env_name = _text(data.attrs.get("env", "LIBERO"))
    env_info = json.loads(_text(data.attrs.get("env_info", "{}")))
    return task, env_name, env_info


def _create_dataset(output: Path, fps: int, image_shape: tuple[int, int, int] | None, state_shape: int) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_shape,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": None,
        },
    }
    if image_shape is not None:
        features.update(
            {
                "observation.images.image": {
                    "dtype": "video",
                    "shape": image_shape,
                    "names": ["height", "width", "channels"],
                },
                "observation.images.image2": {
                    "dtype": "video",
                    "shape": image_shape,
                    "names": ["height", "width", "channels"],
                },
            }
        )
    return LeRobotDataset.create(
        repo_id="local/libero_hdf5",
        root=output,
        fps=fps,
        robot_type="Panda",
        features=features,
        use_videos=image_shape is not None,
        image_writer_threads=2 if image_shape is not None else 0,
    )


def _convert_low_dim(file: h5py.File, dataset: LeRobotDataset, task: str) -> int:
    frames = 0
    for episode_name in sorted(file["data"]):
        episode = file["data"][episode_name]
        states = np.asarray(episode["states"], dtype=np.float32)
        actions = np.asarray(episode["actions"], dtype=np.float32)
        if len(states) != len(actions):
            raise ValueError(f"{episode_name}: states/actions 长度不一致: {len(states)} != {len(actions)}")
        for state, action in zip(states, actions, strict=True):
            dataset.add_frame({"observation.state": state, "action": action, "task": task})
        dataset.save_episode()
        frames += len(states)
    return frames


def _load_libero():
    root = Path(__file__).resolve().parents[1] / "LIBERO-plus"
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(root))
    import libero.libero.utils.utils as libero_utils
    from libero.libero.envs import TASK_MAPPING

    return libero_utils, TASK_MAPPING


def _convert_with_images(
    file: h5py.File,
    dataset: LeRobotDataset,
    task: str,
    env_name: str,
    env_info: dict,
    image_size: int,
    skip_frames: int,
) -> int:
    _, task_mapping = _load_libero()
    import libero.libero.utils.utils as libero_utils
    problem = json.loads(_text(file["data"].attrs["problem_info"]))
    problem_name = problem["problem_name"]
    kwargs = dict(env_info)
    bddl_file = _text(file["data"].attrs["bddl_file_name"])
    if not Path(bddl_file).exists():
        candidate = Path(__file__).resolve().parents[1] / "LIBERO-plus" / bddl_file
        if candidate.exists():
            bddl_file = str(candidate)
    libero_utils.update_env_kwargs(
        kwargs,
        bddl_file_name=bddl_file,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_depths=False,
        camera_names=["robot0_eye_in_hand", "agentview"],
        reward_shaping=True,
        control_freq=20,
        camera_heights=image_size,
        camera_widths=image_size,
        camera_segmentations=None,
    )
    env = task_mapping[problem_name](**kwargs)
    frames = 0
    try:
        for episode_name in sorted(file["data"]):
            episode = file["data"][episode_name]
            states = np.asarray(episode["states"])
            actions = np.asarray(episode["actions"], dtype=np.float32)
            if len(states) != len(actions):
                raise ValueError(f"{episode_name}: states/actions 长度不一致")
            model_xml = _text(episode.attrs["model_file"])
            env.reset_from_xml_string(model_xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()
            for index, action in enumerate(actions):
                obs, _, _, _ = env.step(action)
                if index < skip_frames:
                    continue
                joints = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
                gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
                state = np.concatenate([joints, [float(gripper.mean())]]).astype(np.float32)
                image = np.asarray(obs["agentview_image"], dtype=np.uint8)
                image2 = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
                dataset.add_frame(
                    {
                        "observation.state": state,
                        "action": action,
                        "observation.images.image": image,
                        "observation.images.image2": image2,
                        "task": task,
                    }
                )
                frames += 1
            dataset.save_episode()
    finally:
        env.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="将 LIBERO/robosuite demo.hdf5 转为 LeRobot v3")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--with-images", action="store_true")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--skip-frames", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，请换一个路径: {args.output}")
    with h5py.File(args.input, "r") as file:
        task, env_name, env_info = _metadata(file)
        if args.with_images:
            dataset = _create_dataset(args.output, args.fps, (args.image_size, args.image_size, 3), 8)
            count = _convert_with_images(file, dataset, task, env_name, env_info, args.image_size, args.skip_frames)
        else:
            first = file["data"][sorted(file["data"])[0]]["states"]
            dataset = _create_dataset(args.output, args.fps, None, int(first.shape[1]))
            count = _convert_low_dim(file, dataset, task)
        dataset.finalize()
    print(f"转换完成: {count} 帧 -> {args.output}")


if __name__ == "__main__":
    main()
