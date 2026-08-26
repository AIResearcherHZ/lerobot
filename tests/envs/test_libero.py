from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("libero")

from lerobot.envs.libero import LiberoEnv, _get_suite, get_task_init_states
from lerobot.scripts.lerobot_deploy_libero import _resolve_observation_size, _validate_task_text


def test_standard_libero_suite_ignores_plus_perturbation_tasks(capsys):
    suite = _get_suite("libero_object", is_libero_plus=False)
    capsys.readouterr()

    assert len(suite.tasks) == 10
    assert suite.tasks[0].name == "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    assert get_task_init_states(suite, 0, is_libero_plus=False).shape[0] > 0


def test_libero_plus_suite_keeps_perturbation_tasks(capsys):
    suite = _get_suite("libero_object", is_libero_plus=True)
    capsys.readouterr()

    assert len(suite.tasks) > 10
    assert suite.tasks[0].name == "pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_1"
    assert get_task_init_states(suite, 0, is_libero_plus=True).shape[0] > 0


def test_onscreen_render_uses_wrapped_robosuite_env():
    render_calls = []
    inner_env = SimpleNamespace(
        render=lambda: render_calls.append(True),
        _get_observations=lambda: {},
    )
    env = LiberoEnv.__new__(LiberoEnv)
    env._env = SimpleNamespace(env=inner_env)
    env.onscreen_renderer = True
    env._ensure_env = lambda: None
    env._format_raw_obs = lambda _: {"pixels": {"image": np.zeros((2, 2, 3), dtype=np.uint8)}}

    env.render()

    assert render_calls == [True]


def test_deploy_rejects_task_text_for_another_object():
    with pytest.raises(ValueError, match="任务文本与 task id 不匹配"):
        _validate_task_text(
            "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
            "pick up the alphabet soup and place it in the basket",
        )


def test_deploy_accepts_matching_task_text():
    _validate_task_text(
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick up the alphabet soup and place it in the basket",
    )


def test_deploy_uses_checkpoint_observation_size():
    cfg = SimpleNamespace(
        image_features={"image": SimpleNamespace(shape=(3, 256, 256))}
    )
    assert _resolve_observation_size(cfg, None, None) == (256, 256)


def test_deploy_requires_both_observation_dimensions():
    cfg = SimpleNamespace(image_features={"image": SimpleNamespace(shape=(3, 256, 256))})
    with pytest.raises(ValueError, match="必须同时指定"):
        _resolve_observation_size(cfg, 256, None)
