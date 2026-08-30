from pathlib import Path

import numpy as np

from robotarm.envs.damage import DamageConfig
from robotarm.envs.variable_mujoco_env import VariableMujocoArmEnv


ROOT = Path(__file__).resolve().parents[1]


def _env(robot: str) -> VariableMujocoArmEnv:
    if robot == "genkiarm":
        return VariableMujocoArmEnv(
            ROOT / "sim/assets/genkiarm_push.xml",
            joint_names=tuple(f"j{i}" for i in range(1, 6)),
            actuator_names=tuple(f"m{i}" for i in range(1, 6)),
            object_body="block", object_geom="block_geom",
        )
    return VariableMujocoArmEnv(
        ROOT / "sim/assets/panda_push_grasp.xml",
        joint_names=tuple(f"joint{i}" for i in range(1, 8)),
        actuator_names=tuple(f"actuator{i}" for i in range(1, 8)),
        object_body="task_cube", object_geom="cube_geom", home_keyframe="task_home",
    )


def test_full_joint_normalized_interface_steps_both_actual_models():
    for robot, dof in (("genkiarm", 5), ("panda", 7)):
        env = _env(robot)
        before = env.reset()
        after = env.step(np.linspace(-0.2, 0.2, dof))
        assert before[0].shape == after[0].shape == (dof, 2)
        assert before[1].shape == after[1].shape == (7,)
        assert before[2].shape == after[2].shape == (6,)
        assert np.isfinite(after[0]).all()


def test_same_analytic_lock_rule_is_exact_for_five_and_seven_dof():
    for robot, dof, locked in (("genkiarm", 5, 2), ("panda", 7, 3)):
        env = _env(robot)
        damage = DamageConfig.lock_single(locked, -0.31, dof=dof)
        env.reset(damage)
        for _ in range(5):
            state, _, _ = env.step(np.ones(dof))
            assert state[locked, 0] == damage.lock_angle[locked]
            assert state[locked, 1] == 0.0
            assert env.last_applied_action[locked] == 0.0
