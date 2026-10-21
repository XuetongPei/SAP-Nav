"""RobotwinSceneRef — one RoboTwin task under one task config (pure data,
layer L0).

A RoboTwin scene is not a file: it is a python class under ``envs/`` of the
RoboTwin checkout (``envs/beat_block_hammer.py`` declares
``class beat_block_hammer(Base_Task)``) whose ``load_actors`` builds the
tabletop from the assets and whose ``check_success`` is the task's own
predicate. What the class needs to become a concrete scene is the
embodiment and the task config (which domain randomisation, which cameras)
— those two, plus the task name, are what a scene reference carries. The
per-episode scene seed lives on the ``Episode`` (``info["seed"]``), the way
a LIBERO episode's init-state index does.

``task_config`` names a file of ``env_cfg/task_config/`` and is the axis
RoboTwin's own evaluation is parameterised on (``--task-config
demo_clean | demo_randomized``); the numbers in it are transcribed into the
body's ``RobotwinRandomization``, so the config name here is provenance and
the loaded scene never reads the file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotwinSceneRef:
    task: str                 # the module and class name under envs/, e.g. "beat_block_hammer"
    embodiment: str           # a key of env_cfg/task_config/_embodiment_config.yml, e.g. "aloha-agilex"
    # demo_clean | demo_randomized — which env_cfg/task_config/<name>.yml this line transcribes
    task_config: str
    # the task's own cap on take_action calls (env_cfg/task_config/_eval_step_limit.yml)
    step_limit: int

    @property
    def scene_file(self) -> str:
        """What identifies the scene, as ``scene_file`` does on the other
        engines: the task module's path inside the RoboTwin checkout."""
        return f"envs/{self.task}.py"

    @property
    def instruction_file(self) -> str:
        """The task's language file inside the RoboTwin checkout."""
        return f"description/task_instruction/{self.task}.json"
