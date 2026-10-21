"""BehaviorSceneRef — one BEHAVIOR-1K evaluation scenario (pure data, layer L0).

A BEHAVIOR episode is a **task instance**: an activity (one of the 1 000 BDDL
definitions, 50 of which the 2025 Challenge scores) sampled into one of three
scene models, at one of the 301 shipped initial states.

    task              the BDDL activity name, e.g. ``turning_on_radio`` — the
                      directory under ``bddl/activity_definitions/``
    scene_model       ``house_single_floor`` / ``house_double_floor_lower`` /
                      ``house_double_floor_upper``, from
                      ``joylo/sampled_task/available_tasks.yaml``
    definition_id     the BDDL problem index (``problem<id>.bddl``); every
                      challenge task ships exactly one, 0
    instance_id       which sampled initial state to load — the file
                      ``<scene_model>_task_<task>_<definition_id>_<instance_id>
                      _template-tro_state.json`` under the challenge's
                      ``2025-challenge-task-instances`` release

``get_cached_activity_scene_filename`` (``omnigibson/tasks/behavior_task.py``
L157-174) builds that stem, and the evaluator loads the file by replaying
every task-relevant object's state and the robot's pose
(``learning/eval.py`` ``load_task_instance`` L241-291) — the port does the
same, so an episode id names a reproducible initial state.
"""

from __future__ import annotations

from dataclasses import dataclass

SCENE_MODELS = ("house_single_floor", "house_double_floor_lower", "house_double_floor_upper")


@dataclass(frozen=True)
class BehaviorSceneRef:
    task: str                              # the BDDL activity name
    scene_model: str                       # available_tasks.yaml <task>/<definition_id>/scene_model
    instance_id: int                       # which sampled initial state (0..300 shipped)
    definition_id: int = 0                 # problem<id>.bddl; the challenge tasks ship only 0
    release: str = "2025-challenge"        # which task-instance release the id indexes

    @property
    def scene_file(self) -> str:
        """What identifies the scene, as ``scene_file`` does on the other engines."""
        return f"{self.scene_model}/{self.task}/{self.definition_id}-{self.instance_id}"

    @property
    def tro_stem(self) -> str:
        """The instance file's stem, as ``BehaviorTask.get_cached_activity_scene_filename``
        builds it (``omnigibson/tasks/behavior_task.py`` L157-174)."""
        return f"{self.scene_model}_task_{self.task}_{self.definition_id}_{self.instance_id}_template"

    @property
    def model_key(self) -> str:
        """What a rebuilt stage depends on — the scene model alone. A different
        task in the same model is an ``env.update_task`` (env_base.py L457-478);
        a different model needs the whole environment rebuilt."""
        return self.scene_model

    @property
    def task_key(self) -> tuple[str, str, int]:
        """What a reloaded task depends on: the model, the activity, its definition."""
        return (self.scene_model, self.task, self.definition_id)
