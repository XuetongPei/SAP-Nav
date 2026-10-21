"""CalvinSceneRef — one CALVIN play table on disk (pure data, layer L0).

CALVIN ships four play tables, A / B / C / D: the same furniture (a sliding
cabinet, a drawer, a button, a switch, two lights) at different positions,
with differently coloured table tops
(``calvin_env/conf/scene/calvin_scene_{A,B,C,D}.yaml``). The published board
rows are ABC->D and ABCD->D, and **both evaluate in D** — the letters name
the *training* split, not the evaluation environment (see ``benchmarks/
calvin.py``).

A scene is identified by its letter and by the hydra config the environment
is built from. Upstream's evaluator builds it from the dataset:
``calvin_models/calvin_agent/evaluation/evaluate_policy.py``:48

    def make_env(dataset_path):
        val_folder = Path(dataset_path) / "validation"
        env = get_env(val_folder, show_gui=False)

and ``calvin_env/envs/play_table_env.py``:275 reads
``<val_folder>/.hydra/merged_config.yaml`` from it. ``config_file`` is that
path, so the rig, the robot and the scene are the release's own — nothing is
re-composed here.

Unlike LIBERO's, a CALVIN episode has **no stored init state**: the
evaluator computes the initial ``robot_obs`` / ``scene_obs`` from the
episode's symbolic initial condition
(``calvin_agent/evaluation/utils.py``:207 ``get_env_state_for_initial_condition``),
so there is no index into the dataset to keep here. The condition and the
five-task chain ride ``Episode.info``.
"""

from __future__ import annotations

from dataclasses import dataclass

LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class CalvinSceneRef:
    env_letter: str       # A | B | C | D — the play table; the board always evaluates in D
    config_file: str      # absolute path to <dataset>/validation/.hydra/merged_config.yaml

    def __post_init__(self) -> None:
        if self.env_letter not in LETTERS:
            raise ValueError(f"CalvinSceneRef: env_letter must be one of {LETTERS}, got {self.env_letter!r}")

    @property
    def scene_file(self) -> str:
        """The hydra config the environment is built from — what identifies
        the scene, as ``scene_file`` does on the other engines."""
        return self.config_file
