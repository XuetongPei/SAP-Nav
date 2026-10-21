"""LiberoSceneRef — one LIBERO task on disk (pure data, layer L0).

A task is a BDDL file (the scene, the objects, the goal predicates and the
language instruction) plus its ``init_states`` file, a torch-saved (50, N)
array of MuJoCo states — one per episode. Both ship inside the ``libero``
package (``libero.libero.get_libero_path("bddl_files" | "init_states")``);
the loader resolves them once per split.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiberoSceneRef:
    suite: str            # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
    task_id: int          # position in the suite's task order
    bddl_file: str        # absolute path
    init_states_file: str # absolute path

    @property
    def scene_file(self) -> str:
        """The BDDL file — what identifies the scene, as ``scene_file`` does on the other engines."""
        return self.bddl_file
