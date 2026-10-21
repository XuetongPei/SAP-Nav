"""IsaacSceneRef — where a VLNverse scene lives on disk (pure data, layer L0).

A VLNverse scene is a folder ``<scene_root>/<scan>/`` holding the USD stage
(``start_result_navigation.usd`` in the Hugging Face release) with its
``Meshes/`` and ``Materials/``, plus ``freemap.npy``, the occupancy grid the
kinematics move on (row 0 = world-x of every column, column 0 = world-y of
every row, cells 0 obstacle / 1 reachable / 2 out of bounds).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

USD_CANDIDATES = ("start_result_navigation.usd", "start_result_navigation.usda", "{scan}.usd", "{scan}.usda")


@dataclass(frozen=True)
class IsaacSceneRef:
    scan: str            # e.g. "kujiale_0272"
    scene_dir: str       # absolute path of the scene folder

    @property
    def scene_file(self) -> str:
        """The USD stage — resolved when asked for, so loading a split does not
        touch thousands of folders."""
        for name in USD_CANDIDATES:
            p = os.path.join(self.scene_dir, name.format(scan=self.scan))
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"no USD stage for {self.scan} in {self.scene_dir} (tried {USD_CANDIDATES})")

    @property
    def freemap_file(self) -> str:
        return os.path.join(self.scene_dir, "freemap.npy")
