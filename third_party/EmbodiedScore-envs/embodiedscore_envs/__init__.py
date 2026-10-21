"""EmbodiedScore environments — every benchmark of the workspace behind
Gymnasium 1.3: the habitat lines on one frozen habitat-sim 0.3.3
(EmbodiedScore-habitat), VLNverse on Isaac Sim 5.1 through its render worker,
the LIBERO manipulation lines on robosuite 1.4 / MuJoCo in process, the
RoboTwin bimanual manipulation lines on SAPIEN 3 in process, the RoboCasa /
RoboCasa365 kitchen lines on robosuite 1.5 / MuJoCo in process, the CALVIN
chain line on pybullet in process.

    import embodiedscore_envs as es
    env = es.make("vlnce-r2r", "val_unseen")          # the standard stack
    obs, info = env.reset(options={"episode": 0})
    obs, r, terminated, truncated, info = env.step(es.Act.FORWARD)
    info["metrics"]

Gymnasium ids (``gym.make(id, split=...)`` gives the bare body) are registered
on import, one per benchmark declaration in ``embodiedscore_envs.benchmarks``.
The Isaac, LIBERO, RoboTwin, RoboCasa, CALVIN and BEHAVIOR lines register ``nondeterministic=True``: the
episode, the poses and every fact in ``info`` are reproducible under a seed; the
rendered frames are not bit-identical from one render to the next (Isaac's RTX,
robosuite's offscreen EGL renderer and SAPIEN's ray tracer alike).
"""

from gymnasium.envs.registration import register

from .benchmarks import BENCHMARKS, benchmark, build_env, make
from .benchmarks.env import Act

__version__ = "0.0.1"

for _b in BENCHMARKS.values():
    register(id=_b.gym_id, entry_point="embodiedscore_envs.benchmarks:build_env",
             max_episode_steps=_b.max_episode_steps,
             nondeterministic=_b.engine in ("isaac", "libero", "robotwin", "robocasa", "calvin", "behavior"),
             kwargs={"benchmark_name": _b.name})
del _b

__all__ = ["BENCHMARKS", "benchmark", "build_env", "make", "Act", "__version__"]
