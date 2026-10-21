"""Full LangMap evaluation of the upstream VLFM policy with resumable records.

This is an adaptation, not a released LangMap baseline. It uses VLFM's RGB-D
LoCoBot rig and agent dimensions, with LangMap's navmesh construction.
ActiveSpin's category/prompt interface is used. Goal geometry is evaluator-only.
SR and SPL follow bo-miao/LangMap run.py and run_seq.py (terminal distance < .25m,
including time limits; 500 actions per single-goal episode).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from dataclasses import asdict
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
import quaternion
from embodiedscore_envs.benchmarks.langmap import load_level
from embodiedscore_envs.benchmarks.env.schema import targets_of
from scripts.langmap_world import EVAL_BODY, VLFMWorld, navmesh_protocol
from scripts.langmap_goal_adapter import ACTIVESPIN_COMMIT, goal_attributes
from scripts.langmap_shortest_path import ShortestPathLocalPolicy

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("object", "room", "region", "instance")


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def load_episodes(level, split):
    return load_level(level, split)


def proprioception(position, rotation, start_position, start_rotation):
    q0 = np.quaternion(start_rotation[3], *start_rotation[:3])
    q = np.quaternion(rotation[3], *rotation[:3])
    delta = np.asarray(position) - np.asarray(start_position)
    local = (q0.inverse() * np.quaternion(0, *delta) * q0).imag
    forward = (q * np.quaternion(0, 0, 0, -1) * q.inverse()).imag
    rel_forward = (q0.inverse() * np.quaternion(0, *forward) * q0).imag
    return {
        "gps": np.array([-local[2], local[0]], dtype=np.float32),
        "compass": np.array([math.atan2(-rel_forward[0], -rel_forward[2])], dtype=np.float32),
        "heading": np.array([math.atan2(-forward[0], -forward[2])], dtype=np.float32),
    }


def scores(start_distance, final_distance, length):
    success = float(final_distance < 0.25)
    denom = max(start_distance, length)
    spl = start_distance / denom if success and math.isfinite(start_distance) and denom > 0 else 0.0
    if success and start_distance == 0 and length == 0:
        spl = 1.0
    return success, spl


class PolicyProcess:
    def __init__(self, output, worker):
        self.socket = f"/tmp/vlfm-langmap-{os.getpid()}-{worker}.sock"
        self.log = open(output / f"policy-{worker}.log", "a", buffering=1)
        self.proc = subprocess.Popen(
            [os.environ["VLFM_PYTHON"], "-u", str(ROOT / "scripts/langmap_policy_worker.py"), "--socket", self.socket],
            cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 180
        while not Path(self.socket).exists():
            if self.proc.poll() is not None:
                raise RuntimeError(f"Policy worker exited {self.proc.returncode}; see {self.log.name}")
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError("Policy startup timed out")
            time.sleep(0.2)
        self.conn = Client(self.socket, family="AF_UNIX")
        if not self.conn.poll(180):
            raise TimeoutError("Policy ready message timed out")
        assert self.conn.recv().get("ready")

    def act(self, **request):
        self.conn.send(request)
        local_result = None
        while True:
            if not self.conn.poll(300):
                raise TimeoutError("Policy action timed out after 300s")
            response = self.conn.recv()
            if response.get("op") != "shortest_path":
                break
            local_result = self.local_planner.act(response["goal_xy"], response["stop"])
            # Only the action is returned to VLFM; geometry stays in the simulator.
            self.conn.send({"action": local_result["action"]})
        if "error" in response:
            raise RuntimeError(response["error"])
        if response["action"] not in range(4):
            raise ValueError(f"Invalid policy action: {response}")
        if local_result is not None:
            if response["action"] != local_result["action"]:
                raise ValueError("Policy did not use the shortest-path action")
            response.update(local_result)
        return response

    def set_episode_frame(self, world, start_position, start_rotation):
        self.local_planner = ShortestPathLocalPolicy(world, start_position, start_rotation)

    def close(self):
        if hasattr(self, "conn"):
            try:
                self.conn.send({"op": "close"})
                self.conn.close()
            except (OSError, EOFError):
                pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
        Path(self.socket).unlink(missing_ok=True)


def evaluate(world, policy, ep, level, output, worker, seed, max_actions=500):
    started = time.monotonic()
    world.load_scene(ep.scene)
    world.place(ep.start_position, ep.start_rotation)
    goals = (ep.goal,)
    results, trajectory = [], []
    for goal_index, goal in enumerate(goals):
        attributes = goal_attributes(ep, level)
        text = attributes["instruction"]
        start_pos, start_rot = world.pose()
        policy.set_episode_frame(world, start_pos, start_rot)
        targets = targets_of(goal)
        d0 = world.geodesic(start_pos, targets)
        length, prev_pos = 0.0, start_pos
        step_seed = seed + int(hashlib.sha256(f"{ep.episode_id}:{goal_index}".encode()).hexdigest()[:8], 16) % (2**31)
        for step in range(max_actions):
            obs = world.observe()
            pos, rot = world.pose()
            normalized = np.clip(obs["depth"], 0.5, 5.0)
            normalized = ((normalized - 0.5) / 4.5).astype(np.float32)
            inputs = {"rgb": obs["rgb"], "depth": normalized,
                      **proprioception(pos, rot, start_pos, start_rot)}
            reply = policy.act(observations=inputs, goal_attributes=attributes, first=step == 0, seed=step_seed)
            action = reply["action"]
            collided = world.move(action) if action else False
            pos, rot = world.pose()
            length += float(np.linalg.norm(np.asarray(pos, dtype=np.float32) - np.asarray(prev_pos, dtype=np.float32)))
            prev_pos = pos
            trajectory.append({"goal": goal_index, "step": step + 1, "action": action,
                               "position": pos.tolist(), "rotation": rot.tolist(), "collided": bool(collided), **reply})
            if step % 25 == 0:
                atomic_json(output / f"progress-{worker}.json", {"episode_id": ep.episode_id, "level": level,
                            "goal": goal_index, "step": step + 1, "updated_at": time.time()})
            if action == 0:
                break
        final_pos, final_rot = world.pose()
        df = world.geodesic(final_pos, targets)
        success, spl = scores(d0, df, length)
        results.append({"goal_index": goal_index, "instruction": text, "goal_attributes": attributes,
                        "dataset_instruction": ep.instruction, "success": success, "spl": spl,
                        "start_distance": d0 if math.isfinite(d0) else None,
                        "final_distance": df if math.isfinite(df) else None,
                        "path_length": length, "steps": step + 1,
                        "end_reason": "stop" if action == 0 else "step_limit",
                        "start_position": start_pos.tolist(), "final_position": final_pos.tolist()})
    result = {"episode_id": ep.episode_id, "level": level, "status": "completed", "subgoals": results,
              "success": float(np.mean([r["success"] for r in results])),
              "spl": float(np.mean([r["spl"] for r in results])),
              "all_goals_success": float(all(r["success"] for r in results)),
              "steps": sum(r["steps"] for r in results), "seconds": time.monotonic() - started}
    atomic_json(output / "trajectories" / f"{ep.episode_id}.json", trajectory)
    atomic_json(output / "episodes" / f"{ep.episode_id}.json", result)
    return result


def main():
    from scripts.common import UPSTREAM_COMMIT, read_json, source_hashes, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--levels', nargs='+', choices=('object', 'room', 'region', 'instance'), default=['object', 'room'])
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--max-actions-per-goal', type=int, default=500)
    parser.add_argument('--limit-per-level', type=int, help='Smoke test only; changes the evaluation manifest')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_actions_per_goal <= 500:
        parser.error('--max-actions-per-goal must be in 1..500')
    if not 0 <= args.worker < args.workers:
        parser.error('require 0 <= worker < workers')
    if len(set(args.levels)) != len(args.levels):
        parser.error('duplicate levels')
    if args.limit_per_level is not None and args.limit_per_level < 1:
        parser.error('--limit-per-level must be positive')
    output = args.output.resolve()
    from scripts.check_data import check_data
    annotation_hashes = check_data()
    tasks, counts, full_counts = [], {}, {}
    for level in args.levels:
        episodes = load_episodes(level, 'val')
        expected = read_json(ROOT / f'reference/{level}-episodes.json')
        actual = [ep.episode_id for ep in episodes]
        if actual != expected:
            raise ValueError(f'{level}: loader output differs from the frozen episode IDs/order; check annotations')
        full_counts[level] = len(episodes)
        if args.limit_per_level is not None:
            episodes = episodes[:args.limit_per_level]
        counts[level] = len(episodes)
        tasks.extend((level, ep) for ep in episodes)
    missing = sorted({ep.scene.scene_file for _, ep in tasks if not Path(ep.scene.scene_file).is_file()})
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} scenes; first entries: {missing[:5]}')
    attributes = [(ep.episode_id, goal_attributes(ep, level)) for level, ep in tasks]
    manifest = {
        'upstream_commit': UPSTREAM_COMMIT, 'split': 'val', 'levels': args.levels,
        'annotation_sha256': annotation_hashes,
        'counts': counts, 'full_val_counts': full_counts, 'selected_count': len(tasks),
        'selected_ids_sha256': hashlib.sha256('\n'.join(ep.episode_id for _, ep in tasks).encode()).hexdigest(),
        'body': asdict(EVAL_BODY), 'seed': args.seed, 'max_actions_per_goal': args.max_actions_per_goal,
        'standard_action_budget': args.max_actions_per_goal == 500,
        'smoke_test': args.limit_per_level is not None or args.max_actions_per_goal != 500,
        'success_distance_m': 0.25, 'success_comparison': 'strictly_less_than',
        'evaluation_scope': 'single_goal_selected_levels',
        'goal_adapter_commit': ACTIVESPIN_COMMIT,
        'goal_attributes_sha256': hashlib.sha256(json.dumps(attributes, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        'navmesh_protocol': navmesh_protocol(), 'local_policy': 'habitat_sim_shortest_path_follower',
        'pointnav_model_loaded': False, 'local_waypoint_radius_m': 0.25, 'object_stop_radius_m': 0.9,
        'local_policy_uses_navmesh_geometry': True, 'local_policy_uses_ground_truth_goal': False,
        'policy_inputs': ['rgb', 'normalized_depth', 'gps', 'compass', 'heading', 'goal_attributes'],
        'perception_model_profile': read_json(ROOT / 'configs/models.json'),
    }
    manifest = json.loads(json.dumps(manifest))
    code = source_hashes()
    # Validate before writing anything; completed runs cannot silently mix protocols.
    if (output / 'manifest.json').exists() and read_json(output / 'manifest.json') != manifest:
        raise ValueError('Manifest differs; choose a fresh output directory')
    if (output / 'code-sha256.json').exists() and read_json(output / 'code-sha256.json') != code:
        raise ValueError('Source changed; choose a fresh output directory')
    for folder in ['episodes', 'trajectories', 'errors']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    if args.worker == 0:
        write_json(output / 'manifest.json', manifest)
        write_json(output / 'episode-manifest.json', [
            {'level': level, 'scene': ep.info['langmap_scene'], 'episode_id': ep.episode_id}
            for level, ep in tasks])
        write_json(output / 'code-sha256.json', code)
    print(json.dumps({'counts': counts, 'selected': len(tasks), 'worker': args.worker,
                      'smoke_test': manifest['smoke_test']}), flush=True)
    if args.preflight:
        return
    world = VLFMWorld(EVAL_BODY, gpu_id=0, seed=args.seed)
    policy, failed, status = None, 0, 'error'
    try:
        policy = PolicyProcess(output, args.worker)
        for index, (level, ep) in enumerate(tasks):
            if index % args.workers != args.worker:
                continue
            if (output / 'episodes' / f'{ep.episode_id}.json').exists():
                continue
            try:
                result = evaluate(world, policy, ep, level, output, args.worker, args.seed, args.max_actions_per_goal)
                (output / 'errors' / f'{ep.episode_id}.json').unlink(missing_ok=True)
                print(json.dumps(result), flush=True)
            except Exception:
                error = {'episode_id': ep.episode_id, 'status': 'error', 'traceback': traceback.format_exc()}
                write_json(output / 'errors' / f'{ep.episode_id}.json', error)
                failed += 1
                raise
        status = 'completed'
    finally:
        world.close()
        if policy:
            policy.close()
        write_json(output / f'worker-{args.worker}-exit.json',
                   {'finished_at': time.time(), 'status': status, 'failed': failed})


if __name__ == '__main__':
    main()
