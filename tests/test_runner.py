"""Portable launch, failure cleanup, and aggregation contracts (no GPU needed)."""
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run
from scripts.common import write_json
from scripts.summarize import summarize


@pytest.mark.parametrize('levels', [['object'], ['room'], ['region'], ['instance'],
                                  ['object', 'room', 'region', 'instance']])
def test_cli_selects_any_single_goal_level(levels):
    args = run.parser().parse_args(['--levels', *levels, '--plan'])
    run.validate_args(args)
    command = run.evaluation_command(args, run.runtime_paths())
    assert command[command.index('--levels') + 1:command.index('--seed')] == levels


def test_two_model_sets_share_exactly_sixteen_agents(tmp_path):
    args = run.parser().parse_args(['--output', str(tmp_path), '--plan'])
    specs = run.service_specs(args, run.runtime_paths())
    assert args.workers // args.replicas == 16
    assert len(specs) == 8 and len({s['port'] for s in specs}) == 8
    for replica in [0, 1]:
        assert {s['name'] for s in specs if s['replica'] == replica} == {'grounding_dino', 'blip2itm', 'sam', 'yolov7'}
    for spec in specs:
        assert spec['environment']['CUDA_VISIBLE_DEVICES'] == '0'
        assert spec['environment']['OMP_NUM_THREADS'] == ('4' if spec['name'] == 'grounding_dino' else '1')


@pytest.mark.parametrize('options', [['--workers', '31'], ['--replicas', '0'],
                                   ['--levels', 'room', 'room'], ['--base-port', '65535'],
                                   ['--max-actions-per-goal', '501'], ['--limit-per-level', '0']])
def test_invalid_launch_is_rejected(options):
    with pytest.raises(ValueError):
        run.validate_args(run.parser().parse_args(options))


def test_plan_does_not_create_runtime_or_start_children(tmp_path, monkeypatch, capsys):
    output = tmp_path / 'never-created'
    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: pytest.fail('plan started a process'))
    assert run.run(run.parser().parse_args(['--output', str(output), '--plan'])) == 0
    assert not output.exists()
    assert 'grounding_dino' in capsys.readouterr().out


def test_process_cleanup_terminates_only_owned_child(tmp_path):
    owned = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)
    try:
        run.stop_process_groups([owned])
        assert owned.poll() is not None
        assert other.poll() is None
    finally:
        if other.poll() is None:
            os.killpg(other.pid, signal.SIGKILL)
        other.wait()


def test_missing_and_error_episodes_do_not_become_failures_or_complete(tmp_path):
    write_json(tmp_path / 'manifest.json', {'counts': {'room': 2}, 'smoke_test': False})
    write_json(tmp_path / 'episode-manifest.json', [{'episode_id': 'a', 'level': 'room'}, {'episode_id': 'b', 'level': 'room'}])
    write_json(tmp_path / 'episodes/a.json', {'episode_id': 'a', 'level': 'room', 'status': 'completed',
                                           'success': 1., 'spl': .5, 'steps': 10, 'seconds': 2.})
    write_json(tmp_path / 'errors/b.json', {'episode_id': 'b', 'status': 'error'})
    result = summarize(tmp_path)
    assert result['complete'] is False and result['missing'] == ['b'] and result['errors'] == ['b']
    assert result['levels']['room']['success_rate'] == 1.
    write_json(tmp_path / 'episodes/b.json', {'episode_id': 'b', 'level': 'room', 'status': 'completed',
                                           'success': 0., 'spl': 0., 'steps': 20, 'seconds': 5.})
    result = summarize(tmp_path)
    assert result['complete'] and not result['errors']
    assert result['levels']['room']['success_rate'] == .5
    assert result['levels']['room']['spl'] == .25
