"""Own the entire evaluation lifecycle: preflight, model services, workers, summary."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from pathlib import Path

from scripts.common import ROOT, base_environment, read_json, runtime_paths, write_json
from scripts.download_models import verify_models
from scripts.summarize import summarize


def parser():
    defaults = read_json(ROOT / 'configs/baseline.json')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=runtime_paths()['runtime'] / 'runs/object-room')
    p.add_argument('--levels', nargs='+', choices=['object', 'room', 'region', 'instance'], default=defaults['levels'])
    p.add_argument('--workers', type=int, default=defaults['workers'])
    p.add_argument('--replicas', type=int, default=defaults['replicas'])
    p.add_argument('--dino-threads', type=int, default=defaults['dino_threads'], choices=[1, 2, 4, 8])
    p.add_argument('--base-port', type=int, default=defaults['base_port'])
    p.add_argument('--gpu', default=defaults['gpu'], help='Physical GPU index/UUID exposed to all child processes')
    p.add_argument('--seed', type=int, default=defaults['seed'])
    p.add_argument('--max-actions-per-goal', type=int, default=defaults['max_actions_per_goal'])
    p.add_argument('--limit-per-level', type=int, help='Smoke test only')
    p.add_argument('--preflight', action='store_true', help='Validate data, frozen episode IDs and manifest; do not start models')
    p.add_argument('--plan', action='store_true', help='Print commands and topology without starting processes or writing output')
    return p


def validate_args(args):
    if args.workers < 1 or args.replicas < 1 or args.workers % args.replicas:
        raise ValueError('workers must be positive and divisible by replicas')
    if not args.levels or len(set(args.levels)) != len(args.levels):
        raise ValueError('levels must be nonempty and unique')
    if any(level not in ['object', 'room', 'region', 'instance'] for level in args.levels):
        raise ValueError('unsupported level')
    if not 1024 <= args.base_port <= 65532 - (args.replicas - 1) * 10:
        raise ValueError('service port range must fit in 1024..65535')
    if not 1 <= args.max_actions_per_goal <= 500:
        raise ValueError('action budget must be in 1..500')
    if args.limit_per_level is not None and args.limit_per_level < 1:
        raise ValueError('limit-per-level must be positive')


def evaluation_command(args, paths):
    command = [paths['sim_python'], '-u', '-m', 'scripts.eval_langmap', '--output', str(args.output),
               '--levels', *args.levels, '--seed', str(args.seed),
               '--max-actions-per-goal', str(args.max_actions_per_goal)]
    if args.limit_per_level is not None:
        command += ['--limit-per-level', str(args.limit_per_level)]
    return command


def service_specs(args, paths):
    specs = []
    for replica in range(args.replicas):
        port = args.base_port + 10 * replica
        for offset, name in enumerate(['grounding_dino', 'blip2itm', 'sam', 'yolov7']):
            metadata = args.output / 'services' / f'replica-{replica}-{name}.json'
            env = base_environment(paths, args.gpu)
            # Dependencies/tokenizer are prepared explicitly, never select a new model revision here.
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            if name == 'grounding_dino':
                cmd = [paths['vlfm_python'], '-u', '-m', 'scripts.serve_langmap_dino_threads',
                       '--checkpoint', str(paths['models'] / 'groundingdino_swinb_cogcoor.pth'),
                       '--cpu-threads', str(args.dino_threads), '--metadata', str(metadata)]
                env.update({k: str(args.dino_threads) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']})
            elif name == 'sam':
                cmd = [paths['sam_python'], '-u', '-m', 'scripts.serve_langmap_perception', '--model', 'sam21_large',
                       '--checkpoint', str(paths['models'] / 'sam2.1_hiera_large.pt'), '--metadata', str(metadata)]
                env['PYTHONPATH'] = str(paths['sam_source']) + os.pathsep + env['PYTHONPATH']
            else:
                cmd = [paths['vlfm_python'], '-u', '-m', 'scripts.serve_auxiliary',
                       '--model', name, '--models', str(paths['models'])]
            cmd += ['--port', str(port + offset)]
            specs.append({'replica': replica, 'name': name, 'port': port + offset,
                          'command': cmd, 'environment': env, 'metadata': str(metadata)})
    return specs


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def stop_process_groups(processes):
    # Only groups created by this invocation are signalled, even if a group leader exited.
    for proc in reversed(processes):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for proc in reversed(processes):
        try:
            proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for proc in reversed(processes):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def start_process(command, environment, log_path, processes):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    processes.append(proc)
    return proc


def wait_service(proc, port, processes, timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('A model service exited during startup; inspect services/*.log')
        if port_open(port):
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f'Service on port {port} did not become ready')
        time.sleep(0.25)


def run(args):
    validate_args(args)
    args.output = args.output.resolve()
    paths = runtime_paths()
    command = evaluation_command(args, paths)
    specs = service_specs(args, paths)
    env = base_environment(paths, args.gpu)
    if args.plan:
        print(json.dumps({'preflight': command + ['--preflight'], 'workers': args.workers,
                          'replicas': args.replicas, 'agents_per_replica': args.workers // args.replicas,
                          'services': [{k: v for k, v in s.items() if k != 'environment'} for s in specs]}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run(command + ['--preflight'], cwd=ROOT, env=env, check=True)
        if args.preflight:
            return 0
        if summarize(args.output)['complete']:
            print('All declared episodes are already complete; no model services started.')
            return 0
        verify_models(paths['models'])
        for spec in specs:
            if port_open(spec['port']):
                raise RuntimeError(f'Port {spec["port"]} is occupied; choose another --base-port')
        for python, expected in [(paths['vlfm_python'], '1.12.1+cu113'), (paths['sam_python'], '2.10.0+cu128')]:
            check = ('import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; '
                     f'assert torch.__version__ == {expected!r}, (torch.__version__, {expected!r})')
            subprocess.run([python, '-c', check],
                           env=env, cwd=ROOT, check=True)
        processes, services, workers = [], [], []
        state = {'status': 'starting', 'started_at': time.time(), 'workers': args.workers,
                 'replicas': args.replicas, 'agents_per_replica': args.workers // args.replicas,
                 'gpu': args.gpu, 'dino_threads': args.dino_threads, 'levels': args.levels,
                 'services': [], 'failure': None}
        write_json(args.output / 'deployment.json', state)
        old_handlers = {}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        for sig in [signal.SIGINT, signal.SIGTERM]:
            old_handlers[sig] = signal.signal(sig, interrupted)
        code = 1
        try:
            for spec in specs:
                log = args.output / 'services' / f'replica-{spec["replica"]}-{spec["name"]}.log'
                proc = start_process(spec['command'], spec['environment'], log, processes)
                services.append(proc)
                wait_service(proc, spec['port'], services)
                if spec['name'] in ['grounding_dino', 'sam']:
                    metadata = read_json(spec['metadata'])
                    expected = read_json(ROOT / 'configs/models.json')['models'][spec['name']]
                    if metadata['checkpoint_sha256'] != expected['sha256'] or metadata['dtype'] != 'torch.float32':
                        raise ValueError(f'Unexpected perception profile: {metadata}')
                state['services'].append({'replica': spec['replica'], 'name': spec['name'],
                                          'pid': proc.pid, 'port': spec['port']})
                write_json(args.output / 'deployment.json', state)
                print(f'Ready: replica {spec["replica"]} {spec["name"]} port {spec["port"]}', flush=True)
            state.update(status='running', evaluation_started_at=time.time())
            write_json(args.output / 'deployment.json', state)
            for worker in range(args.workers):
                replica = worker // (args.workers // args.replicas)
                offset = args.base_port + replica * 10
                worker_env = dict(env)
                worker_env.update(dict(zip(['GROUNDING_DINO_PORT', 'BLIP2ITM_PORT', 'SAM_PORT', 'YOLOV7_PORT'],
                                           map(str, range(offset, offset + 4)))))
                proc = start_process(command + ['--worker', str(worker), '--workers', str(args.workers)],
                                     worker_env, args.output / f'worker-{worker}.log', processes)
                workers.append(proc)
                time.sleep(1)
            while True:
                if any(p.poll() is not None for p in services):
                    raise RuntimeError('A model service exited during evaluation')
                if any(p.poll() not in [None, 0] for p in workers):
                    raise RuntimeError('An evaluation worker failed; inspect worker logs/errors')
                if all(p.poll() == 0 for p in workers):
                    break
                time.sleep(1)
            result = summarize(args.output)
            if not result['complete']:
                raise RuntimeError('Workers exited but declared episodes are incomplete')
            state['status'], code = 'complete', 0
            print(json.dumps(result, indent=2), flush=True)
        except BaseException:
            state.update(status='stopped_incomplete', failure=traceback.format_exc())
            print(state['failure'], flush=True)
        finally:
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            stop_process_groups(processes)
            state.update(stopped_at=time.time(), model_services_stopped=True)
            write_json(args.output / 'deployment.json', state)
            summarize(args.output)
            (args.output / 'exit-code').write_text(f'{code}\n')
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return code


def main():
    raise SystemExit(run(parser().parse_args()))


if __name__ == '__main__':
    main()
