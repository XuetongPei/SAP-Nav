"""Fetch exact external source revisions; never reset an existing checkout."""
import argparse
import subprocess
from pathlib import Path
from scripts.common import ROOT, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'external')
    args = parser.parse_args()
    for name, spec in read_json(ROOT / 'configs/dependencies.json').items():
        destination = args.directory / name
        if destination.exists():
            head = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
            if head != spec['commit']:
                raise RuntimeError(f'{destination}: expected {spec["commit"]}, found {head}; use a fresh directory')
            changed = subprocess.check_output(['git', '-C', str(destination), 'diff', 'HEAD', '--stat'], text=True)
            if changed:
                raise RuntimeError(f'Locally modified dependency: {destination}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'remote', 'add', 'origin', spec['repository']], check=True)
        subprocess.run(['git', '-C', str(destination), 'fetch', '--depth=1', 'origin', spec['commit']], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        print(f'{name}: {spec["commit"]}', flush=True)


if __name__ == '__main__':
    main()
