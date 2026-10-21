"""Aggregate exactly the declared episodes; incomplete runs are never successful."""
import argparse
from pathlib import Path
from scripts.common import read_json, write_json


def summarize(output):
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    entries = read_json(output / 'episode-manifest.json')
    expected = {r['episode_id'] for r in entries}
    rows = [read_json(p) for p in sorted((output / 'episodes').glob('*.json'))]
    done = {r['episode_id'] for r in rows}
    errors = [p.stem for p in (output / 'errors').glob('*.json') if p.stem not in done]
    malformed = [r.get('episode_id') for r in rows if r.get('status') != 'completed']
    summary = {'completed': len(done & expected), 'expected': len(expected),
               'missing': sorted(expected - done), 'unexpected_ids': sorted(done - expected),
               'errors': errors, 'invalid_results': malformed, 'smoke_test': manifest['smoke_test'],
               'complete': done == expected and len(rows) == len(done) and not errors and not malformed,
               'levels': {}}
    for level, count in manifest['counts'].items():
        selected = [r for r in rows if r['level'] == level and r['episode_id'] in expected]
        n = len(selected)
        summary['levels'][level] = {
            'completed': n, 'expected': count,
            'success_rate': sum(r['success'] for r in selected) / n if n else None,
            'spl': sum(r['spl'] for r in selected) / n if n else None,
            'total_actions': sum(r['steps'] for r in selected),
            'cumulative_episode_seconds': sum(r['seconds'] for r in selected)}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
