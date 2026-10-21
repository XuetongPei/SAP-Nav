"""Validate the frozen annotation contents without redistributing the dataset."""
import gzip
import hashlib
import json
import os
from pathlib import Path
from scripts.common import ROOT, read_json


def annotation_digest(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        value = json.load(stream)
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def check_data():
    expected = read_json(ROOT / 'reference/annotations-sha256.json')
    root = Path(os.environ['EMBODIEDSCORE_DATA_ROOT']) / 'langmap/annotations'
    actual = {}
    for scene, digest in expected.items():
        plain = root / (scene + '.json')
        path = plain if plain.exists() else root / (scene + '.json.gz')
        actual[scene] = annotation_digest(path)
        if actual[scene] != digest:
            raise ValueError(f'Annotation content differs from the recorded experiment: {scene}')
    return actual


if __name__ == '__main__':
    print(json.dumps({'verified_annotation_scenes': len(check_data())}))
