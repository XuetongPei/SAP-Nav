"""Download the five checkpoint files and verify the recorded SHA-256 values."""
import argparse
import os
import urllib.request
from pathlib import Path
from scripts.common import ROOT, read_json, runtime_paths, sha256


def verify_models(directory):
    for spec in read_json(ROOT / 'configs/models.json')['models'].values():
        path = Path(directory) / spec['filename']
        if not path.is_file() or sha256(path) != spec['sha256']:
            raise ValueError(f'Missing or incorrect checkpoint: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=runtime_paths()['models'])
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--bert', action='store_true', help='Cache the pinned BERT tokenizer/config for DINO and BLIP-2')
    args = parser.parse_args()
    profile = read_json(ROOT / 'configs/models.json')
    if not args.verify_only:
        args.directory.mkdir(parents=True, exist_ok=True)
        for spec in profile['models'].values():
            path = args.directory / spec['filename']
            if path.exists():
                if sha256(path) != spec['sha256']:
                    raise ValueError(f'Existing file has wrong hash: {path}; move it aside explicitly')
                continue
            temp = path.with_suffix(path.suffix + '.part')
            print(f'Downloading {spec["filename"]}', flush=True)
            urllib.request.urlretrieve(spec['url'], temp)
            if sha256(temp) != spec['sha256']:
                raise ValueError(f'Download hash mismatch: {temp}')
            temp.replace(path)
    verify_models(args.directory)
    if args.bert:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(
            **profile['bert'], local_files_only=args.verify_only,
            allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt',
                            'special_tokens_map.json', 'pytorch_model.bin', 'model.safetensors']))
        # Upstream LAVIS/DINO request the default revision internally. Pin that cache
        # reference too, then inference runs offline and cannot select another revision.
        ref = snapshot.parent.parent / 'refs/main'
        if ref.exists() and ref.read_text().strip() != profile['bert']['revision']:
            raise ValueError(f'BERT cache points to a different revision: {ref}; use a fresh HF_HOME')
        if args.verify_only:
            if not ref.exists():
                raise ValueError('Pinned BERT cache ref missing; run without --verify-only first')
        else:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(profile['bert']['revision'])
    print('Checkpoint hashes verified.')


if __name__ == '__main__':
    main()
