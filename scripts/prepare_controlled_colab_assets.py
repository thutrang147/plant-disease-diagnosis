"""Export only canonical train/dev/test rows and Lab baselines for private Colab use.

Reads existing local HF Arrow caches, never downloads data or trains a model.
The output must remain private and is excluded from Git under outputs/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets
from PIL import Image
from scipy.fftpack import dct

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.controlled_field_mixed import CHECKPOINT_NAMES, EXPECTED_INITIAL, SPLITS, atomic_json, sha256


def build_bundle(target, repo=ROOT):
    """Package only Notebook 15 code/docs and canonical splits; no images/weights."""
    repo, target = Path(repo), Path(target)
    names = ['src/controlled_field_mixed.py',
             'notebooks/15_controlled_field_mixed_finetuning.ipynb',
             'docs/notebook15_colab_guide.md', 'data/splits/label_map.json']
    names += [f'data/splits/{name}.csv' for name in SPLITS]
    payloads = {name: (repo / name).read_bytes() for name in names}
    payloads['bundle_manifest.json'] = json.dumps({
        'file_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        'purpose': 'Offline Notebook 15 code and canonical splits; no third-party images.'
    }, indent=2).encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in payloads.items():
            bundle.writestr(name, payload)
    print('Code bundle:', target, flush=True)
    return target


def phash(image):
    gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dct(dct(gray, axis=0), axis=1)[:8, :8]
    value = 0
    for bit in (low > np.median(low)).flatten():
        value = (value << 1) | int(bit)
    return value


def pixel_hash(image):
    import hashlib
    return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()


def export_assets(output, repo=ROOT, cache_root=None, architectures=('resnet18', 'vit_small')):
    output, repo = Path(output), Path(repo)
    output.mkdir(parents=True, exist_ok=False)
    cache_root = Path(cache_root) if cache_root else Path.home()/'.cache/huggingface/datasets'
    frames = {name: pd.read_csv(repo/f'data/splits/{name}.csv') for name in SPLITS}
    manifest = {'schema_version': 1, 'split_sha256': {name: sha256(repo/f'data/splits/{name}.csv') for name in SPLITS},
                'sources': {}, 'checkpoint_sha256': {}, 'images': [],
                'privacy': 'Private Colab assets; never commit or redistribute source images.'}
    for source in ('plantdoc', 'plantvillage'):
        cache_name = f'geraldmc___{source}-full'
        candidates = sorted((cache_root/cache_name).glob('default/*/*'))
        candidates = [p for p in candidates if list(p.glob('*.arrow')) and (p/'dataset_info.json').exists()]
        if len(candidates) != 1:
            raise FileNotFoundError(f'Expected one recorded {source} Arrow cache under {cache_root/cache_name}; found {len(candidates)}. Supply --cache-root pointing to the verified cache.')
        directory = candidates[0]
        shards = sorted(directory.glob('*.arrow'))
        dataset = concatenate_datasets([Dataset.from_file(str(path)) for path in shards])
        requested = pd.concat([f[f.source.eq(source)] for f in frames.values()], ignore_index=True)
        assert not requested.duplicated(['source', 'row_id']).any()
        rows = sorted(requested.row_id.astype(int).tolist())
        indexed = requested.set_index('row_id')
        for row_id in rows:
            record = dataset[row_id]
            row = indexed.loc[row_id]
            assert record['class_label'] == row.original_class
            image = record['image'].convert('RGB')
            exact, perceptual = pixel_hash(image), phash(image)
            if 'exact_hash' in row and pd.notna(row['exact_hash']):
                assert exact == row.exact_hash, (source, row_id, 'stored exact hash')
            if 'phash' in row and pd.notna(row['phash']):
                assert f'{perceptual:016x}' == row.phash
            split = next(name for name, frame in frames.items() if ((frame.source==source)&(frame.row_id==row_id)).any())
            manifest['images'].append({'source': source, 'row_id': row_id, 'split': split,
                                       'label_id': int(row.label_id), 'pixel_sha256': exact, 'phash': f'{perceptual:016x}'})
        subset = dataset.select(rows)
        target = output/'datasets'/source
        subset.save_to_disk(str(target))
        manifest['sources'][source] = {'repository': f'geraldmc/{source}-full', 'cached_rows': len(dataset),
                                      'exported_rows': len(subset), 'source_row_ids': rows, 'cache_key': directory.name,
                                      'cache_arrow_sha256': {p.name: sha256(p) for p in shards},
                                      'limitation': 'Offline original row/class checks; immutable upstream revision not newly resolved.'}
        print(f'{source}: exported {len(subset)} original rows', flush=True)
    records = manifest['images']
    training = [r for r in records if r['split'] in ('field_adapt_train', 'lab_train')]
    heldout = [r for r in records if r['split'] in ('field_dev', 'field_test')]
    matches = []
    for a in training:
        for b in heldout:
            distance = (int(a['phash'], 16)^int(b['phash'], 16)).bit_count()
            if a['pixel_sha256'] == b['pixel_sha256'] or distance <= 4:
                matches.append({'train_source': a['source'], 'train_row': a['row_id'], 'heldout_split': b['split'],
                                'heldout_row': b['row_id'], 'phash_distance': distance,
                                'exact_match': a['pixel_sha256'] == b['pixel_sha256']})
    manifest['leakage_screen'] = {'threshold': 4, 'training_rows': len(training), 'heldout_rows': len(heldout),
                                  'heldout_exact_or_phash_matches': matches,
                                  'scope': 'Adaptation and Lab Train versus Field Dev/Test, exact RGB pixels and DCT pHash. Not proof of physical independence or pretraining-source separation.'}
    (output/'checkpoints').mkdir()
    for architecture in architectures:
        source = repo/'models'/CHECKPOINT_NAMES[architecture]
        if not source.is_file() or sha256(source) != EXPECTED_INITIAL[architecture]:
            raise ValueError(f'Missing or noncanonical baseline: {source}; no weight substitution is allowed.')
        target = output/'checkpoints'/source.name
        shutil.copy2(source, target)
        manifest['checkpoint_sha256'][architecture] = sha256(target)
    manifest['file_sha256'] = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
    atomic_json(output/'asset_manifest.json', manifest)
    print('Content-screen matches requiring review:',len(matches),flush=True)
    print('Private asset directory:',output,flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bundle-output', type=Path)
    parser.add_argument('--bundle-only', action='store_true', help='Reuse the already prepared private assets; only package code/splits.')
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--resnet-only', action='store_true')
    args = parser.parse_args()
    if not args.bundle_only:
        if args.output is None:
            parser.error('--output is required unless --bundle-only is used')
        export_assets(args.output, cache_root=args.cache_root,
                      architectures=('resnet18',) if args.resnet_only else ('resnet18', 'vit_small'))
    if args.bundle_output:
        build_bundle(args.bundle_output)
    elif args.bundle_only:
        parser.error('--bundle-only requires --bundle-output')
