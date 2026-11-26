"""Frozen-checkpoint crop/coverage sensitivity; no training or model selection.

Reuses the four strategies in upstream caafa06, with explicit EXIF handling,
per-image and per-view provenance. The square-448 strategy still forwards 224.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
import timm

CLASS_NAMES = ['Healthy', 'Early Blight', 'Late Blight', 'Bacterial Spot', 'Septoria Leaf Spot']
MODEL_NAMES = {
    'resnet_lab': 'ResNet18 · Lab baseline', 'vit_lab': 'ViT-Small · Lab baseline',
    'resnet_mixed': 'ResNet18 · Geometric + Mixed', 'vit_mixed': 'ViT-Small · Mixed',
}
REGIMES = ('1.00x_full_scene', '0.70x_center_crop', '0.50x_closeup_crop', 'highres_448_5crop')
FIVE_BOXES = ((0, 0, 224, 224), (224, 0, 448, 224), (0, 224, 224, 448),
              (224, 224, 448, 448), (112, 112, 336, 336))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def load_model(root, key, reference):
    """Strictly load the same frozen checkpoints as Notebook 11."""
    provenance = reference['checkpoints_and_preprocessing'][key]
    # Historical absolute paths are resolved to the local models directory.
    checkpoint = root / 'models' / Path(provenance['checkpoint']).name
    source_path = root / 'models' / Path(provenance['source_checkpoint']).name
    for path, field in ((checkpoint, 'checkpoint_sha256'), (source_path, 'source_checkpoint_sha256')):
        assert sha256_file(path) == provenance[field], (key, field)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    for record in (state, source):
        assert list(record['class_names']) == CLASS_NAMES
        assert record['architecture'] == provenance['architecture']
    if key.endswith('mixed'):
        assert state['adaptation_type'] == 'balanced_mixed_domain'
        assert state['source_checkpoint_sha256'] == provenance['source_checkpoint_sha256']
        assert state['experiment'] == {'resnet_mixed': 'R2_resnet_geometric', 'vit_mixed': 'V1_vit_small'}[key]
        assert state['field_dev_sha256'] == sha256_file(root / 'data/splits/field_dev.csv')
        assert state['manifest_sha256']['lab_val'] == sha256_file(root / 'data/splits/lab_val.csv')
    if key.startswith('resnet'):
        assert source['pretrained_weights'] == 'IMAGENET1K_V1'
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
        standard = ResNet18_Weights.IMAGENET1K_V1.transforms()
        mean, std = standard.mean, standard.std
    else:
        assert source['pretrained_weights'] == 'augreg_in1k'
        config = source['preprocessing_config']
        assert tuple(config['input_size']) == (3, 224, 224)
        model = timm.create_model(provenance['architecture'], pretrained=False, num_classes=5)
        standard = timm.data.create_transform(**config, is_training=False)
        mean, std = config['mean'], config['std']
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return model, standard, normalize, {
        'checkpoint': str(checkpoint.relative_to(root)), 'checkpoint_sha256': provenance['checkpoint_sha256'],
        'source_checkpoint': str(source_path.relative_to(root)),
        'source_checkpoint_sha256': provenance['source_checkpoint_sha256'],
        'architecture': provenance['architecture'], 'standard_transform': repr(standard),
        'five_crop_normalization': repr(normalize),
    }


def load_external(root, exif_transpose=True):
    base = root / 'data/external_evaluation/tomato_field_eval_v1'
    manifest = pd.read_csv(base / 'manifest.csv')
    assert manifest.image_id.is_unique and len(manifest) == 62
    images, rows = [], []
    for row in manifest.to_dict('records'):
        path = base / row['image_path']
        assert sha256_file(path) == row['sha256']
        with Image.open(path) as im:
            orientation = int(im.getexif().get(274, 1))
            rgb = (ImageOps.exif_transpose(im) if exif_transpose else im).convert('RGB')
        images.append(rgb)
        rows.append(dict(image_id=row['image_id'], label_id=row['label_id'],
                         image_path=str(path.relative_to(root)), file_sha256=row['sha256'],
                         pixel_sha256=pixel_sha256(rgb), exif_orientation=orientation,
                         source_width=rgb.width, source_height=rgb.height))
    return images, pd.DataFrame(rows)


def load_plantdoc(root, split):
    """Read pinned cached Arrow bytes without mutating the HF cache or splits."""
    from datasets import Dataset, concatenate_datasets
    cache = (Path.home() / '.cache/huggingface/datasets/geraldmc___plantdoc-full/default/0.0.0/'
             '66df4b49a58d9c676aa32d9ab0c0d3fff56fb8ac')
    shards = sorted(cache.glob('*.arrow'))
    if not shards:
        raise FileNotFoundError('PlantDoc v0.1.0 cache unavailable; restore the recorded cache first.')
    source = concatenate_datasets([Dataset.from_file(str(path)) for path in shards])
    manifest = pd.read_csv(root / f'data/splits/{split}.csv')
    assert manifest.source.eq('plantdoc').all()
    assert not manifest.duplicated(['source', 'row_id']).any()
    images, rows = [], []
    for row in manifest.to_dict('records'):
        record = source[int(row['row_id'])]
        assert record['class_label'] == row['original_class']
        rgb = record['image'].convert('RGB')  # Match the established PlantDoc wrapper.
        images.append(rgb)
        rows.append(dict(image_id=f"plantdoc:{int(row['row_id'])}", label_id=row['label_id'],
                         row_id=int(row['row_id']), source='geraldmc/plantdoc-full@v0.1.0',
                         pixel_sha256=pixel_sha256(rgb), source_width=rgb.width, source_height=rgb.height))
    return images, pd.DataFrame(rows), {
        'repository': 'geraldmc/plantdoc-full', 'revision_requested_by_original_wrapper': 'v0.1.0',
        'cache_directory': str(cache), 'cache_arrow_sha256': {p.name: sha256_file(p) for p in shards},
        'cached_rows': len(source), 'fingerprint': source._fingerprint,
        'split_sha256': sha256_file(root / f'data/splits/{split}.csv'),
        'limitation': 'Offline cached Arrow data, validated by row/class and pixel hashes; no fresh Hub revision resolution.',
    }


def prepare_views(image, regime, standard, normalize):
    """Return tensors and boxes; coordinates are right/bottom exclusive."""
    width, height = image.size
    if regime == REGIMES[3]:
        canvas = image.resize((448, 448), Image.Resampling.BICUBIC)
        views = [normalize(canvas.crop(box)) for box in FIVE_BOXES]
        geometry = [{'coordinate_space': 'square_448', 'box': list(box),
                     'source_box_float': [box[0]*width/448, box[1]*height/448,
                                          box[2]*width/448, box[3]*height/448]} for box in FIVE_BOXES]
    else:
        fraction = {REGIMES[0]: 1., REGIMES[1]: .7, REGIMES[2]: .5}[regime]
        cw, ch = (width, height) if fraction == 1 else (max(16, int(round(width*fraction))), max(16, int(round(height*fraction))))
        left, top = (width-cw)//2, (height-ch)//2
        box = (left, top, left+cw, top+ch)
        views = [standard(image.crop(box))]
        geometry = [{'coordinate_space': 'source_rgb', 'box': list(box)}]
    assert all(tuple(view.shape) == (3, 224, 224) for view in views)
    return torch.stack(views), geometry


def predict(model, standard, normalize, images, metadata, key, partition, regime, protocol):
    predictions, views = [], []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            chunk = images[start:start+8]
            prepared = [prepare_views(im, regime, standard, normalize) for im in chunk]
            batch = torch.cat([item[0] for item in prepared])
            probabilities = model(batch).float().softmax(-1).cpu().numpy()
            offset = 0
            for i, (_, boxes) in enumerate(prepared):
                count = len(boxes)
                view_prob = probabilities[offset:offset+count]; offset += count
                probability = view_prob.mean(axis=0)
                record = metadata.iloc[start+i].to_dict()
                record.update(model_key=key, model=MODEL_NAMES[key], partition=partition,
                              regime=regime, protocol=protocol, predicted_label_id=int(probability.argmax()),
                              confidence=float(probability.max()), view_count=count)
                record['correct'] = record['predicted_label_id'] == record['label_id']
                record.update({f'probability_label_{j}': float(probability[j]) for j in range(5)})
                predictions.append(record)
                for j, box in enumerate(boxes):
                    views.append(dict(image_id=record['image_id'], model_key=key, partition=partition,
                                      protocol=protocol, regime=regime, view_index=j,
                                      geometry=json.dumps(box),
                                      **{f'probability_label_{c}': float(view_prob[j,c]) for c in range(5)}))
    return pd.DataFrame(predictions), pd.DataFrame(views)


def summarize(predictions):
    rows = []
    for group, frame in predictions.groupby(['protocol', 'partition', 'model_key', 'regime'], sort=False):
        bact = frame[frame.label_id.eq(3)]
        rows.append(dict(zip(['protocol', 'partition', 'model_key', 'regime'], group),
                         n=len(frame), correct=int(frame.correct.sum()),
                         accuracy=accuracy_score(frame.label_id, frame.predicted_label_id),
                         macro_f1=f1_score(frame.label_id, frame.predicted_label_id, labels=range(5), average='macro', zero_division=0),
                         bacterial_spot_n=len(bact), bacterial_spot_correct=int(bact.predicted_label_id.eq(3).sum()),
                         bacterial_spot_recall=float(bact.predicted_label_id.eq(3).mean()),
                         bacterial_as_septoria=int(bact.predicted_label_id.eq(4).sum()),
                         bacterial_mean_true_probability=float(bact.probability_label_3.mean()),
                         bacterial_mean_septoria_probability=float(bact.probability_label_4.mean())))
    return pd.DataFrame(rows)


def run_experiment(root, output_dir, reference_path):
    """Run both recorded loaders; refuse overwrites and never modify inputs."""
    torch.set_num_threads(4)
    torch.manual_seed(42)
    result = output_dir
    result.mkdir(parents=True, exist_ok=False)
    reference = json.loads(reference_path.read_text())
    external, external_meta = load_external(root, exif_transpose=True)
    raw, raw_meta = load_external(root, exif_transpose=False)
    dev, dev_meta, dev_provenance = load_plantdoc(root, 'field_dev')
    test, test_meta, test_provenance = load_plantdoc(root, 'field_test')
    assert len(dev) == 92 and len(test) == 47
    assert set(dev_meta.image_id).isdisjoint(test_meta.image_id)
    protocol = {
        'upstream_commit': '3c87a91', 'implementation_commit': 'caafa06',
        'reference_manifest_sha256': sha256_file(root / 'data/external_evaluation/tomato_field_eval_v1/manifest.csv'),
        'class_names': CLASS_NAMES, 'models': {},
        'field_dev': dev_provenance, 'field_test': test_provenance,
        'environment': {
            'python': platform.python_version(), 'torch': torch.__version__,
            'torchvision': __import__('torchvision').__version__, 'timm': timm.__version__,
            'Pillow': __import__('PIL').__version__, 'numpy': np.__version__,
            'device': 'cpu', 'threads': 4,
        },
        'regimes': list(REGIMES), 'five_crop_boxes_448': FIVE_BOXES,
        'aggregation': 'mean of five float32 softmax vectors, not mean logits',
        'model_input': [3, 224, 224],
        'protocols': {
            'upstream_raw': 'External PIL RGB, no EXIF transpose; PlantDoc established wrapper',
            'canonical_exif': 'External ImageOps.exif_transpose then RGB; Field Dev unchanged',
        },
        'exploratory': True, 'code_sha256': sha256_file(Path(__file__)),
        'reference_protocol_sha256': sha256_file(reference_path),
    }
    partitions = [
        (raw, raw_meta, 'external_field_v1', 'upstream_raw', REGIMES),
        (dev, dev_meta, 'verified_field_dev', 'upstream_raw', REGIMES),
        (external, external_meta, 'external_field_v1', 'canonical_exif', REGIMES),
        (test, test_meta, 'plantdoc_field_test', 'upstream_raw', REGIMES[:1]),
    ]
    for key in MODEL_NAMES:
        model, standard, normalize, provenance = load_model(root, key, reference)
        protocol['models'][key] = provenance
        frames, views = [], []
        for images, metadata, partition, orientation, regimes in partitions:
            for regime in regimes:
                started = time.time()
                predictions, view_predictions = predict(
                    model, standard, normalize, images, metadata,
                    key, partition, regime, orientation,
                )
                frames.append(predictions)
                views.append(view_predictions)
                print(key, partition, orientation, regime,
                      f'{predictions.correct.sum()}/{len(predictions)}',
                      f'{time.time()-started:.1f}s', flush=True)
        pd.concat(frames, ignore_index=True).to_csv(result / f'predictions_{key}.csv', index=False)
        pd.concat(views, ignore_index=True).to_csv(result / f'views_{key}.csv', index=False)
        del model
    predictions = pd.concat([
        pd.read_csv(result / f'predictions_{key}.csv') for key in MODEL_NAMES
    ], ignore_index=True)
    summarize(predictions).to_csv(result / 'summary.csv', index=False)
    protocol['output_sha256'] = {p.name: sha256_file(p) for p in result.glob('*.csv')}
    (result / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-protocol', type=Path, default=Path(
        'outputs/external_evaluation/tomato_field_eval_v1/20261125_163347_4ca8183f/evaluation_protocol.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    reference = args.reference_protocol if args.reference_protocol.is_absolute() else root / args.reference_protocol
    output = args.output if args.output.is_absolute() else root / args.output
    run_experiment(root, output, reference)


if __name__ == '__main__':
    main()
