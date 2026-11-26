"""Controlled Field-only versus Lab-rehearsal fine-tuning on Colab CUDA.

Training is opt-in and CUDA-only. Existing notebooks, splits and checkpoints
are read-only. Paired, atomic epoch checkpoints permit restart after disconnect.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import hashlib
import json
import math
import random
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
import timm

CLASSES = ['Healthy', 'Early Blight', 'Late Blight', 'Bacterial Spot', 'Septoria Leaf Spot']
CONDITIONS = ('field_only', 'mixed')
SPLITS = ('field_adapt_train', 'field_dev', 'field_test', 'lab_train')
EXPECTED_INITIAL = {
    'resnet18': '65d53766258f26cb505cce4c319c3f783cb0f3c6ef4608680906919731f2782d',
    'vit_small': '42c26a8aa70311d9504d55dda422f20e2ea5f0a5878f5922b9f855c052a18c2e',
}
CHECKPOINT_NAMES = {'resnet18': 'resnet18_lab_best.pt', 'vit_small': 'vit_small_lab_best.pt'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def derived_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    torch.save(value, temporary)
    os.replace(temporary, path)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required. In Colab select Runtime → Change runtime type → GPU. No local or CPU training is allowed.')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda')


def environment():
    import importlib.metadata
    return {'python': platform.python_version(),
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA',
            'cuda': torch.version.cuda,
            'versions': {name: importlib.metadata.version(name) for name in
                         ('torch', 'torchvision', 'timm', 'datasets', 'numpy', 'pandas', 'scikit-learn', 'Pillow')}}


def default_config(architecture, seeds=(42, 43, 44)):
    if architecture not in CHECKPOINT_NAMES:
        raise ValueError('Architecture must be resnet18 or vit_small.')
    if list(seeds) not in ([42], [42, 43, 44]):
        raise ValueError('Predeclare seed 42 alone or all seeds 42,43,44. No best-seed selection.')
    return {
        'schema_version': 1, 'architecture': architecture, 'seeds': list(seeds),
        'field_split': 'field_adapt_train', 'field_rows': 370, 'lab_rows_per_epoch': 370,
        'field_batch_size': 16 if architecture == 'resnet18' else 8,
        'lambda_lab': 1.0, 'head_epochs': 2, 'last_block_epochs': 6,
        'head_lr': 1e-4, 'last_block_lr': 1e-5, 'weight_decay': 1e-4,
        'optimizer': 'AdamW; reset at head-to-last-block transition',
        'schedule': 'constant learning rates within each of the two fixed phases',
        'precision': 'float32; no autocast or gradient scaling',
        'batchnorm': 'frozen running statistics in both conditions',
        'field_policy': 'Each row once per epoch; identical order and augmented tensors for both conditions',
        'lab_policy': '370 distinct Lab Train rows per epoch; reuse across epochs allowed',
        'selection': 'Common epoch maximizing mean five-class Field Dev macro-F1 of the pair; earliest tie; epoch 0 eligible',
        'early_stopping': False, 'phase_rewind': False,
        'checkpoint_frequency': 'atomic matched-pair checkpoint after every complete epoch; uncommitted epoch replayed on resume',
        'evaluation': 'Only after all declared seeds finish and checkpoint hashes are sealed',
        'split_authorization': 'Verified 370-image adaptation subset; original 476 includes Dev/quarantine',
    }


class Assets:
    """Compact offline HF containers, mapped by preserved original source row ID."""
    def __init__(self, repo, directory):
        self.repo, self.directory = Path(repo), Path(directory)
        manifest = self.directory / 'asset_manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'{manifest}: run scripts/prepare_controlled_colab_assets.py and upload its output.')
        self.manifest = json.loads(manifest.read_text())
        self.manifest_hash = sha256(manifest)
        label_map = json.loads((self.repo / 'data/splits/label_map.json').read_text())
        assert label_map == dict(zip(CLASSES, range(5))), 'Canonical class mapping changed'
        self.frames = {}
        for name in SPLITS:
            path = self.repo / f'data/splits/{name}.csv'
            if not path.is_file() or sha256(path) != self.manifest['split_sha256'][name]:
                raise ValueError(f'Missing or altered canonical split: {path}')
            frame = pd.read_csv(path)
            assert not frame.duplicated(['source', 'row_id']).any()
            assert frame.label_id.between(0, 4).all()
            assert all(CLASSES[int(r.label_id)] == r.class_name for r in frame.itertuples())
            self.frames[name] = frame
        for name, expected in self.manifest['file_sha256'].items():
            path = self.directory / name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f'Asset missing or corrupt: {path}')
        self.sources = {source: load_from_disk(str(self.directory / 'datasets' / source))
                        for source in ('plantdoc', 'plantvillage')}
        self.indices = {}
        for source, dataset in self.sources.items():
            ids = list(map(int, self.manifest['sources'][source]['source_row_ids']))
            assert len(ids) == len(dataset)
            assert len(set(ids)) == len(ids)
            self.indices[source] = dict(zip(ids, range(len(ids))))
        for frame in self.frames.values():
            for row in frame.itertuples():
                index = self.indices[row.source].get(int(row.row_id))
                assert index is not None, (row.source, row.row_id)
                assert self.sources[row.source][index]['class_label'] == row.original_class
        for a, b in [('field_adapt_train', 'field_dev'), ('field_adapt_train', 'field_test'), ('field_dev', 'field_test')]:
            assert set(self.frames[a].row_id).isdisjoint(self.frames[b].row_id)
        train, dev = self.frames['field_adapt_train'], self.frames['field_dev']
        assert len(train) == 370 and len(dev) == 92 and len(self.frames['field_test']) == 47
        assert len(self.frames['lab_train']) == 5365
        assert self.frames['field_test'].original_split.eq('test').all()
        assert train.original_split.eq('train').all() and dev.original_split.eq('train').all()
        assert set(train.group_id).isdisjoint(dev.group_id)
        leakage = self.manifest['leakage_screen']
        if leakage['heldout_exact_or_phash_matches']:
            raise ValueError('Unresolved training/held-out content matches in asset manifest. Review before training.')

    def image(self, row):
        record = self.sources[row['source']][self.indices[row['source']][int(row['row_id'])]]
        assert record['class_label'] == row['original_class']
        return record['image'].convert('RGB')

    def checkpoint(self, architecture):
        path = self.directory / 'checkpoints' / CHECKPOINT_NAMES[architecture]
        if not path.is_file() or sha256(path) != EXPECTED_INITIAL[architecture]:
            raise ValueError(f'Expected the recorded Lab baseline checkpoint, without substitution: {path}')
        return path


def load_model(assets, architecture, device):
    path = assets.checkpoint(architecture)
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert list(state['class_names']) == CLASSES
    if architecture == 'resnet18':
        assert state['architecture'] == 'resnet18' and state['pretrained_weights'] == 'IMAGENET1K_V1'
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
        evaluation = ResNet18_Weights.IMAGENET1K_V1.transforms()
        mean, std, interpolation = evaluation.mean, evaluation.std, transforms.InterpolationMode.BILINEAR
    else:
        assert state['architecture'] == 'vit_small_patch16_224.augreg_in1k'
        assert state['pretrained_weights'] == 'augreg_in1k'
        model = timm.create_model(state['architecture'], pretrained=False, num_classes=5)
        config = state['preprocessing_config']
        assert tuple(config['input_size']) == (3, 224, 224)
        evaluation = timm.data.create_transform(**config, is_training=False)
        mean, std = config['mean'], config['std']
        interpolation = transforms.InterpolationMode.BICUBIC
    model.load_state_dict(state['model_state_dict'], strict=True)
    # Historical mild crop/flip/jitter policy, shared exactly across conditions.
    training = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.RandomResizedCrop(224, scale=(.85, 1.), ratio=(.90, 1.10), interpolation=interpolation),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=.10, contrast=.10, saturation=.10)], p=.35),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    return model.to(device), training, evaluation


def set_phase(model, architecture, phase):
    for name, parameter in model.named_parameters():
        if architecture == 'resnet18':
            parameter.requires_grad_(name.startswith('fc.') or (phase == 'last_block' and name.startswith('layer4.')))
        else:
            prefixes = ('head.',) if phase == 'head' else (f'blocks.{len(model.blocks)-1}.', 'norm.', 'fc_norm.', 'head.')
            parameter.requires_grad_(name.startswith(prefixes))
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def make_optimizer(model, architecture, phase, config):
    head = model.fc if architecture == 'resnet18' else model.get_classifier()
    groups = [{'params': list(head.parameters()), 'lr': config['head_lr']}]
    if phase == 'last_block':
        head_ids = {id(p) for p in head.parameters()}
        groups.append({'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids],
                       'lr': config['last_block_lr']})
    return torch.optim.AdamW(groups, weight_decay=config['weight_decay'])


def augmented_batch(assets, frame, indices, transform, seed, epoch, domain):
    tensors, labels, ids = [], [], []
    for index in indices:
        row = frame.iloc[int(index)].to_dict()
        generator = torch.Generator().manual_seed(derived_seed(seed, epoch, domain, row['row_id']))
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(generator.get_state())
            tensors.append(transform(assets.image(row)))
        labels.append(int(row['label_id']))
        ids.append(int(row['row_id']))
    return torch.stack(tensors), torch.tensor(labels), ids


def epoch_plan(config, seed, epoch, lab_count):
    field = np.random.default_rng(derived_seed(seed, epoch, 'field_order')).permutation(config['field_rows'])
    lab = np.random.default_rng(derived_seed(seed, epoch, 'lab_order')).choice(lab_count, config['lab_rows_per_epoch'], replace=False)
    assert len(field) == len(lab) and len(set(field)) == config['field_rows']
    return [(field[i:i+config['field_batch_size']], lab[i:i+config['field_batch_size']])
            for i in range(0, len(field), config['field_batch_size'])]


def prediction_metrics(frame):
    precision, recall, f1, support = precision_recall_fscore_support(
        frame.label_id, frame.predicted_label_id, labels=range(5), zero_division=0)
    summary = {'n': len(frame), 'correct': int(frame.correct.sum()),
               'accuracy': accuracy_score(frame.label_id, frame.predicted_label_id),
               'macro_f1': f1_score(frame.label_id, frame.predicted_label_id, labels=range(5), average='macro', zero_division=0)}
    per_class = pd.DataFrame({'class_name': CLASSES, 'label_id': range(5), 'precision': precision,
                              'recall': recall, 'f1': f1, 'support': support})
    return summary, per_class, confusion_matrix(frame.label_id, frame.predicted_label_id, labels=range(5))


def predict(model, assets, frame, transform, batch_size=16):
    model.eval()
    rows = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            chunk = frame.iloc[start:start+batch_size]
            batch = torch.stack([transform(assets.image(row)) for row in chunk.to_dict('records')]).to(device)
            probabilities = model(batch).float().softmax(-1).cpu().numpy()
            for row, probability in zip(chunk.to_dict('records'), probabilities):
                row.update(image_id=f"{row['source']}:{int(row['row_id'])}",
                           predicted_label_id=int(probability.argmax()), confidence=float(probability.max()))
                row['correct'] = int(row['label_id']) == row['predicted_label_id']
                row.update({f'probability_label_{i}': float(p) for i, p in enumerate(probability)})
                rows.append(row)
    return pd.DataFrame(rows)


def open_experiment(repo, assets, output_root, experiment_id, architecture, seeds=(42, 43, 44), resnet_run=None):
    require_cuda()
    if not experiment_id or experiment_id in ('.', '..') or Path(experiment_id).name != experiment_id:
        raise ValueError('Use a single directory name as the unique experiment ID.')
    if architecture == 'vit_small':
        if resnet_run is None:
            raise ValueError('Complete and validate ResNet18 before enabling ViT-Small.')
        proof = json.loads((Path(resnet_run) / 'training_validation.json').read_text())
        assert json.loads((Path(resnet_run) / 'config.json').read_text())['architecture'] == 'resnet18'
        assert proof['all_declared_seed_pairs_complete'] and proof['matched_exposure_and_updates']
    config = default_config(architecture, seeds)
    config.update(initial_sha256=sha256(assets.checkpoint(architecture)), asset_manifest_sha256=assets.manifest_hash,
                  split_sha256=assets.manifest['split_sha256'], code_sha256=sha256(Path(__file__)))
    path = Path(output_root) / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    if (path / 'config.json').exists():
        if json.loads((path / 'config.json').read_text()) != config:
            raise ValueError('Resume configuration/assets/code differ. Restore the originals or use a new experiment ID.')
        previous = json.loads((path / 'environment.json').read_text())
        if previous['versions'] != environment()['versions']:
            raise ValueError('Library versions changed since this run started. Restore the recorded versions before resuming.')
    else:
        if any(path.iterdir()):
            raise ValueError('Experiment directory is not empty and has no configuration. Use a new ID.')
        atomic_json(path / 'config.json', config)
        atomic_json(path / 'environment.json', environment())
    return path


def smoke_test(run, assets):
    device = require_cuda()
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    if (run / 'smoke_test.json').exists():
        record = json.loads((run / 'smoke_test.json').read_text())
        if (record['environment'] == environment() and record.get('config_hash') == digest_json(config)
                and record['checkpoint_sha256'] == sha256(run / 'smoke_checkpoint.pt')):
            return record
    model, training, _ = load_model(assets, config['architecture'], device)
    initial = state_digest(model.state_dict())
    set_phase(model, config['architecture'], 'last_block')
    optimizer = make_optimizer(model, config['architecture'], 'last_block', config)
    x, y, _ = augmented_batch(assets, assets.frames['field_adapt_train'], [0, 1], training, 123456, 0, 'smoke')
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
    loss.backward()
    lab_x, lab_y, _ = augmented_batch(assets, assets.frames['lab_train'], [0, 1], training, 123456, 0, 'smoke_lab')
    lab_loss = torch.nn.functional.cross_entropy(model(lab_x.to(device)), lab_y.to(device))
    (config['lambda_lab']*lab_loss).backward()
    assert torch.isfinite(loss) and torch.isfinite(lab_loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    updated = cpu_state(model)
    assert state_digest(updated) != initial
    target = run / 'smoke_checkpoint.pt'
    atomic_torch(target, {'model_state_dict': updated})
    model.load_state_dict(torch.load(target, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
    assert state_digest(model.state_dict()) == state_digest(updated)
    record = {'passed': True, 'finite_loss': float(loss.detach().cpu()),
              'finite_lab_loss': float(lab_loss.detach().cpu()), 'config_hash': digest_json(config),
              'environment': environment(),
              'checkpoint_sha256': sha256(target), 'training_checkpoint_modified': False,
              'operations': ['Field and Lab data loading', 'baseline loading', 'CUDA forward',
                             'Field plus weighted Lab backward', 'AdamW step', 'persistent checkpoint write/read']}
    atomic_json(run / 'smoke_test.json', record)
    del model, optimizer
    torch.cuda.empty_cache()
    return record


def _optimizer_cpu(optimizer):
    def clone(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        return value
    return clone(optimizer.state_dict())


def _write_history(seed_dir, history, audit):
    pd.DataFrame(history).to_csv(seed_dir / 'history.csv', index=False)
    pd.DataFrame(audit).to_csv(seed_dir / 'batch_audit.csv', index=False)


def train_pair(run, assets, seed):
    """Run/resume one seed, never touching Field Test predictions during training."""
    device = require_cuda()
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    if seed not in config['seeds']:
        raise ValueError('Seed was not predeclared.')
    if not (run / 'smoke_test.json').exists():
        raise RuntimeError('Complete the CUDA smoke-test cell before training.')
    smoke = json.loads((run / 'smoke_test.json').read_text())
    if not (smoke['passed'] and smoke['config_hash'] == digest_json(config) and smoke['environment'] == environment()):
        raise RuntimeError('Run the CUDA smoke-test cell again for this configuration and runtime.')
    seed_dir = run / f'seed_{seed}'
    seed_dir.mkdir(exist_ok=True)
    if (seed_dir / 'complete.json').exists():
        complete = json.loads((seed_dir / 'complete.json').read_text())
        for name, expected in complete['checkpoint_sha256'].items():
            assert sha256(seed_dir / name) == expected
        return complete
    if (run / 'evaluation_seal.json').exists():
        raise RuntimeError('Evaluation already sealed this experiment. Do not train or retune after test access.')
    models, optimizers = {}, {}
    for condition in CONDITIONS:
        models[condition], training, evaluation = load_model(assets, config['architecture'], device)
    initial = {key: state_digest(model.state_dict()) for key, model in models.items()}
    assert len(set(initial.values())) == 1
    history, audit, counts = [], [], {key: [0]*config['field_rows'] for key in CONDITIONS}
    best, best_score, best_epoch, start_epoch = {}, -1., 0, 1
    checkpoint = seed_dir / 'resume.pt'
    elapsed, phase = 0., None
    condition_seconds = {key: 0. for key in CONDITIONS}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['config_hash'] == digest_json(config) and saved['seed'] == seed
        assert saved['initial_state_hash'] == initial['field_only']
        start_epoch, phase = saved['completed_epoch']+1, saved['phase']
        history, audit, counts = saved['history'], saved['audit'], saved['counts']
        best, best_score, best_epoch = saved['best_states'], saved['best_score'], saved['best_epoch']
        elapsed = saved['elapsed_seconds']
        condition_seconds = saved['condition_training_seconds']
        for key in CONDITIONS:
            models[key].load_state_dict(saved['models'][key], strict=True)
            set_phase(models[key], config['architecture'], phase)
            optimizers[key] = make_optimizer(models[key], config['architecture'], phase, config)
            optimizers[key].load_state_dict(saved['optimizers'][key])
            for entry in optimizers[key].state.values():
                for name, value in entry.items():
                    # AdamW's scalar step remains on CPU when capturable=False.
                    if torch.is_tensor(value) and name != 'step':
                        entry[name] = value.to(device)
        del saved
        _write_history(seed_dir, history, audit)
    else:
        scores = []
        for key in CONDITIONS:
            frame = predict(models[key], assets, assets.frames['field_dev'], evaluation)
            metrics, _, _ = prediction_metrics(frame)
            scores.append(metrics['macro_f1'])
            history.append(dict(seed=seed, condition=key, epoch=0, phase='initial', **metrics))
            best[key] = cpu_state(models[key])
        assert abs(scores[0]-scores[1]) < 1e-12
        best_score = float(np.mean(scores))
    total_epochs = config['head_epochs']+config['last_block_epochs']
    for epoch in range(start_epoch, total_epochs+1):
        started = time.perf_counter()
        current_phase = 'head' if epoch <= config['head_epochs'] else 'last_block'
        if current_phase != phase:
            for key in CONDITIONS:
                set_phase(models[key], config['architecture'], current_phase)
                optimizers[key] = make_optimizer(models[key], config['architecture'], current_phase, config)
            phase = current_phase
        for model in models.values():
            set_phase(model, config['architecture'], phase)
        losses = {key: [0., 0.] for key in CONDITIONS}
        plan = epoch_plan(config, seed, epoch, len(assets.frames['lab_train']))
        for step, (field_indices, lab_indices) in enumerate(plan):
            field_x, field_y, field_ids = augmented_batch(assets, assets.frames['field_adapt_train'], field_indices, training, seed, epoch, 'field')
            lab_x, lab_y, lab_ids = augmented_batch(assets, assets.frames['lab_train'], lab_indices, training, seed, epoch, 'lab')
            tensor_hash = hashlib.sha256(field_x.numpy().tobytes()).hexdigest()
            field_x, field_y = field_x.to(device), field_y.to(device)
            for key in CONDITIONS:
                torch.cuda.synchronize()
                condition_started = time.perf_counter()
                # Same stochastic network sequence for each condition's Field pass.
                torch.manual_seed(derived_seed(seed, epoch, step, 'network'))
                torch.cuda.manual_seed_all(derived_seed(seed, epoch, step, 'network'))
                optimizer, model = optimizers[key], models[key]
                optimizer.zero_grad(set_to_none=True)
                field_loss = torch.nn.functional.cross_entropy(model(field_x), field_y)
                if not torch.isfinite(field_loss):
                    raise FloatingPointError(f'{key}: non-finite Field loss')
                field_loss.backward()
                losses[key][0] += float(field_loss.detach().cpu())*len(field_indices)
                if key == 'mixed':
                    lab_loss = torch.nn.functional.cross_entropy(model(lab_x.to(device)), lab_y.to(device))
                    if not torch.isfinite(lab_loss):
                        raise FloatingPointError('Non-finite Lab loss')
                    (config['lambda_lab']*lab_loss).backward()
                    losses[key][1] += float(lab_loss.detach().cpu())*len(lab_indices)
                optimizer.step()
                torch.cuda.synchronize()
                condition_seconds[key] += time.perf_counter()-condition_started
                for index in field_indices:
                    counts[key][int(index)] += 1
                audit.append(dict(seed=seed, epoch=epoch, batch=step, condition=key,
                                  field_row_ids=json.dumps(field_ids), field_tensor_sha256=tensor_hash,
                                  field_presentations=len(field_ids), optimizer_updates=1,
                                  lab_presentations=len(lab_ids) if key == 'mixed' else 0,
                                  lab_row_ids=json.dumps(lab_ids) if key == 'mixed' else '[]'))
        assert counts['field_only'] == counts['mixed'] == [epoch]*config['field_rows']
        validation = {}
        for key in CONDITIONS:
            predictions = predict(models[key], assets, assets.frames['field_dev'], evaluation)
            metrics, _, _ = prediction_metrics(predictions)
            validation[key] = metrics['macro_f1']
            history.append(dict(seed=seed, condition=key, epoch=epoch, phase=phase,
                                field_loss=losses[key][0]/config['field_rows'],
                                lab_loss=losses[key][1]/config['lab_rows_per_epoch'] if key == 'mixed' else None,
                                **metrics))
        score = float(np.mean(list(validation.values())))
        if score > best_score+1e-12:
            best_score, best_epoch = score, epoch
            best = {key: cpu_state(model) for key, model in models.items()}
        torch.cuda.synchronize()
        elapsed += time.perf_counter()-started
        payload = {
            'config_hash': digest_json(config), 'seed': seed, 'initial_state_hash': initial['field_only'],
            'completed_epoch': epoch, 'phase': phase,
            'models': {key: cpu_state(model) for key, model in models.items()},
            'optimizers': {key: _optimizer_cpu(value) for key, value in optimizers.items()},
            'best_states': best, 'best_score': best_score, 'best_epoch': best_epoch,
            'history': history, 'audit': audit, 'counts': counts, 'elapsed_seconds': elapsed,
            'condition_training_seconds': condition_seconds,
        }
        # One atomic file commits both conditions and the shared selection state.
        atomic_torch(checkpoint, payload)
        del payload
        _write_history(seed_dir, history, audit)
        print(f'Seed {seed} · epoch {epoch}/{total_epochs} · Field Dev F1: '
              f'Field-only {validation["field_only"]:.4f}, Mixed {validation["mixed"]:.4f} · '
              f'paired epoch checkpoint saved', flush=True)
    updates_per_epoch = math.ceil(config['field_rows']/config['field_batch_size'])
    checkpoint_hashes = {}
    for key in CONDITIONS:
        path = seed_dir / f'{key}_selected.pt'
        atomic_torch(path, {
            'model_state_dict': best[key], 'class_names': CLASSES, 'architecture': config['architecture'],
            'seed': seed, 'condition': key, 'selected_epoch': best_epoch,
            'initial_checkpoint_sha256': config['initial_sha256'], 'config_hash': digest_json(config),
            'selection_split': 'field_dev', 'common_selection_score': best_score,
            'selected_field_presentations': best_epoch*config['field_rows'],
            'selected_lab_presentations': best_epoch*config['lab_rows_per_epoch'] if key == 'mixed' else 0,
            'selected_optimizer_updates': best_epoch*updates_per_epoch,
        })
        checkpoint_hashes[path.name] = sha256(path)
    exposure = pd.DataFrame({'source_row_id': assets.frames['field_adapt_train'].row_id,
                             'field_only': counts['field_only'], 'mixed': counts['mixed']})
    exposure.to_csv(seed_dir / 'field_exposure.csv', index=False)
    result = {'seed': seed, 'complete': True, 'epochs': total_epochs, 'initial_state_hash': initial['field_only'],
              'initialization_identical': True, 'field_presentations_per_condition': total_epochs*config['field_rows'],
              'optimizer_updates_per_condition': total_epochs*updates_per_epoch,
              'additional_lab_presentations_mixed': total_epochs*config['lab_rows_per_epoch'],
              'selected_common_epoch': best_epoch, 'selected_field_presentations_per_condition': best_epoch*config['field_rows'],
              'selected_updates_per_condition': best_epoch*updates_per_epoch,
              'joint_field_dev_macro_f1': best_score, 'elapsed_pair_seconds': elapsed,
              'condition_training_seconds_excluding_data_and_validation': condition_seconds,
              'training_image_forwards': {'field_only': total_epochs*config['field_rows'],
                                           'mixed': total_epochs*(config['field_rows']+config['lab_rows_per_epoch'])},
              'checkpoint_sha256': checkpoint_hashes}
    atomic_json(seed_dir / 'complete.json', result)
    del models, optimizers, best
    torch.cuda.empty_cache()
    return result


def validate_training(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    records = []
    for seed in config['seeds']:
        seed_dir = run / f'seed_{seed}'
        complete = json.loads((seed_dir / 'complete.json').read_text())
        assert complete['complete'] and complete['initialization_identical']
        epochs = config['head_epochs']+config['last_block_epochs']
        steps = math.ceil(config['field_rows']/config['field_batch_size'])
        assert complete['seed'] == seed and complete['epochs'] == epochs
        assert complete['field_presentations_per_condition'] == epochs*config['field_rows']
        assert complete['optimizer_updates_per_condition'] == epochs*steps
        exposure = pd.read_csv(seed_dir / 'field_exposure.csv')
        assert len(exposure) == config['field_rows'] and exposure.source_row_id.is_unique
        assert exposure.field_only.eq(exposure.mixed).all()
        assert exposure.field_only.eq(complete['epochs']).all()
        audit = pd.read_csv(seed_dir / 'batch_audit.csv')
        a = audit[audit.condition.eq('field_only')].set_index(['epoch', 'batch']).sort_index()
        b = audit[audit.condition.eq('mixed')].set_index(['epoch', 'batch']).sort_index()
        assert a.index.equals(b.index)
        assert not a.index.has_duplicates and len(a) == epochs*steps
        for field in ('field_row_ids', 'field_tensor_sha256', 'field_presentations', 'optimizer_updates'):
            assert a[field].equals(b[field])
        assert a.optimizer_updates.sum() == b.optimizer_updates.sum() == complete['optimizer_updates_per_condition']
        assert a.optimizer_updates.eq(1).all() and b.optimizer_updates.eq(1).all()
        assert a.field_presentations.sum() == complete['field_presentations_per_condition']
        for epoch in range(1, epochs+1):
            rows = [row for batch in a.loc[epoch].field_row_ids for row in json.loads(batch)]
            assert sorted(rows) == sorted(exposure.source_row_id.tolist())
        assert a.lab_presentations.sum() == 0
        assert b.lab_presentations.sum() == complete['additional_lab_presentations_mixed']
        history = pd.read_csv(seed_dir / 'history.csv')
        assert not history.duplicated(['epoch', 'condition']).any()
        assert len(history) == 2*(epochs+1)
        assert set(history.condition) == set(CONDITIONS)
        assert set(history.epoch) == set(range(epochs+1))
        scores = history.groupby('epoch').macro_f1.mean().sort_index()
        expected = next(int(epoch) for epoch, score in scores.items() if score >= scores.max()-1e-12)
        assert complete['selected_common_epoch'] == expected
        for key in CONDITIONS:
            path = seed_dir / f'{key}_selected.pt'
            assert sha256(path) == complete['checkpoint_sha256'][path.name]
            selected = torch.load(path, map_location='cpu', weights_only=True)
            assert selected['selected_epoch'] == expected and selected['selection_split'] == 'field_dev'
            assert selected['config_hash'] == digest_json(config)
            assert selected['initial_checkpoint_sha256'] == config['initial_sha256']
            assert selected['selected_field_presentations'] == expected*config['field_rows']
            assert selected['selected_optimizer_updates'] == expected*steps
            assert selected['selected_lab_presentations'] == (expected*config['lab_rows_per_epoch'] if key == 'mixed' else 0)
        records.append(complete)
    result = {'all_declared_seed_pairs_complete': True, 'matched_exposure_and_updates': True,
              'selection_used_field_dev_only': True, 'seeds': config['seeds'], 'pairs': records}
    atomic_json(run / 'training_validation.json', result)
    return result


def evaluate_heldout(run, assets):
    """Seal all planned checkpoints before any held-out model inference."""
    device = require_cuda()
    run = Path(run)
    validation = validate_training(run)
    config = json.loads((run / 'config.json').read_text())
    seal = {'config_hash': digest_json(config), 'seed_checkpoints': {
        str(row['seed']): row['checkpoint_sha256'] for row in validation['pairs']}}
    if (run / 'evaluation_seal.json').exists():
        assert json.loads((run / 'evaluation_seal.json').read_text()) == seal
    else:
        atomic_json(run / 'evaluation_seal.json', seal)
    results, classes = [], []
    for seed in config['seeds']:
        directory = run / f'seed_{seed}'
        for key in CONDITIONS:
            model, _, transform = load_model(assets, config['architecture'], device)
            checkpoint = torch.load(directory / f'{key}_selected.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            dev = predict(model, assets, assets.frames['field_dev'], transform)
            dev.to_csv(directory / f'predictions_{key}_field_dev.csv', index=False)
            dev_metrics, _, _ = prediction_metrics(dev)
            history = pd.read_csv(directory / 'history.csv')
            expected_dev = history[history.condition.eq(key) & history.epoch.eq(checkpoint['selected_epoch'])].iloc[0]
            assert abs(dev_metrics['macro_f1']-expected_dev.macro_f1) < 1e-10
            frame = predict(model, assets, assets.frames['field_test'], transform)
            frame['seed'], frame['condition'], frame['architecture'] = seed, key, config['architecture']
            frame['checkpoint_sha256'] = sha256(directory / f'{key}_selected.pt')
            frame['selected_epoch'] = checkpoint['selected_epoch']
            frame.to_csv(directory / f'predictions_{key}_field_test.csv', index=False)
            summary, per_class, matrix = prediction_metrics(frame)
            results.append(dict(seed=seed, condition=key, architecture=config['architecture'], **summary))
            per_class['seed'], per_class['condition'] = seed, key
            classes.append(per_class)
            pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(directory / f'confusion_{key}_field_test.csv')
            del model
    summary = pd.DataFrame(results)
    per_class = pd.concat(classes, ignore_index=True)
    summary.to_csv(run / 'field_test_summary.csv', index=False)
    per_class.to_csv(run / 'field_test_per_class.csv', index=False)
    for row in summary.itertuples():
        frame = pd.read_csv(run / f'seed_{row.seed}/predictions_{row.condition}_field_test.csv')
        recomputed, _, _ = prediction_metrics(frame)
        assert abs(recomputed['accuracy']-row.accuracy) < 1e-12 and abs(recomputed['macro_f1']-row.macro_f1) < 1e-12
    atomic_json(run / 'evaluation_validation.json', {'metrics_recomputed_from_saved_predictions': True,
                'field_test_n': 47, 'seed_count': len(config['seeds']),
                'no_test_based_checkpoint_selection': True,
                'independence_scope': 'Verified source rows, existing groups and asset exact/pHash screen; historical test already inspected, not a fresh prospective test.'})
    torch.cuda.empty_cache()
    return summary, per_class
