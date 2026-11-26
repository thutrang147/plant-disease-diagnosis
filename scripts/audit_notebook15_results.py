"""Audit imported Notebook 15 artifacts without training, inference or run writes.

Run from the repository root: .venv/bin/python scripts/audit_notebook15_results.py
Writes only to outputs/controlled_field_mixed/notebook15_finalization/.
"""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.controlled_field_report import load_evidence
from src.controlled_field_mixed import state_digest

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()

def main():
    root = PROJECT_ROOT
    run = root / 'outputs/controlled_field_mixed/controlled_resnet18_s42_43_44_v1'
    out = root / 'outputs/controlled_field_mixed/notebook15_finalization'
    out.mkdir(parents=True, exist_ok=True)
    files = {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    (out / 'imported_artifacts_before.json').write_text(json.dumps(files, indent=2) + '\n')
    config = json.loads((run / 'config.json').read_text())
    env = json.loads((run / 'environment.json').read_text())
    seal = json.loads((run / 'evaluation_seal.json').read_text())
    tv = json.loads((run / 'training_validation.json').read_text())
    ev = json.loads((run / 'evaluation_validation.json').read_text())
    smoke = json.loads((run / 'smoke_test.json').read_text())
    assert config['architecture'] == 'resnet18' and config['seeds'] == [42, 43, 44]
    assert (config['field_rows'], config['lab_rows_per_epoch'], config['field_batch_size'], config['head_epochs'], config['last_block_epochs'], config['lambda_lab']) == (370, 370, 16, 2, 6, 1.0)
    assert config['code_sha256'] == sha(root / 'src/controlled_field_mixed.py')
    for name, h in config['split_sha256'].items():
        assert sha(root / f'data/splits/{name}.csv') == h
    assets = root / 'outputs/controlled_field_mixed/colab_assets_v1'
    assert sha(assets / 'asset_manifest.json') == config['asset_manifest_sha256']
    assert sha(assets / 'checkpoints/resnet18_lab_best.pt') == config['initial_sha256']
    assert seal['config_hash'] == digest(config)
    assert tv['all_declared_seed_pairs_complete'] and tv['matched_exposure_and_updates'] and tv['selection_used_field_dev_only']
    assert ev['metrics_recomputed_from_saved_predictions'] and ev['field_test_n'] == 47 and (ev['seed_count'] == 3)
    assert smoke['passed'] and smoke['config_hash'] == digest(config) and (smoke['environment'] == env)
    baseline = torch.load(assets / 'checkpoints/resnet18_lab_best.pt', map_location='cpu', weights_only=True)
    initial_digest = state_digest(baseline['model_state_dict'])
    del baseline
    reference = load_evidence(root)
    canonical = {name: pd.read_csv(root / f'data/splits/{name}.csv') for name in ['field_adapt_train', 'lab_train', 'field_dev', 'field_test']}
    assert not set(canonical['field_adapt_train'].row_id) & set(canonical['field_dev'].row_id)
    assert not set(canonical['field_adapt_train'].row_id) & set(canonical['field_test'].row_id)
    assert not set(canonical['field_dev'].row_id) & set(canonical['field_test'].row_id)
    summary = []
    class_rows = []
    selection = []
    for seed in config['seeds']:
        sd = run / f'seed_{seed}'
        complete = json.loads((sd / 'complete.json').read_text())
        history = pd.read_csv(sd / 'history.csv')
        audit = pd.read_csv(sd / 'batch_audit.csv')
        exposure = pd.read_csv(sd / 'field_exposure.csv')
        assert complete == next((p for p in tv['pairs'] if p['seed'] == seed))
        assert complete['complete'] and complete['initialization_identical'] and (complete['initial_state_hash'] == initial_digest)
        assert complete['epochs'] == 8 and complete['field_presentations_per_condition'] == 2960 and (complete['optimizer_updates_per_condition'] == 192) and (complete['additional_lab_presentations_mixed'] == 2960)
        assert len(exposure) == 370 and set(exposure.source_row_id) == set(canonical['field_adapt_train'].row_id)
        assert exposure.field_only.eq(8).all() and exposure.mixed.eq(8).all()
        assert not audit.duplicated(['epoch', 'batch', 'condition']).any() and len(audit) == 384
        a = audit[audit.condition.eq('field_only')].set_index(['epoch', 'batch']).sort_index()
        b = audit[audit.condition.eq('mixed')].set_index(['epoch', 'batch']).sort_index()
        assert a.index.equals(b.index) and len(a) == 192
        for col in ['field_row_ids', 'field_tensor_sha256', 'field_presentations', 'optimizer_updates']:
            assert a[col].equals(b[col])
        assert a.field_presentations.sum() == b.field_presentations.sum() == 2960
        assert a.optimizer_updates.eq(1).all() and b.optimizer_updates.eq(1).all()
        assert a.lab_presentations.eq(0).all() and b.lab_presentations.sum() == 2960
        for epoch in range(1, 9):
            aa = a.loc[epoch]
            bb = b.loc[epoch]
            field = [x for s in aa.field_row_ids for x in json.loads(s)]
            lab = [x for s in bb.lab_row_ids for x in json.loads(s)]
            assert sorted(field) == sorted(canonical['field_adapt_train'].row_id.tolist())
            assert len(lab) == len(set(lab)) == 370 and set(lab) <= set(canonical['lab_train'].row_id)
            assert aa.field_presentations.tolist() == [16] * 23 + [2]
            assert all((len(json.loads(s)) == n for s, n in zip(aa.field_row_ids, aa.field_presentations)))
            assert all((len(json.loads(s)) == n for s, n in zip(bb.lab_row_ids, bb.lab_presentations)))
        assert len(history) == 18 and (not history.duplicated(['epoch', 'condition']).any())
        assert set(history.epoch) == set(range(9)) and set(history.condition) == {'field_only', 'mixed'}
        scores = history.groupby('epoch').macro_f1.mean().sort_index()
        selected = next((int(e) for e, s in scores.items() if s >= scores.max() - 1e-12))
        assert complete['selected_common_epoch'] == selected
        assert abs(complete['joint_field_dev_macro_f1'] - scores.loc[selected]) < 1e-12
        assert complete['selected_field_presentations_per_condition'] == selected * 370 and complete['selected_updates_per_condition'] == selected * 24
        selection.append({'seed': seed, 'selected_epoch': selected, 'joint_dev_macro_f1': float(scores.loc[selected]), 'field_presentations': selected * 370, 'updates': selected * 24})
        for condition in ['field_only', 'mixed']:
            cp = sd / f'{condition}_selected.pt'
            actual = sha(cp)
            assert actual == complete['checkpoint_sha256'][cp.name] == seal['seed_checkpoints'][str(seed)][cp.name]
            checkpoint = torch.load(cp, map_location='cpu', weights_only=True)
            assert checkpoint['selected_epoch'] == selected and checkpoint['selection_split'] == 'field_dev'
            assert checkpoint['initial_checkpoint_sha256'] == config['initial_sha256'] and checkpoint['config_hash'] == digest(config)
            assert checkpoint['selected_field_presentations'] == selected * 370 and checkpoint['selected_optimizer_updates'] == selected * 24
            assert checkpoint['selected_lab_presentations'] == (selected * 370 if condition == 'mixed' else 0)
            del checkpoint
            for split in ['field_dev', 'field_test']:
                frame = pd.read_csv(sd / f'predictions_{condition}_{split}.csv')
                canon = canonical[split]
                assert len(frame) == len(canon) and (not frame.duplicated(['source', 'row_id']).any())
                pairs = canon[['source', 'row_id', 'label_id']].merge(frame, on=['source', 'row_id'], how='outer', suffixes=('_canonical', ''), indicator=True, validate='one_to_one')
                assert pairs['_merge'].eq('both').all() and pairs.label_id.eq(pairs.label_id_canonical).all()
                probs = frame[[f'probability_label_{i}' for i in range(5)]].to_numpy()
                assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all() and np.allclose(probs.sum(1), 1, atol=1e-06)
                assert np.array_equal(probs.argmax(1), frame.predicted_label_id) and np.allclose(probs.max(1), frame.confidence, atol=1e-12)
                assert np.array_equal(frame.correct, frame.label_id.eq(frame.predicted_label_id))
                mf1 = f1_score(frame.label_id, frame.predicted_label_id, labels=range(5), average='macro', zero_division=0)
                if split == 'field_dev':
                    expected = history[history.condition.eq(condition) & history.epoch.eq(selected)].iloc[0]
                    assert abs(mf1 - expected.macro_f1) < 1e-12
                else:
                    assert frame.seed.eq(seed).all() and frame.condition.eq(condition).all() and frame.architecture.eq('resnet18').all()
                    assert frame.checkpoint_sha256.eq(actual).all() and frame.selected_epoch.eq(selected).all()
                    matrix = confusion_matrix(frame.label_id, frame.predicted_label_id, labels=range(5))
                    assert np.array_equal(matrix, reference['matrices'][seed, condition])
                    assert np.array_equal(matrix, pd.read_csv(sd / f'confusion_{condition}_field_test.csv', index_col=0).to_numpy())
                    summary.append(dict(seed=seed, condition=condition, architecture='resnet18', n=len(frame), correct=int(frame.correct.sum()), accuracy=accuracy_score(frame.label_id, frame.predicted_label_id), macro_f1=mf1))
                    prec, rec, f1, support = precision_recall_fscore_support(frame.label_id, frame.predicted_label_id, labels=range(5), zero_division=0)
                    class_rows.extend((dict(seed=seed, condition=condition, label_id=i, precision=prec[i], recall=rec[i], f1=f1[i], support=int(support[i])) for i in range(5)))
    computed = pd.DataFrame(summary)
    saved = pd.read_csv(run / 'field_test_summary.csv')
    for col in ['n', 'correct', 'accuracy', 'macro_f1']:
        assert np.allclose(computed.set_index(['seed', 'condition'])[col].sort_index(), saved.set_index(['seed', 'condition'])[col].sort_index(), atol=1e-12)
    computed_classes = pd.DataFrame(class_rows).set_index(['seed', 'condition', 'label_id']).sort_index()
    saved_classes = pd.read_csv(run / 'field_test_per_class.csv').set_index(['seed', 'condition', 'label_id']).sort_index()
    for col in ['precision', 'recall', 'f1', 'support']:
        assert np.allclose(computed_classes[col], saved_classes[col], atol=1e-12)
    assert files == {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob('*')) if p.is_file()}
    result = {'status': 'passed', 'experiment_id': run.name, 'seeds': [42, 43, 44], 'file_sha256': files, 'config_hash': digest(config), 'initial_state_hash': initial_digest, 'selection': selection, 'checks': {'canonical_source_rows_and_labels': True, 'prediction_probabilities_and_correctness': True, 'six_test_summary_rows': True, 'thirty_class_rows': True, 'six_confusion_matrices': True, 'six_dev_predictions_match_selected_history': True, 'all_training_batch_ids_and_tensor_hashes_match': True, 'eight_presentations_per_field_row': True, 'full_budget_2960_field_192_updates_2960_added_lab': True, 'predetermined_common_epoch_selection_from_full_precision_history': True, 'six_selected_checkpoint_hashes_match_completion_and_seal': True, 'selected_checkpoint_metadata_and_baseline_hashes': True, 'training_source_split_and_asset_hashes': True, 'identical_initial_state_hash_matches_local_lab_baseline': True}, 'limitations': ['Artifacts verify recorded protocol consistency; they do not prove the absence of undocumented human inspection.', 'Historical 47-image Field Test and three seeds sharing the same images remain dependent observations.'], 'raw_artifacts_modified': False, 'training_or_inference_executed': False}
    (out / 'imported_artifact_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    computed.to_csv(out / 'recomputed_from_predictions.csv', index=False)
    print('PASSED original artifact audit; selected epochs:', [r['selected_epoch'] for r in selection])
    print(computed.to_string(index=False))
    print('Recorded', len(files), 'input file hashes.')
if __name__ == '__main__':
    main()
