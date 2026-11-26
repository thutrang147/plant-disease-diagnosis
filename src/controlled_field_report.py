"""Read-only presentation and reconciliation of Notebook 15's saved Colab evidence.

This module imports no training code and never loads models or writes to a run.
Original prediction exports supply the metrics. Earlier transcribed count CSVs
remain an independent reconciliation with the saved notebook outputs.
"""
from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ['Healthy', 'Early Blight', 'Late Blight', 'Bacterial Spot', 'Septoria Leaf Spot']
SEEDS = (42, 43, 44)
CONDITIONS = ('field_only', 'mixed')
NAMES = {'field_only': 'Field-only', 'mixed': 'Mixed'}
COLORS = {'field_only': '#4477AA', 'mixed': '#228833'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], [], None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag in ('th', 'td'):
            self.cell = ''

    def handle_data(self, text):
        if self.cell is not None:
            self.cell += text

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        elif tag == 'tr' and self.row:
            self.rows.append(self.row)


def _table(notebook, cell, output=0):
    parser = _TableParser()
    parser.feed(''.join(notebook['cells'][cell]['outputs'][output]['data']['text/html']))
    return parser.rows


def _viewer(notebook, cell, output):
    data = notebook['cells'][cell]['outputs'][output]['data'][
        'application/vnd.microsoft.datawrangler.viewer.v0+json']
    frame = pd.DataFrame(data['rows'], columns=[c['name'] for c in data['columns']])
    return frame.drop(columns='index', errors='ignore')


def metrics_from_matrix(matrix):
    """Calculate five-class metrics directly from integer confusion counts."""
    matrix = np.asarray(matrix)
    if matrix.shape != (5, 5) or (matrix < 0).any() or not np.equal(matrix, matrix.astype(int)).all():
        raise ValueError('Expected a nonnegative integer 5×5 confusion matrix.')
    true_count, predicted_count, correct = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    precision = np.divide(correct, predicted_count, out=np.zeros(5), where=predicted_count != 0)
    recall = np.divide(correct, true_count, out=np.zeros(5), where=true_count != 0)
    f1 = np.divide(2 * correct, true_count + predicted_count, out=np.zeros(5),
                   where=true_count + predicted_count != 0)
    summary = dict(n=int(matrix.sum()), correct=int(correct.sum()),
                   accuracy=float(correct.sum() / matrix.sum()), macro_f1=float(f1.mean()))
    classes = pd.DataFrame(dict(class_name=CLASSES, label_id=range(5), support=true_count,
                                correct=correct, predicted=predicted_count,
                                precision=precision, recall=recall, f1=f1))
    return summary, classes


def load_evidence(project_root):
    """Reconcile original predictions, count transcription and saved notebook tables."""
    root = Path(project_root)
    evidence = root / 'data/results/notebook15'
    provenance = json.loads((evidence / 'provenance.json').read_text())
    original = root / provenance['original_notebook']
    if sha256(original) != provenance['original_notebook_sha256']:
        raise ValueError('Original Colab notebook differs from the preserved source.')
    count_path = evidence / 'confusion_counts_from_notebook.csv'
    if sha256(count_path) != provenance['confusion_counts_sha256']:
        raise ValueError('Derived count file differs from its recorded source.')
    notebook = json.loads(original.read_text())
    counts = pd.read_csv(count_path)
    keys = ['seed', 'condition', 'true_label_id', 'predicted_label_id']
    assert not counts.duplicated(keys).any() and len(counts) == 150
    assert set(zip(counts.seed, counts.condition)) == {(s, c) for s in SEEDS for c in CONDITIONS}
    canonical = pd.read_csv(root / 'data/splits/field_test.csv')
    support = canonical.label_id.value_counts().reindex(range(5), fill_value=0).to_numpy()
    prediction_matrices = {}
    for record in provenance.get('prediction_exports', []):
        path = root / record['path']
        if sha256(path) != record['sha256']:
            raise ValueError(f'Original prediction export differs: {path.name}')
        predictions = pd.read_csv(path)
        checked = canonical[['source', 'row_id', 'label_id']].merge(
            predictions, on=['source', 'row_id'], how='outer', validate='one_to_one',
            suffixes=('_canonical', ''), indicator=True)
        assert len(checked) == 47 and checked['_merge'].eq('both').all()
        assert checked.label_id.eq(checked.label_id_canonical).all()
        assert checked.predicted_label_id.isin(range(5)).all()
        assert predictions.seed.eq(record['seed']).all()
        assert predictions.condition.eq(record['condition']).all()
        assert predictions.correct.eq(predictions.label_id.eq(predictions.predicted_label_id)).all()
        probabilities = predictions[[f'probability_label_{i}' for i in range(5)]].to_numpy()
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()
        assert np.allclose(probabilities.sum(1), 1, atol=1e-6)
        assert np.array_equal(probabilities.argmax(1), predictions.predicted_label_id)
        assert np.allclose(probabilities.max(1), predictions.confidence, atol=1e-12)
        matrix = np.zeros((5, 5), dtype=int)
        np.add.at(matrix, (checked.label_id.to_numpy(dtype=int),
                          checked.predicted_label_id.to_numpy(dtype=int)), 1)
        prediction_matrices[record['seed'], record['condition']] = matrix
    if prediction_matrices:
        assert set(prediction_matrices) == {(s, c) for s in SEEDS for c in CONDITIONS}
    for name, expected in provenance.get('preserved_record_sha256', {}).items():
        assert sha256(evidence / name) == expected
    matrices, rows, classes = {}, [], []
    for seed in SEEDS:
        for condition in CONDITIONS:
            frame = counts[counts.seed.eq(seed) & counts.condition.eq(condition)]
            matrix = frame.pivot(index='true_label_id', columns='predicted_label_id', values='images')
            matrix = matrix.reindex(index=range(5), columns=range(5)).to_numpy()
            if prediction_matrices:
                assert np.array_equal(matrix, prediction_matrices[seed, condition])
                matrix = prediction_matrices[seed, condition]
            assert np.array_equal(matrix.sum(1), support) and matrix.sum() == len(canonical) == 47
            score, by_class = metrics_from_matrix(matrix)
            matrices[seed, condition] = matrix
            rows.append(dict(seed=seed, condition=condition, **score))
            classes.append(by_class.assign(seed=seed, condition=condition))
    summary, per_class = pd.DataFrame(rows), pd.concat(classes, ignore_index=True)
    if prediction_matrices:
        saved = pd.read_csv(evidence / 'original_field_test_summary.csv').set_index(['seed', 'condition'])
        actual = summary.set_index(['seed', 'condition'])
        for metric in ('n', 'correct', 'accuracy', 'macro_f1'):
            np.testing.assert_allclose(actual[metric].sort_index(), saved[metric].sort_index(),
                                       rtol=0, atol=1e-12)
        saved = pd.read_csv(evidence / 'original_field_test_per_class.csv').set_index(
            ['seed', 'condition', 'label_id'])
        actual = per_class.set_index(['seed', 'condition', 'label_id'])
        for metric in ('support', 'precision', 'recall', 'f1'):
            np.testing.assert_allclose(actual[metric].sort_index(), saved[metric].sort_index(),
                                       rtol=0, atol=1e-12)
    cells = provenance['source_cells_zero_based']
    # Compare to the original, independently saved six-row summary and 30-row class table.
    summary_table = _table(notebook, cells['field_test_summary'])
    saved_summary = pd.DataFrame([row[1:] for row in summary_table[1:]], columns=summary_table[0][1:])
    for row in summary.itertuples():
        original_row = saved_summary[(saved_summary.seed == str(row.seed)) &
                                     (saved_summary.condition == row.condition)].iloc[0]
        assert original_row.architecture == 'resnet18'
        assert row.correct == int(original_row.correct) and row.n == int(original_row.n)
        for metric in ('accuracy', 'macro_f1'):
            assert f'{getattr(row, metric):.4f}' == original_row[metric]
    class_table = _table(notebook, cells['per_class'])
    saved_class = pd.DataFrame([row[1:] for row in class_table[1:]], columns=class_table[0][1:])
    for row in per_class.itertuples():
        original_row = saved_class[(saved_class.seed == str(row.seed)) &
                                   (saved_class.condition == row.condition) &
                                   (saved_class.class_name == row.class_name)].iloc[0]
        assert row.support == int(original_row.support)
        for metric in ('precision', 'recall'):
            assert f'{getattr(row, metric):.3f}' == original_row[metric]
    aggregate = summary.groupby('condition')[['accuracy', 'macro_f1']].agg(['mean', 'std'])
    saved_aggregate = _table(notebook, cells['field_test_summary'], output=1)
    for row in saved_aggregate[3:]:
        calculated = aggregate.loc[row[0]].to_numpy()
        assert [f'{v:.4f}' for v in calculated] == row[1:]
    paired = summary.pivot(index='seed', columns='condition', values=['accuracy', 'macro_f1'])
    delta = pd.DataFrame({m: paired[m]['mixed'] - paired[m]['field_only']
                          for m in ('accuracy', 'macro_f1')})
    saved_delta = _table(notebook, cells['field_test_summary'], output=2)
    for row in saved_delta[2:]:
        seed = int(row[0])
        assert f"{paired['macro_f1'].loc[seed, 'field_only']:.4f}" == row[1]
        assert f"{paired['macro_f1'].loc[seed, 'mixed']:.4f}" == row[2]
        assert f"{delta.loc[seed, 'macro_f1']:.4f}" == row[3]
    completed = _viewer(notebook, cells['training_completion_and_selection'], output=1).astype(int)
    assert completed.seed.tolist() == list(SEEDS)
    assert completed.field_presentations_per_condition.eq(370 * 8).all()
    assert completed.optimizer_updates_per_condition.eq(24 * 8).all()
    assert completed.selected_common_epoch.tolist() == [8, 7, 8]
    assert (completed.selected_field_presentations_per_condition == completed.selected_common_epoch * 370).all()
    assert (completed.selected_updates_per_condition == completed.selected_common_epoch * 24).all()
    if prediction_matrices:
        validation = json.loads((evidence / 'original_training_validation.json').read_text())
        imported = pd.DataFrame(validation['pairs'])[completed.columns].astype(int)
        pd.testing.assert_frame_equal(completed, imported)
    planned = _viewer(notebook, cells['planned_budget'], output=0).astype(int)
    assert planned['Seed'].tolist() == list(SEEDS)
    assert planned['Additional Mixed Lab presentations'].eq(2960).all()
    # The Early Blight row is also independently printed in the original notebook.
    early = _viewer(notebook, cells['early_blight_destinations'], output=0)
    for index, (seed, condition) in enumerate((s, c) for s in SEEDS for c in CONDITIONS):
        row = early.iloc[index]
        assert row.iloc[0] == f"(np.int64({seed}), '{condition}')"
        assert np.array_equal(row[CLASSES].astype(int).to_numpy(), matrices[seed, condition][1])
    return dict(provenance=provenance, summary=summary, per_class=per_class,
                aggregate=aggregate, paired=paired, delta=delta, completed=completed,
                matrices=matrices, planned=planned)


def verify_prediction_exports(project_root, run, evidence):
    """Optionally reconcile real CSV exports without inference or artifact writes.

    An absent directory is reported as missing, never replaced by invented predictions.
    """
    root, run = Path(project_root), Path(run)
    if not run.exists():
        return {'available': False, 'status': 'Not imported; notebook-output reconciliation only'}
    canonical = pd.read_csv(root / 'data/splits/field_test.csv')
    for seed in SEEDS:
        for condition in CONDITIONS:
            path = run / f'seed_{seed}/predictions_{condition}_field_test.csv'
            frame = pd.read_csv(path)  # Incomplete exports stop rather than silently fall back.
            assert len(frame) == 47 and not frame.duplicated(['source', 'row_id']).any()
            checked = canonical[['source', 'row_id', 'label_id']].merge(
                frame, on=['source', 'row_id'], how='outer', validate='one_to_one',
                suffixes=('_canonical', ''), indicator=True)
            assert checked['_merge'].eq('both').all()
            assert checked.label_id.eq(checked.label_id_canonical).all()
            assert checked.predicted_label_id.isin(range(5)).all()
            assert frame.seed.eq(seed).all() and frame.condition.eq(condition).all()
            matrix = np.zeros((5, 5), dtype=int)
            np.add.at(matrix, (checked.label_id.to_numpy(dtype=int),
                              checked.predicted_label_id.to_numpy(dtype=int)), 1)
            assert np.array_equal(matrix, evidence['matrices'][seed, condition])
    return {'available': True, 'status': 'All six prediction CSVs match the preserved counts'}
