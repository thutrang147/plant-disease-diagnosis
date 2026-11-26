"""Recompute cross-run comparisons from ID-aligned, saved predictions."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from src.evaluate_resolution_sensitivity import MODEL_NAMES, REGIMES, sha256_file

REFERENCE_RUN = Path('outputs/external_evaluation/tomato_field_eval_v1/20261125_163347_4ca8183f')
UPSTREAM_KEYS = dict(zip(MODEL_NAMES, (
    'model1_resnet18_lab_base', 'model2_vit_small_lab_base',
    'model3_resnet18_geom_mixed', 'model4_vit_small_mixed')))
UPSTREAM_PROBS = ['prob_healthy', 'prob_early_blight', 'prob_late_blight',
                  'prob_bacterial_spot', 'prob_septoria_leaf_spot']
PROBS = [f'probability_label_{i}' for i in range(5)]


def indexed(frame):
    assert frame.image_id.is_unique
    return frame.set_index('image_id').sort_index()


def metrics(frame):
    return {'accuracy': accuracy_score(frame.label_id, frame.predicted_label_id),
            'macro_f1': f1_score(frame.label_id, frame.predicted_label_id,
                                 labels=range(5), average='macro', zero_division=0)}


def paired(a, b):
    a, b = indexed(a), indexed(b)
    assert a.index.equals(b.index) and a.label_id.equals(b.label_id)
    ac, bc = a.predicted_label_id.eq(a.label_id), b.predicted_label_id.eq(b.label_id)
    only_a, only_b = int((ac & ~bc).sum()), int((~ac & bc).sum())
    return {'both_correct': int((ac & bc).sum()), 'a_only': only_a, 'b_only': only_b,
            'both_wrong': int((~ac & ~bc).sum()),
            'p_exact_two_sided': binomtest(only_a, only_a+only_b, .5).pvalue if only_a+only_b else 1.}


def confidence_counts(frame):
    correct = frame.predicted_label_id.eq(frame.label_id)
    high = frame.confidence.ge(.90)
    wrong_high = int((~correct & high).sum())
    return {'errors': int((~correct).sum()), 'high_confidence_predictions': int(high.sum()),
            'high_confidence_errors': wrong_high,
            'share_of_errors_high_confidence': wrong_high/(~correct).sum() if (~correct).any() else np.nan,
            'error_rate_among_high_confidence': wrong_high/high.sum() if high.any() else np.nan}


def reconcile(root, evidence):
    """Check saved inference, preserve upstream values and write compact evidence."""
    root, evidence = Path(root), Path(evidence)
    upstream = evidence / 'upstream/outputs/external_confirmatory_eval'
    study = evidence / 'inference'
    rows, differences, numerical, stats, confidence, residuals = [], [], [], [], [], []
    all_predictions = {}
    for key, tag in UPSTREAM_KEYS.items():
        ours = indexed(pd.read_csv(root / REFERENCE_RUN / f'predictions_{key}.csv'))
        theirs = indexed(pd.read_csv(upstream / f'predictions_{tag}_external_v1.csv'))
        new = pd.read_csv(study / f'predictions_{key}.csv')
        raw = indexed(new.query("protocol == 'upstream_raw' and partition == 'external_field_v1' and regime == '1.00x_full_scene'"))
        canonical = indexed(new.query("protocol == 'canonical_exif' and regime == '1.00x_full_scene'"))
        assert ours.index.equals(theirs.index) and ours.index.equals(raw.index)
        assert ours.label_id.equals(theirs.label_id) and ours.sha256.equals(theirs.sha256)
        assert np.array_equal(raw.predicted_label_id, theirs.predicted_label_id)
        assert np.array_equal(canonical.predicted_label_id, ours.predicted_label_id)
        err_raw = float(np.abs(raw[PROBS].to_numpy()-theirs[UPSTREAM_PROBS].to_numpy()).max())
        err_ours = float(np.abs(canonical[PROBS].to_numpy()-ours[PROBS].to_numpy()).max())
        # GPU/CPU and six-decimal CSV precision can differ numerically.
        assert err_raw < 2e-5 and err_ours < 2e-5, (key, err_raw, err_ours)
        numerical.append({'model_key': key, 'upstream_max_abs_probability_difference': err_raw,
                          'canonical_max_abs_probability_difference': err_ours})
        for image_id in ours.index[ours.predicted_label_id.ne(theirs.predicted_label_id)]:
            differences.append({'model_key': key, 'image_id': image_id,
                                'original_filename': ours.loc[image_id, 'original_filename'],
                                'our_label': int(ours.loc[image_id, 'predicted_label_id']),
                                'upstream_label': int(theirs.loc[image_id, 'predicted_label_id']),
                                'exif_orientation': int(raw.loc[image_id, 'exif_orientation'])})
        for metric in ('accuracy', 'macro_f1'):
            old, other, repeat = metrics(ours)[metric], metrics(theirs)[metric], metrics(canonical)[metric]
            rows.append({'model': MODEL_NAMES[key], 'dataset_split': 'external_v1_n62', 'metric': metric,
                         'our_result': old, 'teammate_result': other, 'recomputed_result': repeat,
                         'difference_teammate_minus_ours': other-old,
                         'verified_cause': 'EXIF orientation before transform; reproduced both loaders with fixed local weights',
                         'recommended_canonical_result': old,
                         'evidence': f'{REFERENCE_RUN}/predictions_{key}.csv; {study.relative_to(root)}/predictions_{key}.csv'})
        for label, frame in [('canonical_exif', ours), ('upstream_raw', theirs)]:
            confidence.append(dict(protocol=label, model_key=key, **confidence_counts(frame)))
        all_predictions[key] = new
    summary = pd.read_csv(study / 'summary.csv')
    original = pd.read_csv(upstream / 'bacterial_spot_resolution_experiment.csv')
    for row in original.to_dict('records'):
        key = next(k for k, v in MODEL_NAMES.items() if v == row['model'])
        match = summary[summary.protocol.eq('upstream_raw') & summary.partition.eq(row['partition']) & summary.model_key.eq(key) & summary.regime.eq(row['regime'])]
        assert len(match) == 1
        repeat = match.iloc[0]
        for old, new in [('overall_n', 'n'), ('overall_accuracy', 'accuracy'), ('overall_macro_f1', 'macro_f1'),
                         ('bacterial_spot_n', 'bacterial_spot_n'), ('bacterial_spot_correct', 'bacterial_spot_correct'),
                         ('bacterial_spot_recall', 'bacterial_spot_recall'),
                         ('bacterial_spot_misclassified_as_septoria', 'bacterial_as_septoria'),
                         ('bacterial_spot_mean_true_prob', 'bacterial_mean_true_probability'),
                         ('bacterial_spot_mean_septoria_prob', 'bacterial_mean_septoria_probability')]:
            delta = float(repeat[new])-float(row[old])
            if old in ('bacterial_spot_mean_true_prob', 'bacterial_spot_mean_septoria_prob'):
                residuals.append(dict(model_key=key, partition=row['partition'], regime=row['regime'],
                                      metric=old, upstream=row[old], reproduced=float(repeat[new]),
                                      difference=delta, matches_saved_precision=abs(delta) < .000051))
            else:
                assert abs(delta) < .000051, (key, row['regime'], old, repeat[new], row[old])
    for partition in ('external_field_v1', 'plantdoc_field_test'):
        for a, b in [('resnet_mixed', 'resnet_lab'), ('vit_mixed', 'vit_lab'),
                     ('resnet_mixed', 'vit_mixed'), ('vit_lab', 'resnet_lab')]:
            fa = all_predictions[a].query('protocol == "upstream_raw" and partition == @partition and regime == "1.00x_full_scene"')
            fb = all_predictions[b].query('protocol == "upstream_raw" and partition == @partition and regime == "1.00x_full_scene"')
            stats.append(dict(partition=partition, model_a=a, model_b=b, **paired(fa,fb)))
    upstream_summary = pd.read_csv(upstream / 'confirmatory_summary_all_partitions.csv')
    for old in upstream_summary.to_dict('records'):
        key = next(k for k, v in MODEL_NAMES.items() if v == old['model'])
        repeat = summary[summary.protocol.eq('upstream_raw') & summary.partition.eq(old['partition'])
                         & summary.model_key.eq(key) & summary.regime.eq(REGIMES[0])].iloc[0]
        for field in ('n', 'correct', 'accuracy', 'macro_f1'):
            assert abs(repeat[field]-old[field]) < .000051, (key, field)
    upstream_stats = pd.read_csv(upstream / 'confirmatory_paired_mcnemar_tests.csv')
    for old, repeat in zip(upstream_stats.to_dict('records'), stats):
        assert old['partition'] == repeat['partition']
        assert old['model_a'] == MODEL_NAMES[repeat['model_a']]
        assert old['model_b'] == MODEL_NAMES[repeat['model_b']]
        for old_field, new_field in [('both_correct','both_correct'), ('only_a_correct','a_only'),
                                     ('only_b_correct','b_only'), ('both_wrong','both_wrong'),
                                     ('exact_mcnemar_p','p_exact_two_sided')]:
            assert abs(old[old_field]-repeat[new_field]) < 0.00000051
    # Canonical external comparison is a different, explicitly recorded protocol.
    fa = pd.read_csv(root / REFERENCE_RUN / 'predictions_resnet_mixed.csv')
    fb = pd.read_csv(root / REFERENCE_RUN / 'predictions_resnet_lab.csv')
    stats.append(dict(partition='canonical_external_v1', model_a='resnet_mixed', model_b='resnet_lab', **paired(fa,fb)))
    result = {'metric_reconciliation': rows, 'prediction_disagreements': differences,
              'probability_reproduction': numerical, 'paired_statistics': stats, 'confidence_denominators': confidence, 'resolution_probability_residuals': residuals}
    for name, records in result.items():
        pd.DataFrame(records).to_csv(evidence / f'{name}.csv', index=False)
    check = {'all_62_images_identical': True, 'all_labels_identical': True,
             'canonical_and_upstream_external_labels_reproduced': True,
             'all_32_upstream_resolution_accuracy_f1_and_counts_reproduced_to_saved_precision': True,
             'resolution_probability_mean_residual_max': max(abs(r['difference']) for r in residuals),
             'resolution_probability_mean_residual_cause': 'Unresolved; upstream lacks per-image multi-crop probabilities and complete environment provenance.',
             'probability_tolerance': 2e-5,
             'unresolved_upstream_checkpoint_hash': 'Not recorded upstream; local frozen hashes reproduce outputs, which is consistency evidence rather than proof of remote file identity.'}
    (evidence / 'reconciliation_validation.json').write_text(json.dumps(check, indent=2))
    return result


def audit_duplicate_scope(root, evidence):
    """Extend the upstream DCT-pHash screen from 476 to all 523 PlantDoc rows."""
    from datasets import Dataset, concatenate_datasets
    from scipy.fftpack import dct
    from PIL import Image
    from src.evaluate_resolution_sensitivity import load_external, pixel_sha256

    def phash(image):
        pixels = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        low = dct(dct(pixels, axis=0), axis=1)[:8, :8]
        bits = (low > np.median(low)).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        return value

    cache = Path.home() / '.cache/huggingface/datasets/geraldmc___plantdoc-full/default/0.0.0/66df4b49a58d9c676aa32d9ab0c0d3fff56fb8ac'
    source = concatenate_datasets([Dataset.from_file(str(p)) for p in sorted(cache.glob('*.arrow'))])
    references = []
    for split in ('field_adapt_train', 'field_dev', 'field_quarantined_train', 'field_test'):
        manifest = pd.read_csv(root / f'data/splits/{split}.csv')
        for row in manifest.itertuples():
            record = source[int(row.row_id)]
            assert record['class_label'] == row.original_class
            image = record['image'].convert('RGB')
            references.append(dict(split=split, row_id=int(row.row_id),
                                   pixel_sha256=pixel_sha256(image), phash=phash(image)))
    assert len(references) == 523
    assert len({r['row_id'] for r in references}) == 523
    results = []
    for orientation in (False, True):
        images, metadata = load_external(root, orientation)
        hashes = [phash(im) for im in images]
        for i, row in enumerate(metadata.to_dict('records')):
            distances = [(hashes[i] ^ ref['phash']).bit_count() for ref in references]
            j = int(np.argmin(distances))
            results.append(dict(protocol='canonical_exif' if orientation else 'upstream_raw',
                                image_id=row['image_id'], nearest_plantdoc_row=references[j]['row_id'],
                                nearest_partition=references[j]['split'], minimum_phash_distance=distances[j],
                                exact_pixel_matches=sum(row['pixel_sha256']==ref['pixel_sha256'] for ref in references),
                                matches_at_distance_le4=sum(d<=4 for d in distances),
                                within_external_matches_le4=sum((hashes[i]^h).bit_count()<=4 for k,h in enumerate(hashes) if k!=i)))
    frame = pd.DataFrame(results)
    frame.to_csv(evidence / 'extended_duplicate_screen.csv', index=False)
    return frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('outputs/research_integration/20261126_reconciliation'))
    args = parser.parse_args()
    reconcile(Path.cwd(), args.evidence.resolve())
    audit_duplicate_scope(Path.cwd(), args.evidence.resolve())
