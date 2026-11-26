"""CPU-only protocol checks. No model forward/backward or optimizer update."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src import controlled_field_mixed as controlled
from scripts.prepare_controlled_colab_assets import build_bundle


class ProtocolTests(unittest.TestCase):
    def test_all_declared_schedules_and_resume_replay(self):
        for architecture, expected_updates in [('resnet18', 192), ('vit_small', 376)]:
            config = controlled.default_config(architecture)
            for seed in config['seeds']:
                counts = np.zeros(config['field_rows'], dtype=int)
                updates, lab_count = 0, 0
                for epoch in range(1, 9):
                    plan = controlled.epoch_plan(config, seed, epoch, 5365)
                    replay = controlled.epoch_plan(config, seed, epoch, 5365)
                    lab_ids = []
                    for (field, lab), (field2, lab2) in zip(plan, replay):
                        np.testing.assert_array_equal(field, field2)
                        np.testing.assert_array_equal(lab, lab2)
                        self.assertEqual(len(field), len(lab))
                        counts[field] += 1
                        updates += 1
                        lab_count += len(lab)
                        lab_ids.extend(lab)
                    self.assertEqual(len(set(lab_ids)), 370)
                np.testing.assert_array_equal(counts, np.full(370, 8))
                self.assertEqual(updates, expected_updates)
                self.assertEqual(lab_count, 2960)

    def test_invalid_seed_budget_is_rejected(self):
        for seeds in [(43,), (42, 44), (42, 42), ()]:
            with self.assertRaises(ValueError):
                controlled.default_config('resnet18', seeds)

    def test_cuda_gate_refuses_local_training(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA required'):
                controlled.require_cuda()

    def test_augmentation_replays_without_consuming_global_cpu_rng(self):
        class FakeAssets:
            def image(self, row):
                return Image.fromarray(np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3))
        frame = pd.DataFrame({'row_id': [100, 101], 'label_id': [0, 1]})
        transform = transforms.Compose([transforms.RandomResizedCrop(16),
                                         transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        before = torch.get_rng_state().clone()
        a = controlled.augmented_batch(FakeAssets(), frame, [0, 1], transform, 42, 1, 'field')
        b = controlled.augmented_batch(FakeAssets(), frame, [0, 1], transform, 42, 1, 'field')
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(a[2], [100, 101])

    def test_bundle_is_self_contained_and_refuses_overwrite(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'bundle.zip'
            build_bundle(target)
            with zipfile.ZipFile(target) as archive:
                manifest = json.loads(archive.read('bundle_manifest.json'))
                self.assertEqual(len(manifest['file_sha256']), 8)
                for name, expected in manifest['file_sha256'].items():
                    import hashlib
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
                self.assertNotIn('.git', archive.namelist())
            with self.assertRaises(FileExistsError):
                build_bundle(target)

    def test_training_validation_rejects_tampered_evidence(self):
        # Synthetic audit evidence tests integrity checking, without training any model.
        config = controlled.default_config('resnet18', (42,))
        config['initial_sha256'] = 'synthetic-initial-hash'
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'config.json').write_text(json.dumps(config))
            seed_dir = run / 'seed_42'
            seed_dir.mkdir()
            rows, history = [], []
            for epoch in range(1, 9):
                plan = controlled.epoch_plan(config, 42, epoch, 5365)
                for batch, (field, lab) in enumerate(plan):
                    for condition in controlled.CONDITIONS:
                        rows.append(dict(epoch=epoch, batch=batch, condition=condition,
                                         field_row_ids=json.dumps(field.tolist()), field_tensor_sha256='synthetic',
                                         field_presentations=len(field), optimizer_updates=1,
                                         lab_presentations=len(lab) if condition == 'mixed' else 0))
            for epoch in range(9):
                for condition in controlled.CONDITIONS:
                    history.append(dict(epoch=epoch, condition=condition, macro_f1=.25))
            audit = pd.DataFrame(rows)
            audit.to_csv(seed_dir / 'batch_audit.csv', index=False)
            pd.DataFrame(history).to_csv(seed_dir / 'history.csv', index=False)
            pd.DataFrame({'source_row_id': range(370), 'field_only': [8]*370,
                          'mixed': [8]*370}).to_csv(seed_dir / 'field_exposure.csv', index=False)
            hashes = {}
            for condition in controlled.CONDITIONS:
                path = seed_dir / f'{condition}_selected.pt'
                torch.save({'selected_epoch': 0, 'selection_split': 'field_dev',
                            'config_hash': controlled.digest_json(config),
                            'initial_checkpoint_sha256': config['initial_sha256'],
                            'selected_field_presentations': 0, 'selected_optimizer_updates': 0,
                            'selected_lab_presentations': 0}, path)
                hashes[path.name] = controlled.sha256(path)
            complete = dict(seed=42, complete=True, initialization_identical=True, epochs=8,
                            field_presentations_per_condition=2960, optimizer_updates_per_condition=192,
                            additional_lab_presentations_mixed=2960, selected_common_epoch=0,
                            checkpoint_sha256=hashes)
            (seed_dir / 'complete.json').write_text(json.dumps(complete))
            self.assertTrue(controlled.validate_training(run)['matched_exposure_and_updates'])
            audit.loc[1, 'field_tensor_sha256'] = 'tampered'
            audit.to_csv(seed_dir / 'batch_audit.csv', index=False)
            with self.assertRaises(AssertionError):
                controlled.validate_training(run)


if __name__ == '__main__':
    unittest.main()
