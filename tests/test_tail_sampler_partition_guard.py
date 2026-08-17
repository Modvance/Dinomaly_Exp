import os
import sys
import unittest

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tail_sampler import (
    PLATEAU_GAP_GUARD_V1,
    build_tail_sampler,
    predict_few_shot_class_samples,
    predict_num_samples_per_class,
)


def _expand_counts(counts):
    return torch.cat([
        torch.full((count,), float(support), dtype=torch.float32)
        for support, count in counts.items()
    ])


class TailSamplerPartitionGuardTest(unittest.TestCase):
    def test_repairs_k1_repeated_401_boundary(self):
        class_sizes = _expand_counts({
            1: 4,
            401: 939,
            402: 4,
            403: 1,
            483: 512,
            484: 19,
            485: 5,
            487: 32,
            488: 1,
            804: 804,
            805: 280,
            806: 34,
            807: 5,
            808: 31,
            809: 717,
            810: 22,
            811: 1,
        })
        original = class_sizes.clone()

        legacy = predict_few_shot_class_samples(class_sizes, percentile=0.15)
        guarded, diagnostics = predict_few_shot_class_samples(
            class_sizes,
            percentile=0.15,
            partition_guard=PLATEAU_GAP_GUARD_V1,
            return_partition_diagnostics=True,
        )

        self.assertEqual(int(legacy.sum()), 943)
        self.assertEqual(int(guarded.sum()), 4)
        self.assertEqual(legacy.dtype, torch.long)
        self.assertEqual(guarded.shape, class_sizes.shape)
        self.assertTrue(torch.equal(class_sizes, original))
        self.assertFalse(bool(guarded[class_sizes == 401].any()))
        self.assertTrue(diagnostics['triggered'])
        self.assertEqual(diagnostics['original_cutoff'], 401.0)
        self.assertEqual(diagnostics['effective_cutoff'], 1.0)
        self.assertEqual(diagnostics['boundary_multiplicity'], 2)
        self.assertEqual(diagnostics['num_removed_by_guard'], 939)

    def test_repairs_k4_before_intermediate_support_component(self):
        class_sizes = _expand_counts({
            3: 20,
            284: 3,
            376: 2,
            398: 13,
            399: 3,
            400: 1,
            401: 921,
            402: 1,
            810: 7,
            812: 583,
            813: 256,
            814: 4,
            815: 92,
            816: 4,
            818: 8,
            1606: 149,
            1607: 1621,
            1610: 120,
        })

        inferred = predict_num_samples_per_class(class_sizes).sort()[0]
        self.assertEqual(
            inferred.tolist(),
            [3, 3, 3, 3, 3, 3, 97, 401, 401, 764, 2127],
        )

        legacy = predict_few_shot_class_samples(class_sizes, percentile=0.15)
        guarded, diagnostics = predict_few_shot_class_samples(
            class_sizes,
            percentile=0.15,
            partition_guard=PLATEAU_GAP_GUARD_V1,
            return_partition_diagnostics=True,
        )

        self.assertEqual(int(legacy.sum()), 963)
        self.assertEqual(int(guarded.sum()), 20)
        self.assertEqual(diagnostics['original_cutoff'], 401.0)
        self.assertEqual(diagnostics['effective_cutoff'], 3.0)
        self.assertEqual(diagnostics['gap_upper'], 97.0)
        self.assertEqual(diagnostics['num_removed_by_guard'], 943)

    def test_repeated_lowest_support_platform_is_preserved(self):
        class_sizes = _expand_counts({5: 10, 100: 200})
        legacy = predict_few_shot_class_samples(class_sizes, percentile=0.15)
        guarded, diagnostics = predict_few_shot_class_samples(
            class_sizes,
            percentile=0.15,
            partition_guard=PLATEAU_GAP_GUARD_V1,
            return_partition_diagnostics=True,
        )

        self.assertTrue(torch.equal(legacy, guarded))
        self.assertEqual(int(guarded.sum()), 10)
        self.assertFalse(diagnostics['triggered'])
        self.assertEqual(diagnostics['original_cutoff'], 5.0)
        self.assertEqual(diagnostics['gap_lower'], 5.0)
        self.assertEqual(diagnostics['boundary_multiplicity'], 2)

    def test_guard_is_scoped_to_canonical_adaptive_trim_mode(self):
        adaptive, _ = build_tail_sampler('adaptive', plateau_gap_guard=True)
        trim_legacy, _ = build_tail_sampler('adaptive_trim_mode')
        trim_guarded, _ = build_tail_sampler(
            'adaptive_trim_mode',
            plateau_gap_guard=True,
        )

        self.assertIsNone(adaptive.partition_guard)
        self.assertIsNone(trim_legacy.partition_guard)
        self.assertEqual(trim_guarded.partition_guard, PLATEAU_GAP_GUARD_V1)
        with self.assertRaises(ValueError):
            build_tail_sampler(
                'adaptive_trim_mode',
                threshold_type='double_max_step',
                vote_type='none',
                plateau_gap_guard=True,
            )


if __name__ == '__main__':
    unittest.main()
