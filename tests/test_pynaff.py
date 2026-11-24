import unittest

import numpy as np

import PyNAFF as pnf


class NaffTests(unittest.TestCase):
    def setUp(self):
        self.turns = 300
        self.samples = np.arange(501)

    def test_single_bpm_preserves_two_dimensional_result(self):
        signal = 2.5 * np.sin(2.0 * np.pi * 0.12345 * self.samples)
        result = pnf.naff(signal, turns=self.turns)
        self.assertEqual(result.shape, (1, 5))
        self.assertAlmostEqual(result[0, 1], 0.12345, places=6)
        self.assertAlmostEqual(result[0, 2], 1.25, places=5)

    def test_multiple_bpms_are_processed_by_column(self):
        tunes = np.array([0.12345, 0.27123, 0.34876])
        amplitudes = np.array([1.0, 2.0, 0.5])
        data = np.column_stack(
            [
                amplitude * np.sin(2.0 * np.pi * tune * self.samples)
                for tune, amplitude in zip(tunes, amplitudes)
            ]
        )
        result = pnf.naff(data, turns=self.turns)
        self.assertEqual(result.shape, (3, 1, 5))
        np.testing.assert_allclose(result[:, 0, 1], tunes, atol=2.0e-6)
        np.testing.assert_allclose(
            result[:, 0, 2], amplitudes / 2.0, atol=1.0e-4
        )

    def test_multiple_terms_are_extracted_independently(self):
        signals = []
        expected = []
        for first, second in ((0.11, 0.31), (0.17, 0.37)):
            signals.append(
                2.0 * np.sin(2.0 * np.pi * first * self.samples)
                + 0.5 * np.sin(2.0 * np.pi * second * self.samples)
            )
            expected.append((first, second))
        result = pnf.naff(
            np.column_stack(signals), turns=self.turns, nterms=2
        )
        self.assertEqual(result.shape, (2, 2, 5))
        np.testing.assert_allclose(result[:, :, 1], expected, atol=1.0e-4)

    def test_skip_turns_selects_the_requested_interval(self):
        data = np.zeros(601)
        data[100:] = np.sin(
            2.0 * np.pi * 0.23456 * np.arange(data.size - 100)
        )
        result = pnf.naff(
            data, turns=self.turns, skipTurns=100, nterms=1
        )
        self.assertAlmostEqual(result[0, 1], 0.23456, places=5)

    def test_complex_input_can_find_negative_frequency(self):
        signal = 3.0 * np.exp(-2.0j * np.pi * 0.2 * self.samples)
        result = pnf.naff(
            signal,
            turns=self.turns,
            nterms=1,
            getFullSpectrum=True,
        )
        self.assertAlmostEqual(result[0, 1], -0.2, places=7)
        self.assertAlmostEqual(result[0, 2], 3.0, places=6)

    def test_dc_component_is_not_double_subtracted(self):
        signal = np.full(self.samples.shape, 3.0)
        result = pnf.naff(
            signal, turns=self.turns, nterms=1, warnings=False
        )
        self.assertAlmostEqual(result[0, 1], 0.0, places=12)
        self.assertAlmostEqual(result[0, 2], 3.0, places=12)

    def test_multi_bpm_result_matches_individual_calls(self):
        data = np.column_stack(
            [
                np.sin(2.0 * np.pi * 0.13 * self.samples),
                np.sin(2.0 * np.pi * 0.29 * self.samples),
            ]
        )
        batch = pnf.naff(
            data, turns=self.turns, nterms=2, warnings=False
        )
        for bpm in range(data.shape[1]):
            individual = pnf.naff(
                data[:, bpm],
                turns=self.turns,
                nterms=2,
                warnings=False,
            )
            np.testing.assert_allclose(
                batch[bpm, : len(individual)], individual
            )

    def test_non_finite_input_is_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            data = np.zeros(self.turns + 1)
            data[-1] = invalid
            with self.assertRaisesRegex(ValueError, "finite"):
                pnf.naff(data, turns=self.turns)

    def test_upstream_compatibility_options_are_accepted(self):
        signal = np.sin(2.0 * np.pi * 0.2 * self.samples)
        result = pnf.naff(
            signal,
            turns=self.turns,
            tol=1.0e-3,
            warnings=False,
        )
        self.assertAlmostEqual(result[0, 1], 0.2, places=6)

    def test_legacy_large_duplicate_tolerance_is_accepted(self):
        signal = np.sin(2.0 * np.pi * 0.2 * self.samples)

        result = pnf.naff(
            signal,
            turns=self.turns,
            tol=10000,
            warnings=False,
        )

        self.assertAlmostEqual(result[0, 1], 0.2, places=6)

    def test_invalid_boolean_options_are_rejected(self):
        signal = np.sin(2.0 * np.pi * 0.2 * self.samples)
        with self.assertRaisesRegex(ValueError, "getFullSpectrum"):
            pnf.naff(signal, getFullSpectrum="no")
        with self.assertRaisesRegex(ValueError, "tol"):
            pnf.naff(signal, tol=True)

    def test_high_window_order_does_not_overflow(self):
        signal = np.sin(2.0 * np.pi * 0.2 * self.samples)
        result = pnf.naff(
            signal, turns=self.turns, window=100, warnings=False
        )
        self.assertTrue(np.all(np.isfinite(result)))

    def test_larger_tolerance_can_continue_past_duplicate_residual(self):
        frequencies = [
            0.10,
            0.104,
            0.108,
            0.15,
            0.19,
            0.23,
            0.27,
            0.31,
            0.35,
            0.39,
            0.43,
            0.47,
        ]
        amplitudes = np.geomspace(1.0, 0.01, len(frequencies))
        signal = sum(
            amplitude
            * np.sin(2.0 * np.pi * frequency * self.samples)
            for frequency, amplitude in zip(frequencies, amplitudes)
        )
        strict = pnf.naff(
            signal,
            turns=self.turns,
            nterms=20,
            tol=1.0e-4,
            warnings=False,
        )
        permissive = pnf.naff(
            signal,
            turns=self.turns,
            nterms=20,
            tol=0.1,
            warnings=False,
        )
        self.assertGreater(len(permissive), len(strict))

    def test_requires_turns_plus_one_observations(self):
        with self.assertRaisesRegex(ValueError, r"turns \+ 1"):
            pnf.naff(np.zeros(self.turns), turns=self.turns)


if __name__ == "__main__":
    unittest.main()
