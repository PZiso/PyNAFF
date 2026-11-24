import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

import PyNAFF as pnf


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.turns = 300
        self.samples = np.arange(self.turns + 1)

    def test_quadratic_remains_the_default(self):
        signal = np.sin(2.0 * np.pi * 0.123456789 * self.samples)

        default = pnf.naff(signal, turns=self.turns)
        explicit = pnf.naff(
            signal, turns=self.turns, optimizer="quadratic"
        )

        np.testing.assert_array_equal(default, explicit)

    def test_both_optimizers_recover_isolated_frequency(self):
        tune = 0.271234567
        signal = np.sin(2.0 * np.pi * tune * self.samples)

        for optimizer in ("quadratic", "brent"):
            with self.subTest(optimizer=optimizer):
                result = pnf.naff(
                    signal,
                    turns=self.turns,
                    optimizer=optimizer,
                )
                self.assertAlmostEqual(result[0, 1], tune, places=6)

    def test_brent_improves_objective_for_distorted_boundary_peak(self):
        turns = 30
        samples = np.arange(turns + 1)
        signal = np.sin(
            2.0 * np.pi * 0.4788205136454488 * samples
        ) + 1.4994520735101986 * np.sin(
            2.0 * np.pi * 0.4600503010454674 * samples
            + 2.1429396289211295
        )

        quadratic = pnf.naff(
            signal,
            turns=turns,
            optimizer="quadratic",
            warnings=False,
        )
        brent = pnf.naff(
            signal,
            turns=turns,
            optimizer="brent",
            warnings=False,
        )

        self.assertAlmostEqual(quadratic[0, 1], 0.5, places=12)
        self.assertGreater(brent[0, 2], quadratic[0, 2])
        self.assertGreater(brent[0, 1], 0.45)
        self.assertLess(brent[0, 1], 0.5)

    def test_brent_supports_multiple_bpms(self):
        tunes = np.array([0.12345, 0.27123, 0.34876])
        data = np.column_stack(
            [
                np.sin(2.0 * np.pi * tune * self.samples)
                for tune in tunes
            ]
        )

        result = pnf.naff(
            data,
            turns=self.turns,
            optimizer="brent",
        )

        self.assertEqual(result.shape, (3, 1, 5))
        np.testing.assert_allclose(result[:, 0, 1], tunes, atol=2.0e-6)

    def test_brent_supports_negative_frequencies(self):
        signal = np.exp(-2.0j * np.pi * 0.2 * self.samples)

        result = pnf.naff(
            signal,
            turns=self.turns,
            getFullSpectrum=True,
            optimizer="brent",
        )

        self.assertAlmostEqual(result[0, 1], -0.2, places=7)

    def test_invalid_optimizer_is_rejected(self):
        signal = np.sin(2.0 * np.pi * 0.2 * self.samples)

        with self.assertRaisesRegex(ValueError, "optimizer"):
            pnf.naff(signal, optimizer="newton")

    def test_import_from_package_directory_used_by_debug_notebook(self):
        package_directory = Path(__file__).parents[1] / "PyNAFF"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from PyNAFF import naff; "
                    "assert callable(naff); "
                    "print(naff.__module__)"
                ),
            ],
            cwd=package_directory,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "PyNAFF")


if __name__ == "__main__":
    unittest.main()
