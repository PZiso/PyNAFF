import unittest

import numpy as np

import PyNAFF as pnf


QS = 0.01616
Q0 = 0.79470
F0 = 1.0 - Q0
S2 = 0.0193           # squared modulation index of the chromatic sidebands
DETUNING = 1.25e-4    # mu_xx * eps_x


def decohering_centroid(kick, samples):
    """Centroid of a kicked Gaussian beam with chromaticity and amplitude
    detuning (closed form): chromatic factor times the Meller factor."""
    theta = 2.0 * np.pi * DETUNING * samples
    chromatic = np.exp(-2.0 * S2 * np.sin(np.pi * QS * samples) ** 2)
    detuning = (1.0 - 1.0j * theta) ** -2 * np.exp(
        0.5 * kick**2 * 1.0j * theta / (1.0 - 1.0j * theta)
    )
    return np.real(chromatic * detuning * np.exp(2.0j * np.pi * Q0 * samples))


def exact_sideband_ratio():
    """(A_1 + A_-1) / A_0 of the chromatic factor, from its Fourier
    coefficients over one synchrotron period (2 I_1(s^2) / I_0(s^2))."""
    phase = np.linspace(0.0, 2.0 * np.pi, 20001)[:-1]
    chromatic = np.exp(-S2 * (1.0 - np.cos(phase)))
    c0 = abs(np.mean(chromatic))
    c1 = abs(np.mean(chromatic * np.exp(-1.0j * phase)))
    return 2.0 * c1 / c0


def sideband_ratio(result, qs=QS):
    frequency, amplitude = np.abs(result[:, 1]), result[:, 2]
    main = np.abs(frequency - F0) < 0.5 * qs
    q0 = frequency[main][np.argmax(amplitude[main])]
    a0 = amplitude[main].max()
    sidebands = []
    for sign in (1, -1):
        near = np.abs(frequency - (q0 + sign * qs)) < 0.3 * qs
        if not near.any():
            return np.nan
        sidebands.append(amplitude[near].max())
    return sum(sidebands) / a0


class DecoheringSignalTests(unittest.TestCase):
    def setUp(self):
        self.turns = int(round(6 / QS))
        self.samples = np.arange(self.turns + 1)
        self.signal = decohering_centroid(3.0, self.samples)
        self.signal = self.signal - self.signal.mean()
        self.expected = exact_sideband_ratio()

    def test_default_stops_before_the_weak_sidebands(self):
        result = pnf.naff(
            self.signal, turns=self.turns, nterms=30, warnings=False
        )
        self.assertTrue(np.isnan(sideband_ratio(result)))

    def test_close_terms_add_recovers_sidebands_of_decaying_line(self):
        result = pnf.naff(
            self.signal,
            turns=self.turns,
            nterms=30,
            close_terms="add",
            warnings=False,
        )
        ratio = sideband_ratio(result)
        self.assertLess(abs(ratio / self.expected - 1.0), 0.02)

    def test_envelope_normalization_recovers_sidebands(self):
        normalized = pnf.normalize_envelope(
            self.signal, int(round(1 / QS)), band=(F0 - 0.08, F0 + 0.08)
        )
        result = pnf.naff(normalized, turns=self.turns, nterms=30, warnings=False)
        ratio = sideband_ratio(result)
        self.assertLess(abs(ratio / self.expected - 1.0), 0.02)

    def test_close_terms_add_does_not_change_stationary_signals(self):
        samples = np.arange(501)
        signal = 1.0 * np.sin(2.0 * np.pi * 0.11 * samples) + 0.3 * np.cos(
            2.0 * np.pi * 0.31 * samples
        )
        default = pnf.naff(signal, turns=300, nterms=2)
        added = pnf.naff(signal, turns=300, nterms=2, close_terms="add")
        np.testing.assert_allclose(default, added)

    def test_close_terms_are_reported_after_regular_terms(self):
        result = pnf.naff(
            self.signal,
            turns=self.turns,
            nterms=30,
            close_terms="add",
            warnings=False,
        )
        np.testing.assert_array_equal(result[:, 0], np.arange(len(result)))
        self.assertAlmostEqual(result[0, 1], F0, delta=0.5 / self.turns)

    def test_normalize_envelope_flattens_a_decaying_sinusoid(self):
        samples = np.arange(2000)
        signal = np.exp(-samples / 800.0) * np.cos(2.0 * np.pi * 0.21 * samples)
        normalized = pnf.normalize_envelope(signal, 60)
        result = pnf.naff(normalized, turns=1800, nterms=1)
        self.assertAlmostEqual(result[0, 1], 0.21, places=5)
        self.assertAlmostEqual(2.0 * result[0, 2], 1.0, delta=0.02)

    def test_normalize_envelope_keeps_bpm_columns(self):
        samples = np.arange(600)
        data = np.column_stack(
            [np.cos(2.0 * np.pi * q * samples) for q in (0.2, 0.3)]
        )
        self.assertEqual(pnf.normalize_envelope(data, 50).shape, data.shape)

    def test_invalid_options_are_rejected(self):
        with self.assertRaises(ValueError):
            pnf.naff(self.signal, turns=self.turns, close_terms="merge")
        with self.assertRaises(ValueError):
            pnf.normalize_envelope(self.signal, 0)
        with self.assertRaises(ValueError):
            pnf.normalize_envelope(self.signal, len(self.signal))
        with self.assertRaises(ValueError):
            pnf.normalize_envelope(self.signal, 60, band=(0.3, 0.2))


if __name__ == "__main__":
    unittest.main()
