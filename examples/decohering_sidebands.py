"""Weak synchrotron sidebands next to a decohering betatron line.

The signal is the closed-form centroid of a kicked Gaussian beam with
chromaticity (sidebands at Q +- Q_s, about 1 % of the main line) and amplitude
detuning (the main line decays and drifts in frequency within the record), with
the parameters of a KARA tracking study. The exact sideband ratio
(A_1 + A_-1) / A_0 = 2 I_1(s^2) / I_0(s^2) is known, so each analysis option can
be scored. Only NumPy is needed.

A decohering line is not a single constant-amplitude line: after it is
subtracted, a residual remains within one FFT bin of it. By default NAFF stops
there (or, with tol >= 1, merges the residual into the main line), so the
residual's leakage biases the weak sidebands. Two remedies:
  close_terms="add"      absorb the residual with additional cleaning terms
  normalize_envelope()   divide out the slow decay before the analysis

Run: python examples/decohering_sidebands.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import PyNAFF as pnf  # noqa: E402

QS, Q0, S2, DETUNING = 0.01616, 0.79470, 0.0193, 1.25e-4
F0 = 1.0 - Q0


def centroid(kick, samples):
    theta = 2.0 * np.pi * DETUNING * samples
    chromatic = np.exp(-2.0 * S2 * np.sin(np.pi * QS * samples) ** 2)
    detuning = (1.0 - 1.0j * theta) ** -2 * np.exp(
        0.5 * kick**2 * 1.0j * theta / (1.0 - 1.0j * theta)
    )
    return np.real(chromatic * detuning * np.exp(2.0j * np.pi * Q0 * samples))


def exact_ratio():
    phase = np.linspace(0.0, 2.0 * np.pi, 20001)[:-1]
    chromatic = np.exp(-S2 * (1.0 - np.cos(phase)))
    return 2.0 * abs(np.mean(chromatic * np.exp(-1.0j * phase))) / abs(np.mean(chromatic))


def ratio(result):
    frequency, amplitude = np.abs(result[:, 1]), result[:, 2]
    main = np.abs(frequency - F0) < 0.5 * QS
    if not main.any():
        return np.nan
    q0 = frequency[main][np.argmax(amplitude[main])]
    sidebands = []
    for sign in (1, -1):
        near = np.abs(frequency - (q0 + sign * QS)) < 0.3 * QS
        if not near.any():
            return np.nan
        sidebands.append(amplitude[near].max())
    return sum(sidebands) / amplitude[main].max()


OPTIONS = {
    "default": lambda s, n: pnf.naff(s, turns=n, nterms=30, warnings=False),
    "tol=2": lambda s, n: pnf.naff(s, turns=n, nterms=100, tol=2.0, warnings=False),
    'close_terms="add"': lambda s, n: pnf.naff(
        s, turns=n, nterms=30, close_terms="add", warnings=False
    ),
    "normalize_envelope": lambda s, n: pnf.naff(
        pnf.normalize_envelope(s, int(round(1 / QS)), band=(F0 - 0.08, F0 + 0.08)),
        turns=n,
        nterms=30,
        warnings=False,
    ),
}

if __name__ == "__main__":
    expected = exact_ratio()
    print(f"exact sideband ratio {expected:.5f}; error of the measured ratio [%]")
    print("synchrotron periods:        3       4       6       8")
    for kick in (0.5, 1.5, 3.0):
        print(f"kick {kick} sigma")
        for name, run in OPTIONS.items():
            row = []
            for periods in (3, 4, 6, 8):
                turns = int(round(periods / QS))
                signal = centroid(kick, np.arange(turns + 1))
                value = ratio(run(signal - signal.mean(), turns))
                row.append(f"{100 * (value / expected - 1):+7.2f}" if np.isfinite(value) else "   stop")
            print(f"  {name:20s} " + " ".join(row))
