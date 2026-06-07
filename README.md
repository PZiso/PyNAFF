# PyNAFF

Authors:

* Foteini Asvesta (fasvesta .at. cern .dot. ch)
* Nikos Karastathis (nkarast .at. cern .dot. ch)
* Panagiotis Zisopoulos (pzisopou .at. cern .dot. ch)

A Python implementation of J. Laskar's Numerical Analysis of Fundamental
Frequencies (NAFF) method.

## Installation

```bash
python -m pip install PyNAFF
```

## Single BPM

```python
import numpy as np
import PyNAFF as pnf

t = np.arange(3001)
signal = np.sin(2.0 * np.pi * 0.12345 * t)
result = pnf.naff(signal, turns=500, nterms=1, window=1)

# Each row is:
# [order, frequency, amplitude, real amplitude, imaginary amplitude]
frequency = result[0, 1]
```

`turns` is the number of integration intervals, so the input must contain at
least `turns + 1` observations. For real sinusoids, the reported amplitude is
the magnitude of one complex Fourier coefficient, equal to half the sinusoid's
peak amplitude.

## Multiple BPMs

Place observations on axis 0 and BPMs on axis 1:

```python
signals = np.column_stack([
    np.sin(2.0 * np.pi * 0.12345 * t),
    2.0 * np.sin(2.0 * np.pi * 0.27123 * t),
])
results = pnf.naff(signals, turns=500, nterms=1)

# results.shape == (2 BPMs, 1 term, 5 values)
frequencies = results[:, 0, 1]
amplitudes = results[:, 0, 2]
```

For multi-BPM input, unused term rows are filled with `NaN` when extraction
for a BPM stops before `nterms`.

The `tol` option controls duplicate residual handling as a fraction of one FFT
bin. If NAFF stops early because a residual peak is very close to a previously
extracted frequency, increasing `tol` can let it remove that residual and
continue to weaker frequencies. The default is `1e-4`; large values can also
turn spectral leakage into spurious frequencies, so compare results across
several values. `tol` does not control optimizer convergence. Because duplicate
testing is applied only within one FFT bin, every value `tol >= 1` has the same
classification effect, including legacy choices such as `tol=10000`.
`nterms` is an upper bound, not a guaranteed result count.

For real input, prefer `getFullSpectrum=False`. A full spectrum contains both
positive and negative conjugate frequencies, and each one occupies a result
row.

## Frequency optimizer

The FFT peak can be refined with either optimizer:

```python
fast = pnf.naff(signals, turns=500, optimizer="quadratic")
bounded = pnf.naff(signals, turns=500, optimizer="brent")
```

`quadratic` is the backward-compatible default and repeatedly fits three
nearby objective values. `brent` keeps the search inside the neighboring FFT
bin and falls back to golden-section steps when parabolic interpolation is
unreliable. Brent can be more robust for distorted peaks, while either method
can be preferable for a particular signal and window.
