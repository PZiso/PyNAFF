# PyNAFF

Authors:

* Foteini Asvesta (fasvesta .at. cern .dot. ch)
* Nikos Karastathis (nkarast .at. cern .dot. ch)
* Panagiotis Zisopoulos (pzisopou .at. cern .dot. ch)

A Python implementation of J. Laskar's Numerical Analysis of Fundamental
Frequencies (NAFF) method, for one or more signals, e.g. turn-by-turn data
from many beam position monitors.

Features:

* Frequencies and complex amplitudes of the strongest lines of a signal.
* Several BPM signals at once, as the columns of a two-dimensional array.
* Windowed analysis (`window`) and two frequency optimizers
  (`"quadratic"`, `"brent"`).
* Decohering signals, whose amplitude or frequency changes within the record:
  extra cleaning terms (`close_terms="add"`) and envelope normalization
  (`normalize_envelope`), e.g. for weak synchrotron sidebands next to a
  decohering betatron line.
* Only NumPy is required.

New in 1.3.0: `optimizer="brent"`, `close_terms="add"`, `normalize_envelope`,
and automatic releases from version tags. See `CHANGELOG.md`.

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

## Decohering signals

NAFF assumes a sum of lines with constant amplitudes. A line whose amplitude
or frequency changes within the record, for example the betatron line of a
kicked beam that decoheres through amplitude detuning, leaves a residual within
one FFT bin of it after subtraction. By default PyNAFF stops there, and with
`tol >= 1` it merges the residual into the main line. In both cases the
residual's leakage biases weaker lines nearby, such as synchrotron sidebands.

Two options address this:

```python
# Absorb the residual of a non-stationary line with additional cleaning terms.
# They are reported after the regular terms and are not physical frequencies.
result = pnf.naff(signal, turns=500, nterms=30, close_terms="add")

# Or divide out the slow decay first. With period = one synchrotron period,
# the modulation at Q_s (which carries the sidebands) is kept.
normalized = pnf.normalize_envelope(signal, period=62, band=(0.15, 0.25))
result = pnf.naff(normalized, turns=500, nterms=30)
```

`examples/decohering_sidebands.py` compares the options on a signal with a
known answer (sidebands at 1 % of a decohering main line). For records of six
synchrotron periods or more, `close_terms="add"` recovers the sideband ratio to
within 0.5 %; `normalize_envelope` is the better choice for short records.

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
