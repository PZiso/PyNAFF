# Changelog

## 1.3.0

- Add `optimizer="quadratic"` and `optimizer="brent"` frequency-refinement
  options while preserving quadratic refinement as the default.
- Add a dependency-free bounded Brent implementation with golden-section
  fallback.
- Add `close_terms="add"`: a residual peak within one FFT bin of an extracted
  frequency is subtracted and reported as an additional term instead of
  stopping the extraction. This keeps the residual of a decohering line out of
  weak nearby lines such as synchrotron sidebands. The default `"stop"`
  preserves the previous behavior.
- Add `normalize_envelope(data, period, band=None)`, which divides a signal by
  its envelope averaged over `period` samples, to remove slow amplitude decay
  before the analysis.
- Add known-answer tests and `examples/decohering_sidebands.py` for weak
  sidebands next to a decohering betatron line.

## 1.2.0

- Accept multiple BPM signals as columns in an `(observations, bpms)` array.
- Return multi-BPM results with shape `(bpms, nterms, 5)`.
- Preserve the existing one-dimensional input and output API.
- Vectorize Hardy quadrature and reuse shared analysis arrays.
- Correct real-signal deflation so subsequent harmonics can be extracted.
- Reject non-numeric and non-finite input before frequency refinement.
- Preserve the `tol` and `warnings` options from PyNAFF 1.1.7.
