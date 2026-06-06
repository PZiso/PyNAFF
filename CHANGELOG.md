# Changelog

## 1.2.0

- Accept multiple BPM signals as columns in an `(observations, bpms)` array.
- Return multi-BPM results with shape `(bpms, nterms, 5)`.
- Preserve the existing one-dimensional input and output API.
- Vectorize Hardy quadrature and reuse shared analysis arrays.
- Correct real-signal deflation so subsequent harmonics can be extracted.
- Reject non-numeric and non-finite input before frequency refinement.
- Preserve the `tol` and `warnings` options from PyNAFF 1.1.7.
