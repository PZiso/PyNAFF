"""Numerical Analysis of Fundamental Frequencies (NAFF)."""

from math import lgamma, log
import warnings as warnings_module

import numpy as np

if __package__:
    from ._version import __version__
else:
    from _version import __version__

__PyVersion = [3]
__authors__ = ["F. Asvesta", "N. Karastathis", "P. Zisopoulos"]
__contact__ = ["nkarast .at. cern .dot. ch"]
__all__ = ["naff"]


def _hardy_weights(turns):
    """Return the composite Hardy quadrature weights used by NAFF."""
    weights = np.empty(turns + 1, dtype=np.float64)
    weights[0] = 41.0
    weights[-1] = 41.0
    pattern = np.array([216.0, 27.0, 272.0, 27.0, 216.0, 82.0])
    weights[1:-1] = np.resize(pattern, turns - 1)
    weights *= 6.0 / (840.0 * turns)
    return weights


def _prepare_context(turns, window):
    samples = np.arange(turns + 1, dtype=np.float64)
    centered_time = 2.0 * np.pi * samples - np.pi * turns
    scale = np.exp(
        window * log(2.0)
        + 2.0 * lgamma(window + 1)
        - lgamma(2 * window + 1)
    )
    taper = scale * (1.0 + np.cos(centered_time / turns)) ** window
    return samples, taper, _hardy_weights(turns)


def _integral(signal, frequency, samples, integration_weights):
    phase = np.exp(-2.0j * np.pi * frequency * samples)
    value = np.dot(signal * integration_weights, phase)
    return abs(value), value.real, value.imag


def _refine_frequency_quadratic(
    signal,
    frequency,
    step,
    tolerance,
    samples,
    integration_weights,
):
    """Maximize the norm of the NAFF scalar product near an FFT bin."""
    epsilon = 1.0e-15
    x2 = frequency
    y2, a2, b2 = _integral(signal, x2, samples, integration_weights)
    x1 = x2 - step
    x3 = x2 + step
    y1, a1, b1 = _integral(signal, x1, samples, integration_weights)
    y3, a3, b3 = _integral(signal, x3, samples, integration_weights)

    for _ in range(1000):
        if step < tolerance or abs(y3 - y1) < epsilon:
            break

        if y1 < y2 and y3 < y2:
            slope2 = (y1 - y2) / (x1 - x2)
            slope3 = (y1 - y3) / (x1 - x3)
            parabola = (slope2 - slope3) / (x2 - x3)
            if parabola == 0.0:
                break
            linear = slope2 - parabola * (x1 + x2)
            candidate = -linear / (2.0 * parabola)
            new_step = abs(candidate - x2)

            if candidate > x2:
                x1, y1, a1, b1 = x2, y2, a2, b2
                x2 = candidate
                y2, a2, b2 = _integral(
                    signal, x2, samples, integration_weights
                )
                x3 = x2 + new_step
                y3, a3, b3 = _integral(
                    signal, x3, samples, integration_weights
                )
            else:
                x3, y3, a3, b3 = x2, y2, a2, b2
                x2 = candidate
                y2, a2, b2 = _integral(
                    signal, x2, samples, integration_weights
                )
                x1 = x2 - new_step
                y1, a1, b1 = _integral(
                    signal, x1, samples, integration_weights
                )
            step = new_step
        else:
            if y1 > y3:
                x2, y2, a2, b2 = x1, y1, a1, b1
            else:
                x2, y2, a2, b2 = x3, y3, a3, b3

            x1 = x2 - step
            x3 = x2 + step
            y1, a1, b1 = _integral(
                signal, x1, samples, integration_weights
            )
            y3, a3, b3 = _integral(
                signal, x3, samples, integration_weights
            )

    return x2, y2, a2, b2


def _refine_frequency_brent(
    signal,
    lower,
    upper,
    tolerance,
    samples,
    integration_weights,
):
    """Maximize the NAFF scalar product with bounded Brent optimization."""
    golden_ratio = 0.5 * (3.0 - np.sqrt(5.0))
    sqrt_epsilon = np.sqrt(np.finfo(np.float64).eps)
    x = lower + golden_ratio * (upper - lower)
    w = x
    v = x
    objective, _, _ = _integral(signal, x, samples, integration_weights)
    fx = -objective
    fw = fx
    fv = fx
    step = 0.0
    previous_step = 0.0

    for _ in range(500):
        midpoint = 0.5 * (lower + upper)
        tolerance1 = sqrt_epsilon * abs(x) + tolerance / 3.0
        tolerance2 = 2.0 * tolerance1
        if abs(x - midpoint) <= tolerance2 - 0.5 * (upper - lower):
            break

        if abs(previous_step) > tolerance1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q

            saved_step = previous_step
            previous_step = step
            if (
                q != 0.0
                and abs(p) < abs(0.5 * q * saved_step)
                and p > q * (lower - x)
                and p < q * (upper - x)
            ):
                step = p / q
                candidate = x + step
                if (
                    candidate - lower < tolerance2
                    or upper - candidate < tolerance2
                ):
                    step = (
                        tolerance1 if x < midpoint else -tolerance1
                    )
            else:
                previous_step = (
                    upper - x if x < midpoint else lower - x
                )
                step = golden_ratio * previous_step
        else:
            previous_step = upper - x if x < midpoint else lower - x
            step = golden_ratio * previous_step

        if abs(step) >= tolerance1:
            candidate = x + step
        else:
            candidate = x + (
                tolerance1 if step > 0.0 else -tolerance1
            )

        candidate_objective, _, _ = _integral(
            signal, candidate, samples, integration_weights
        )
        candidate_value = -candidate_objective
        if candidate_value <= fx:
            if candidate < x:
                upper = x
            else:
                lower = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = candidate, candidate_value
        else:
            if candidate < x:
                lower = candidate
            else:
                upper = candidate
            if candidate_value <= fw or w == x:
                v, fv = w, fw
                w, fw = candidate, candidate_value
            elif candidate_value <= fv or v == x or v == w:
                v, fv = candidate, candidate_value

    value, real, imaginary = _integral(
        signal, x, samples, integration_weights
    )
    return x, value, real, imaginary


def _refine_frequency(
    signal,
    frequency,
    resolution,
    tolerance,
    samples,
    integration_weights,
    optimizer,
    get_full_spectrum,
):
    if optimizer == "quadratic":
        return _refine_frequency_quadratic(
            signal,
            frequency,
            resolution / 3.0,
            tolerance,
            samples,
            integration_weights,
        )

    lower = frequency - resolution
    upper = frequency + resolution
    if get_full_spectrum:
        lower = max(lower, -0.5)
        upper = min(upper, 0.5)
    else:
        lower = max(lower, 0.0)
        upper = min(upper, 0.5)
    return _refine_frequency_brent(
        signal,
        lower,
        upper,
        tolerance,
        samples,
        integration_weights,
    )


def _frequency_status(
    frequency,
    frequencies,
    fundamental_resolution,
    tolerance,
):
    if not frequencies:
        return 1, 0

    distances = np.abs(np.asarray(frequencies) - frequency)
    nearby = np.flatnonzero(distances < abs(fundamental_resolution))
    if nearby.size == 0:
        return 1, 0

    duplicate = nearby[
        distances[nearby] / abs(fundamental_resolution) < tolerance
    ]
    if duplicate.size:
        return -1, int(duplicate[0])
    return 0, 0


def _basis_overlap(
    frequency,
    old_frequency,
    samples,
    integration_weights,
):
    phase = np.exp(
        -2.0j * np.pi * (frequency - old_frequency) * samples
    )
    return np.dot(integration_weights, phase)


def _remove_component(
    residual,
    contribution,
    frequency,
    samples,
    real_spectrum,
):
    component = contribution * np.exp(
        2.0j * np.pi * frequency * samples
    )
    is_unpaired_bin = np.isclose(frequency, 0.0) or np.isclose(
        abs(frequency), 0.5
    )
    if real_spectrum and not is_unpaired_bin:
        residual -= 2.0 * component.real
    elif real_spectrum:
        residual -= component.real
    else:
        residual -= component


def _add_frequency(
    residual,
    frequency,
    integral,
    frequencies,
    amplitudes,
    coefficients,
    samples,
    integration_weights,
    real_spectrum,
):
    """Orthonormalize a new exponential and subtract its projection."""
    old_count = len(frequencies)
    overlaps = np.ones(old_count + 1, dtype=np.complex128)
    for index, old_frequency in enumerate(frequencies):
        overlaps[index] = _basis_overlap(
            frequency,
            old_frequency,
            samples,
            integration_weights,
        )

    row = np.zeros(old_count + 1, dtype=np.complex128)
    for k in range(old_count):
        for i in range(old_count):
            row[k] -= (
                np.dot(
                    np.conj(coefficients[i, : i + 1]),
                    overlaps[: i + 1],
                )
                * coefficients[i, k]
            )
    row[old_count] = 1.0

    divisor = np.sqrt(abs(np.dot(np.conj(row), overlaps)))
    if divisor == 0.0:
        return False
    row /= divisor

    frequencies.append(frequency)
    coefficients[old_count, : old_count + 1] = row
    multiplier = integral / divisor

    for index, old_frequency in enumerate(frequencies):
        contribution = row[index] * multiplier
        amplitudes[index] += contribution
        _remove_component(
            residual,
            contribution,
            old_frequency,
            samples,
            real_spectrum,
        )
    return True


def _naff_1d(
    data,
    turns,
    nterms,
    skip_turns,
    get_full_spectrum,
    samples,
    taper,
    quadrature_weights,
    tolerance,
    show_warnings,
    optimizer,
):
    residual = np.asarray(
        data[skip_turns : skip_turns + turns + 1],
        dtype=np.complex128,
    ).copy()
    integration_weights = taper * quadrature_weights
    frequencies = []
    amplitudes = np.zeros(nterms, dtype=np.complex128)
    coefficients = np.zeros((nterms, nterms), dtype=np.complex128)
    resolution = 1.0 / turns
    real_spectrum = not get_full_spectrum

    dc_warned = False
    for _ in range(nterms):
        fft_input = residual[:-1] * taper[:-1]
        if get_full_spectrum:
            spectrum = np.fft.fft(fft_input)
            frequency = np.fft.fftfreq(turns)[
                int(np.argmax(np.abs(spectrum)))
            ]
        else:
            spectrum = np.fft.rfft(fft_input.real)
            frequency = np.fft.rfftfreq(turns)[
                int(np.argmax(np.abs(spectrum)))
            ]

        if frequency == 0.0 and show_warnings and not dc_warned:
            warnings_module.warn(
                "PyNAFF found a DC component; subtract the signal mean "
                "before analysis if DC is not of interest.",
                UserWarning,
                stacklevel=3,
            )
            dc_warned = True

        frequency, _, real, imaginary = _refine_frequency(
            residual,
            frequency,
            resolution,
            resolution / 1.0e8,
            samples,
            integration_weights,
            optimizer,
            get_full_spectrum,
        )
        status, existing_index = _frequency_status(
            frequency, frequencies, resolution, tolerance
        )

        if status == 0:
            if show_warnings:
                warnings_module.warn(
                    "PyNAFF stopped because a residual peak is within one "
                    "FFT bin of an extracted frequency but outside tol. "
                    "A larger tol may continue extraction, but can also "
                    "accept leakage artifacts.",
                    UserWarning,
                    stacklevel=3,
                )
            break
        if status == -1:
            contribution = complex(real, imaginary)
            amplitudes[existing_index] += contribution
            _remove_component(
                residual,
                contribution,
                frequency,
                samples,
                real_spectrum,
            )
            continue

        if not _add_frequency(
            residual,
            frequency,
            complex(real, imaginary),
            frequencies,
            amplitudes,
            coefficients,
            samples,
            integration_weights,
            real_spectrum,
        ):
            break

    result = np.empty((len(frequencies), 5), dtype=np.float64)
    for order, frequency in enumerate(frequencies):
        amplitude = amplitudes[order]
        result[order] = (
            order,
            frequency,
            abs(amplitude),
            amplitude.real,
            amplitude.imag,
        )
    return result


def naff(
    data,
    turns=300,
    nterms=1,
    skipTurns=0,
    getFullSpectrum=False,
    window=1,
    tol=1.0e-4,
    warnings=True,
    optimizer="quadratic",
):
    """Extract the fundamental frequencies of one or more BPM signals.

    A single signal has shape ``(observations,)``. Multiple BPM signals
    have shape ``(observations, bpms)``.

    The scalar result has shape ``(found_terms, 5)``. The multi-BPM result
    has shape ``(bpms, nterms, 5)``, with unused rows filled with ``NaN``.
    Each row contains ``[order, frequency, amplitude, real amplitude,
    imaginary amplitude]``.

    ``tol`` controls when a refined frequency is considered a duplicate of
    one already found. Set ``warnings=False`` to suppress the DC warning.

    ``optimizer="quadratic"`` preserves the fast legacy three-point
    interpolation. ``optimizer="brent"`` uses a bounded parabolic search with
    golden-section fallback, making it more conservative near non-ideal peaks.
    """
    values = np.asarray(data)
    if values.ndim not in (1, 2):
        raise ValueError(
            "data must have shape (observations,) or (observations, bpms)"
        )
    if values.ndim == 2 and values.shape[1] == 0:
        raise ValueError("data must contain at least one BPM")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("data must contain numeric values")
    if (
        isinstance(turns, (bool, np.bool_))
        or not isinstance(turns, (int, np.integer))
        or turns < 6
    ):
        raise ValueError("turns must be an integer of at least 6")
    if (
        isinstance(nterms, (bool, np.bool_))
        or not isinstance(nterms, (int, np.integer))
        or nterms < 1
    ):
        raise ValueError("nterms must be a positive integer")
    if (
        isinstance(skipTurns, (bool, np.bool_))
        or not isinstance(skipTurns, (int, np.integer))
        or skipTurns < 0
    ):
        raise ValueError("skipTurns must be a non-negative integer")
    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window < 0
    ):
        raise ValueError("window must be a non-negative integer")
    if (
        isinstance(tol, (bool, np.bool_))
        or
        not isinstance(tol, (int, float, np.integer, np.floating))
        or not np.isfinite(tol)
        or tol <= 0
        or tol > 1
    ):
        raise ValueError("tol must be a finite number in the interval (0, 1]")
    if not isinstance(warnings, (bool, np.bool_)):
        raise ValueError("warnings must be a boolean")
    if not isinstance(getFullSpectrum, (bool, np.bool_)):
        raise ValueError("getFullSpectrum must be a boolean")
    if not isinstance(optimizer, str) or optimizer not in (
        "quadratic",
        "brent",
    ):
        raise ValueError("optimizer must be 'quadratic' or 'brent'")
    if np.iscomplexobj(values) and not getFullSpectrum:
        raise ValueError("getFullSpectrum must be True for complex input")

    turns -= turns % 6
    required_observations = skipTurns + turns + 1
    if values.shape[0] < required_observations:
        raise ValueError(
            "data must contain at least skipTurns + turns + 1 observations"
        )
    analysis_values = values[:required_observations]
    if not np.all(np.isfinite(analysis_values)):
        raise ValueError("data must contain only finite values")

    samples, taper, quadrature_weights = _prepare_context(turns, window)
    if values.ndim == 1:
        return _naff_1d(
            values,
            turns,
            nterms,
            skipTurns,
            getFullSpectrum,
            samples,
            taper,
            quadrature_weights,
            float(tol),
            bool(warnings),
            optimizer,
        )

    result = np.full((values.shape[1], nterms, 5), np.nan)
    for bpm in range(values.shape[1]):
        bpm_result = _naff_1d(
            values[:, bpm],
            turns,
            nterms,
            skipTurns,
            getFullSpectrum,
            samples,
            taper,
            quadrature_weights,
            float(tol),
            bool(warnings),
            optimizer,
        )
        result[bpm, : len(bpm_result)] = bpm_result
    return result
