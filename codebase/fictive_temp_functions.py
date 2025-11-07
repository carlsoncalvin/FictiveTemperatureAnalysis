from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict

@dataclass(frozen=True)
class Baselines:
    """
    Linear baseline parameters for a DSC curve.

    Stores the slope and intercept of the fitted glass and liquid baselines,
    defined over temperature ranges below and above the glass transition.

    Attributes
    ----------
    glass : Tuple[float, float]
        (slope, intercept) of the baseline fitted in the glass region [T1a, T1b].
        Represents the heat-flow trend before the glass transition.
    liquid : Tuple[float, float]
        (slope, intercept) of the baseline fitted in the liquid region [T2a, T2b].
        Represents the heat-flow trend after the glass transition.

    Notes
    -----
    These parameters are typically obtained using a linear least-squares fit
    and can be evaluated at any temperature using `_evaluate_line()`.
    They are used in both baseline correction and fictive temperature analysis
    to represent the ideal heat capacity behavior outside the transition region.
    """
    glass: Tuple[float, float]   # (slope, intercept)
    liquid: Tuple[float, float]

def _evaluate_line(params: Tuple[float, float], x_values: np.ndarray) -> np.ndarray:
    """Evaluate a line y = m*x + b at x_values."""
    m, b = params
    return m*x_values+b

def _fit_range(x_data: np.ndarray,
               y_data: np.ndarray,
               lim1: float,
               lim2: float):
    """Least-squares line fit y = m*x + b over range lim1 ≤ x ≤ lim2."""
    # boolean index desired data
    range_mask = (x_data >= lim1) & (x_data <= lim2)
    if not np.any(range_mask):
        raise ValueError("No points in the requested window.")
    x_range = x_data[range_mask]
    y_range = y_data[range_mask]

    # perform fit
    m, b = np.polyfit(x_range, y_range, 1)

    return m, b

def _compute_baselines(curve: Dict[str, np.ndarray],
                       T1: Tuple[float, float],
                       T2: Tuple[float, float]) -> Baselines:
    """Compute glass and liquid baselines for a DSC curve. Returns Baselines object."""
    T = np.asarray(curve["T"])
    y = np.asarray(curve["dsc"])
    glass = _fit_range(T, y, *T1)
    liquid = _fit_range(T, y, *T2)
    return Baselines(glass=glass, liquid=liquid)

def _bridge_line(T: np.ndarray,
                 T1b: float,
                 T2a: float,
                 y_at_T1b: float,
                 y_at_T2a: float) -> np.ndarray:
    """
    Construct a straight line across [T1b, T2a] that connects the two endpoint values.
    For T outside [T1b, T2a] the caller will ignore this function's output.
    """
    # y = m*T + b through (T1b, y_at_T1b) and (T2a, y_at_T2a)
    m = (y_at_T2a - y_at_T1b) / (T2a - T1b)
    b = (T1b * y_at_T2a - T2a * y_at_T1b) / (T1b - T2a)
    return m * T + b

def baseline_correction(sample: Dict[str, np.ndarray],
                        reference: Dict[str, np.ndarray],
                        T1: Tuple[float, float],
                        T2: Tuple[float, float],
                        *,
                        return_components: bool = False):
    """
    Piecewise linear correction that aligns the sample's glass/liquid baselines to the reference.

    Parameters
    ----------
    sample, reference : dict with keys {"T", "dsc"}
        Arrays can have different T grids.
    T1 : (T1a, T1b)
        Glass-region fit window.
    T2 : (T2a, T2b)
        Liquid-region fit window.
    return_components : bool
        If True, also return the piecewise correction and the two difference lines.

    Returns
    -------
    y_corrected : np.ndarray
        Sample y after subtracting the composite correction.
    (optional) components : dict
        Dict with keys {"corr", "diff_glass", "diff_liquid"} on the sample T grid.
    """

    T = np.asarray(sample["T"])
    y = np.asarray(sample["dsc"])

    # unpack temperature bounds
    (T1a, T1b), (T2a, T2b) = T1, T2

    # fit baselines for sample and reference
    b_sample = _compute_baselines(sample, T1, T2)
    b_ref = _compute_baselines(reference, T1, T2)

    # find difference lines (sample - reference) for glass and liquid
    diff_glass = _evaluate_line((b_sample.glass[0]  - b_ref.glass[0],
                                 b_sample.glass[1]  - b_ref.glass[1]), T)
    diff_liquid = _evaluate_line((b_sample.liquid[0] - b_ref.liquid[0],
                                  b_sample.liquid[1] - b_ref.liquid[1]), T)

    # construct piecewise correction: glass, bridge, liquid
    correction = np.empty_like(T, dtype=float)

    mask_glass = T <  T1b
    mask_mid   = (T >= T1b) & (T <= T2a)
    mask_liq   = T >  T2a

    # glass segment
    correction[mask_glass] = diff_glass[mask_glass]

    # middle bridge segment
    y_T1b = (b_sample.glass[0]  - b_ref.glass[0])  * T1b + (b_sample.glass[1]  - b_ref.glass[1])
    y_T2a = (b_sample.liquid[0] - b_ref.liquid[0]) * T2a + (b_sample.liquid[1] - b_ref.liquid[1])
    correction[mask_mid] = _bridge_line(T[mask_mid], T1b, T2a, y_T1b, y_T2a)

    # liquid segment
    correction[mask_liq] = diff_liquid[mask_liq]

    # apply correction
    y_corrected = y - correction

    if return_components:
        return y_corrected, {
            "correction": correction,
            "diff_glass": diff_glass,
            "diff_liquid": diff_liquid,
        }

    return y_corrected

def fictive_temperature(sample: Dict[str, np.ndarray],
                        T1: Tuple[float, float],
                        T2: Tuple[float, float]):
    """
    Compute fictive temperature by the equal-area method.

    Parameters
    ----------
    sample : dict with keys {"T", "dsc"}
        Temperature (T) and signal (dsc) arrays for one DSC curve.
    T1 : (float, float)
        Glass-region fit window (T1a, T1b).
    T2 : (float, float)
        Liquid-region fit window (T2a, T2b).

    Returns
    -------
    result : dict
        {
          "Tf" : float,                  # fictive temperature
          "T_slice" : np.ndarray,        # T values within [T1a, T2b]
          "area_liq_minus_glass" : np.ndarray,  # tail integrals of (y_liq - y_glass)
          "area_exp_minus_glass" : np.ndarray,  # tail integrals of (y_exp - y_glass)
          "glass_line" : np.ndarray,     # baseline fit for glass region
          "liquid_line" : np.ndarray,    # baseline fit for liquid region
          "y_slice" : np.ndarray,        # experimental data over [T1a, T2b]
          "idx_min" : int                # index of equal-area crossing
        }
    """
    # unpack inputs
    T = np.asarray(sample["T"])
    y = np.asarray(sample["dsc"])
    T1a, T1b = T1
    T2a, T2b = T2

    # get glass and liquid line params
    baselines = _compute_baselines(sample, T1, T2)

    # slice data to [T1a, T2b]
    mask = (T >= T1a) & (T <= T2b)
    T_slice = T[mask]
    y_slice = y[mask]

    # compute lines over the slice
    y_glass = _evaluate_line(baselines.glass,  T_slice)
    y_liquid = _evaluate_line(baselines.liquid, T_slice)

    # build difference arrays
    liq_minus_glass = y_liquid - y_glass
    exp_minus_glass = y_slice - y_glass

    # compute tail integrals
    dT = np.diff(T_slice)
    seg_liq = 0.5 * (liq_minus_glass[:-1] + liq_minus_glass[1:]) * dT
    seg_exp = 0.5 * (exp_minus_glass[:-1] + exp_minus_glass[1:]) * dT

    # cumulative sum from the end
    A = np.zeros_like(T_slice)
    B = np.zeros_like(T_slice)
    if len(seg_liq): # check that seg_liq is valid
        A[:-1] = np.cumsum(seg_liq[::-1])[::-1]
        B[:-1] = np.cumsum(seg_exp[::-1])[::-1]
    # A[-1] and B[-1] stay 0

    # find fictive temperature as where A - B is first 0
    #diff = A - B
    # different calculation:  should include low T divergence from glass line
    diff = A - max(B)
    sign_change = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_change):
        idx_min = sign_change[0]  # first crossing
    else:
        idx_min = np.argmin(np.abs(diff))  # fallback
    Tf = T_slice[idx_min]

    return {
        "Tf": Tf,
        "T_slice": T_slice,
        "area_liq_minus_glass": A,
        "area_exp_minus_glass": B,
        "glass_line": y_glass,
        "liquid_line": y_liquid,
        "y_slice": y_slice,
        "idx_min": idx_min,
    }

