import numpy as np

def average_curves(curves: dict):
    '''Simple average of curves. Assumes that curves are element-wise compatible.'''
    all_T = np.array([v["T"] for v in curves.values()])
    all_dsc = np.array([v["dsc"] for v in curves.values()])

    avg_T = np.mean(all_T, axis=0)
    avg_dsc = np.mean(all_dsc, axis=0)
    return {"T": avg_T, "dsc": avg_dsc}


def subtract_background(curves: dict, background: dict) -> dict:
    """
    Subtract a background DSC signal from a set of curves.

    Parameters
    ----------
    curves : dict
        Nested dictionary of curves, e.g.
        {
            1: {"T": array([...]), "dsc": array([...])},
            2: {"T": array([...]), "dsc": array([...])},
            ...
        }
    background : dict
        Background curve with keys "T" and "dsc".

    Returns
    -------
    dict
        New dictionary with the same structure as `curves`,
        where each 'dsc' is background-subtracted.
    """
    Tb = background["T"]
    dsc_b = background["dsc"]

    result = {}
    for seg, data in curves.items():
        T = np.asarray(data["T"])
        dsc = np.asarray(data["dsc"])
        # Interpolate background to measurement T grid
        dsc_b_interp = np.interp(T, Tb, dsc_b)
        result[seg] = {
            "T"  : T,
            "dsc": dsc - dsc_b_interp,
        }
    return result



def slice_curve_idx(curve: dict, i_start=None, i_end=None) -> dict:
    """
    Slice a single DSC curve dictionary by index range.

    Parameters
    ----------
    curve : dict
        A dictionary with at least "T" and "dsc" keys.
        e.g. {"T": array([...]), "dsc": array([...]), ...}
    i_start : int, optional
        Start index (inclusive). Defaults to the beginning.
    i_end : int, optional
        End index (exclusive). Defaults to the end.

    Returns
    -------
    dict
        New dictionary containing only the specified index range.
        Other metadata (if any) is preserved.
    """
    T = np.asarray(curve["T"])
    n = len(T)

    # Normalize indices
    if i_start is None:
        i_start = 0
    if i_end is None or i_end > n:
        i_end = n

    # Slice all array-like fields that match T’s shape
    sliced = {}
    for key, value in curve.items():
        if isinstance(value, np.ndarray) and value.shape == T.shape:
            sliced[key] = value[i_start:i_end]
        else:
            sliced[key] = value

    return sliced