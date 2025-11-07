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