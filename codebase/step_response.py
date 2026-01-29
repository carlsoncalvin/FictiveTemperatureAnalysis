import numpy as np
import warnings
from scipy.optimize import curve_fit

from data_parser import parse_mettler_dsc_txt_file

def compute_step_response_cp(data_path, window, static_temp_corr=0):
    data = parse_mettler_dsc_txt_file(data_path)

    # only 1 segment, unnest dict
    data = data[list(data.keys())[0]]["data"]
    data["Ts"] = static_temp_corr + data["Ts"]

    n = len(data["t"])

    T = np.array([], dtype=float)
    cp_data = np.empty(shape=(int(n / window), window), dtype=complex)

    for start in range(0, n, window):

        end = start + window

        # compute heat flow FFT
        hf_fft = np.fft.fft(data["Value"][start:end])

        # get temperature
        T_array = data["Ts"][start:end]

        # get isothermal temp ignoring dynamic portion
        # arbitrary index of 500 chosen to take only tail end of temps
        T_avg = np.mean(T_array[500:])
        T = np.append(T, T_avg)

        # get instantaneous heat rate and compute FFT
        t_array = data["t"][start:end]
        q = np.gradient(T_array, t_array)
        q_fft = np.fft.fft(q)

        # calculate complex cp
        # sometimes there is a divide by zero warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)

            cp = hf_fft / q_fft
        if w:
            print(f"{w[0].category.__name__}: {w[0].message} at block {start // window}")

        cp_data[int(start / window)] = cp

    return cp_data, T


def boxcar(x, box_length):
    """
    Boxcar average using cumulative sums.
    Output is edge padded with first/last interior value.
    """
    x = np.asarray(x)
    box_length = int(box_length)
    if box_length < 1:
        raise ValueError("box length must be >= 1")
    if box_length > len(x):
        raise ValueError("box length must be <= len(x)")

    c = np.cumsum(np.insert(x, 0, 0))
    interior = (c[box_length:] - c[:-box_length]) / box_length  # length = N - box length + 1

    N = len(x)
    M = len(interior)
    left = box_length // 2
    right = N - (left + M)
    left_pad = np.full(left, interior[0])
    right_pad = np.full(right, interior[-1])
    return np.concatenate((left_pad, interior, right_pad))


def gaussian(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b


def fit_curve(T, cp_data, col_idx, p0=None):
    """
    Fit a Gaussian to the data in the form:

    $$a \exp\left(-\frac{(x - x_0)^2}{2 \sigma^2}\right) + b$$

    Returns x0 of peak. If p0 is provided, use it as the initial guess for the fit, otherwise use
    parameters:
        a     = max(y)
        x0    = center of data (calculated as x[len(x) // 2])
        sigma = 10
        b     = 0
    """
    y = -np.gradient(boxcar(np.abs(cp_data[:, col_idx]), 3))
    x = T

    if not p0:
        p0 = (max(y), x[len(x) // 2], 10, 0)

    params, _ = curve_fit(gaussian, x, y, p0=p0)
    a, x0, sigma, b = params
    return x0

def fit_all_curves(T, cp_data, window, p0=None):
    """
    Fits multiple curves to the given step response data and calculates the maximum number of
    frequencies that can be determined within the specified window. This function uses a
    step-by-step process to fit curves incrementally and stops once the calculation fails or the
    window limit is reached.

    Fits a Gaussian to the data in the form:

    $$a \\exp\\left(-\\frac{(x - x_0)^2}{2 \\sigma^2}\\right) + b$$

    If p0 is not provided, default parameters are calculated from each curve as follows:
        a     = max(y)
        x0    = center of data (calculated as len(x) // 2)
        sigma = 10
        b     = 0

    Parameters
    ----------
    T : array-like
        Temperature data or independent variable array used for curve fitting.
    cp_data : array-like
        Heat capacity data or dependent variable array corresponding to the temperatures.
    window : int
        The number of data points in each step.
    p0 : array-like, optional
        Initial guess for the fitting parameters. If not provided, the default
        initialization is used.

    Returns
    -------
    inflections: array-like
        An array containing the results of the curve fitting for all successfully
        determined frequencies. Each result corresponds to a successful curve fit.
    stop: int
        integer marking the last row of the data used for curve fitting.
    """

    # find max number of frequencies that can be calculated
    stop = 1
    inflections = np.array([])
    while stop < window:
        try:
            res = fit_curve(T, cp_data, stop, p0=p0)
            inflections = np.append(inflections, res)
            stop += 1
        except RuntimeError as e:
            print(f"Error:\n  {e}\nencountered after {stop} iterations")
            break

    return inflections, stop