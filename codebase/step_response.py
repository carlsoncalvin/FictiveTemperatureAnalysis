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
        raise ValueError("window must be >= 1")
    if box_length > len(x):
        raise ValueError("window must be <= len(x)")

    c = np.cumsum(np.insert(x, 0, 0))
    interior = (c[box_length:] - c[:-box_length]) / box_length  # length = N - window + 1

    N = len(x)
    M = len(interior)
    left = box_length // 2
    right = N - (left + M)
    left_pad = np.full(left, interior[0])
    right_pad = np.full(right, interior[-1])
    return np.concatenate((left_pad, interior, right_pad))


def gaussian(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b


def fit_curves(T, cp_data, stop, p0=None):
    inflections = np.empty(stop - 1)
    for i in range(1, stop):
        y = -np.gradient(boxcar(np.abs(cp_data[:, i]), 3))
        x = T

        if not p0:
            p0 = (max(y), x[len(x) // 2], 10, 0)

        params, _ = curve_fit(gaussian, x, y, p0=p0)
        a, x0, sigma, b = params

        inflections[i - 1] = x0

    return inflections

# def step_response_wrapper(path, window, sampling_freq, freq_cutoff, temp_corr=0)