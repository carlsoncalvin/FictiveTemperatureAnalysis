import numpy as np
import matplotlib.pyplot as plt
import pickle

from data_parser import parse_mettler_dsc_txt_file, extract_T_and_dsc
from fictive_temp_functions import baseline_correction, fictive_temperature
from utils import average_curves, subtract_background, slice_curve_idx


# ---------- Small helpers ----------

def _load_curves(data_path, bg_path=None):
    """Load sample data and optional background, returning (data, background)."""
    # main data
    results = parse_mettler_dsc_txt_file(data_path)
    data = extract_T_and_dsc(results)

    background = None
    if bg_path is not None:
        bg_results = parse_mettler_dsc_txt_file(bg_path)
        bg_data = extract_T_and_dsc(bg_results)
        background = average_curves(bg_data)

    return data, background


def _apply_temp_polynomial(data, background, temp_polynomial):
    """Apply T_corrected - T_exp = a + b*T_exp to all curves and background."""
    if temp_polynomial is None:
        return data, background

    a, b = temp_polynomial

    for curve in data.values():
        T = curve["T"]
        curve["T"] = T + a + b * T

    if background is not None:
        T_bg = background["T"]
        background["T"] = T_bg + a + b * T_bg

    return data, background


def _slice_curves(data, cut_idx):
    """Slice all curves from index cut_idx onwards."""
    if cut_idx is None:
        return data
    for k in list(data.keys()):
        data[k] = slice_curve_idx(data[k], i_start=cut_idx)
    return data


def _subtract_background_with_optional_plots(data, background, quiet):
    """Subtract background; optionally show before/after plots."""
    if background is None:
        return data

    if quiet:
        return subtract_background(data, background)

    # Show raw vs bg-subtracted
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for curve in data.values():
        ax1.plot(curve["T"], curve["dsc"])
    ax1.set_title("Raw data")

    data = subtract_background(data, background)

    for curve in data.values():
        ax2.plot(curve["T"], curve["dsc"])
    ax2.set_title("Background subtracted")

    for ax in (ax1, ax2):
        ax.set_xlabel("T / °C")
        ax.set_ylabel("Heat Flow / mW")

    plt.tight_layout()
    plt.show()

    return data


def _plot_reference_segments_for_selection(data, ref_segs, quiet):
    """Plot candidate reference segments to help decide which to keep."""
    if quiet:
        return

    plt.figure()
    for seg in ref_segs:
        if seg in data:
            plt.plot(data[seg]["T"], data[seg]["dsc"], label=f"Segment {seg}")
    plt.legend()
    plt.title("Candidate reference segments")
    plt.xlabel("T / °C")
    plt.ylabel("Heat Flow / mW")
    plt.show()


def _choose_reference_segments(ref_segs, segs_to_keep):
    """Decide which reference segments to keep (interactive if segs_to_keep is None)."""
    if segs_to_keep is not None:
        return list(segs_to_keep)

    while True:
        keep_segs = input(f"Keep all reference segments {ref_segs}? (y/n) ")
        if keep_segs.lower() in {"y", "yes"}:
            return list(ref_segs)
        if keep_segs.lower() in {"n", "no"}:
            while True:
                segs_to_keep_str = input("Enter segment numbers to keep separated by spaces: ")
                try:
                    segs = [int(seg) for seg in segs_to_keep_str.split()]
                except ValueError:
                    print("Invalid input. Please enter integer segment numbers separated by spaces.")
                    continue
                if all(seg in ref_segs for seg in segs):
                    return segs
                print(f"Invalid input. Valid segment numbers are {ref_segs}.")
        print("Invalid input. Please enter 'y' or 'n'.")


def _build_reference_curve(data, ref_segs, segs_to_keep, multiref):
    """Return reference curve and remaining sample data. Cuts references to same size before
    averaging."""
    segs_to_keep = _choose_reference_segments(ref_segs, segs_to_keep)

    selected = {k: data[k] for k in segs_to_keep}

    if multiref is not None:
        multiref_data = {k: data[k] for k in multiref}
    else :
        multiref_data = None

    # cut references to same size if differ in length
    # IT IS ASSUMED THAT LENGTH DIFFERS AT THE END
    # THERE IS NO CHECK FOR THIS
    min_len = np.inf
    for seg in selected.values():
        min_len = min(min_len, len(seg["T"]))

    for k in list(selected.keys()):
        selected[k] = slice_curve_idx(selected[k], i_end=min_len)

    ref = average_curves(selected)

    # Remove all reference segments from the data dict
    for seg in ref_segs:
        data.pop(seg, None)

    return ref, data, multiref_data


def _plot_reference_and_sample_for_ranges(ref, sample, plot_range, quiet):
    """Show plots used to visually choose T1, T2."""
    if quiet:
        return

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    ax1, ax2 = axs

    for ax in axs:
        ax.plot(ref["T"], ref["dsc"], label="Reference")
        ax.set_xlabel("T / °C")
        ax.grid(axis="x")

    ax1.plot(sample["T"], sample["dsc"], label="Sample")
    ax1.set_ylabel("Heat Flow / mW")

    ax2.plot(sample["T"], sample["dsc"], label="Sample")
    x1, x2 = plot_range
    ax2.set_xlim(x1, x2)
    ax2.set_xticks(range(x1, x2, 20))

    for ax in axs:
        ax.legend()

    fig.tight_layout()
    plt.show()


def _choose_T_ranges(T1, T2, quiet):
    """Choose T1, T2 for baseline regions (interactive if missing)."""
    if T1 is not None and T2 is not None:
        return T1, T2

    if quiet:
        # fall back to defaults if we can't ask
        return (0, 60), (140, 200)

    while True:
        keep = input("Keep default T1 and T2? (0, 60), (140, 200) (y/n) ")
        if keep.lower() in {"y", "yes"}:
            return (0, 60), (140, 200)

        if keep.lower() in {"n", "no"}:
            while True:
                T_ranges_str = input("Enter T1 and T2 ranges as 4 integers (T1_low T1_high T2_low T2_high): ")
                try:
                    vals = [int(T) for T in T_ranges_str.split()]
                except ValueError:
                    print("Invalid input. Please enter integer temperatures separated by spaces.")
                    continue
                if len(vals) == 4:
                    return (vals[0], vals[1]), (vals[2], vals[3])
                print("Invalid input. Please enter exactly 4 integers separated by spaces.")

        print("Invalid input. Please enter 'y' or 'n'.")


def _baseline_correct_all(data, ref, sample, T1, T2, quiet, multiref_data):
    """Baseline correct sample and all curves; return sample_corrected and updated data."""
    dsc_corrected = baseline_correction(sample, ref, T1, T2)

    sample_corrected = sample.copy()
    sample_corrected["dsc"] = dsc_corrected

    if not quiet:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        ax1, ax2 = axs

        for ax in axs:
            ax.plot(ref["T"], ref["dsc"], label="Reference")
            ax.set_xlabel("T / °C")

        ax1.plot(sample["T"], sample["dsc"], label="Sample")
        ax1.set_title("Sample and Reference Raw")
        ax1.set_ylabel("Heat Flow / mW")

        ax2.plot(sample_corrected["T"], sample_corrected["dsc"], label="Corrected Sample")
        ax2.set_title("Sample and Reference Corrected")

        for ax in axs:
            ax.legend()

        fig.tight_layout()
        plt.show()

    # Correct all curves in-place
    if multiref_data is not None:
        for key in data:
            if int(key)+4 in multiref_data:
                multiref = multiref_data[int(key)+4]
            else:
                min_key = sorted(multiref_data.keys())[0]
                multiref = multiref_data[min_key]
            data[key]["dsc"] = baseline_correction(data[key], multiref, T1, T2)
    else:
        for curve in data.values():
            curve["dsc"] = baseline_correction(curve, ref, T1, T2)

    return sample_corrected, data


def _plot_fictive_example(sample_corrected, sample_raw, T1, T2, quiet):
    """Plot the example fictive temperature construction on one curve."""
    if quiet:
        return None

    Tf_results = fictive_temperature(sample_corrected, T1, T2)

    plt.figure()
    plt.plot(sample_corrected["T"], sample_corrected["dsc"], label="Corrected sample")

    T_slice = Tf_results["T_slice"]
    plt.plot(T_slice, Tf_results["glass_line"], label="Glass line")
    plt.plot(T_slice, Tf_results["liquid_line"], label="Liquid line")

    Tf_val = Tf_results["Tf"]
    plt.vlines(
        Tf_val,
        np.min(sample_raw["dsc"]),
        np.max(sample_raw["dsc"]),
        color="k",
        linestyles="--",
        label=f"T$_f$={Tf_val:.2f} °C",
    )

    plt.xlabel("T / °C")
    plt.ylabel("Heat Flow / mW")
    plt.legend()
    plt.title("Fictive temperature construction")
    plt.show()

    return Tf_results


def _plot_all_corrected(data, quiet):
    if quiet:
        return
    plt.figure()
    for curve in data.values():
        plt.plot(curve["T"], curve["dsc"])
    plt.xlabel("T / °C")
    plt.ylabel("Heat Flow / mW")
    plt.title("All corrected curves")
    plt.show()


def _generate_default_ta(num_curves, t_a_start_exponent):
    """Generate default aging times 1,2,5 × 10^e from e=t_a_start_exponent to 5."""
    all_t_a = np.empty(num_curves, dtype=float)

    i = 0
    for e in range(t_a_start_exponent, 6):  # up to 10^5
        for p in (1, 2, 5):
            if i == num_curves:
                break
            all_t_a[i] = p * 10.0 ** e
            i += 1
        if i == num_curves:
            break

    # special-case tweak
    if num_curves > 0 and all_t_a[-1] == 100000:
        all_t_a[-1] = 90000

    return all_t_a


def _compute_all_Tf(data, T1, T2):
    """Compute fictive temperature for each curve in data, sorted by segment key."""
    keys = sorted(data.keys())
    all_Tf = np.empty(len(keys), dtype=float)
    for i, key in enumerate(keys):
        Tf_results = fictive_temperature(data[key], T1, T2)
        all_Tf[i] = Tf_results["Tf"]
    return all_Tf


def _plot_Tf_vs_ta(all_t_a, all_Tf, quiet):
    if quiet:
        return
    plt.figure()
    plt.scatter(all_t_a, all_Tf)
    plt.xscale("log")
    plt.xlabel("t$_a$ / s")
    plt.ylabel("T$_f$ / °C")
    plt.title("Fictive temperature vs aging time")
    plt.show()


def _compute_Ta_corrected(T_a, temp_polynomial):
    """Return Ta_corrected if a polynomial is given, else just T_a."""
    if temp_polynomial is None:
        return T_a
    a, b = temp_polynomial
    Ta_corrected = T_a + a + b * T_a
    return Ta_corrected


def _maybe_save_summary(summary, data_path, T_a, autosave):
    """Interactive saving logic"""
    data_branch = "/".join(data_path.split("/")[:-1]) or "."

    while True:
        if autosave:
            # autosave acts as a file name if it's a string,
            # otherwise use default name based on T_a
            if isinstance(autosave, str):
                name = autosave
            else:
                name = str(T_a)
            break

        save = input("Save data? (y/n) ")
        if save.lower() in {"y", "yes"}:
            dname = input(f"Use default name? [{T_a}] (y/n) : ")
            if dname.lower() in {"y", "yes"}:
                name = str(T_a)
                break
            if dname.lower() in {"n", "no"}:
                name = input("Enter file name (.pkl is automatically appended): ")
                break
            print("Invalid input. Please enter 'y' or 'n'.")
        elif save.lower() in {"n", "no"}:
            return  # do not save
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    filepath = f"{data_branch}/{name}.pkl"
    with open(filepath, "wb") as handle:
        pickle.dump(summary, handle)


# ---------- Main function ----------

def summarize_flash_data(
    T_a,
    data_path,
    ref_segs,
    *,
    bg_path=None,
    cut_idx=200,
    temp_polynomial=None,
    static_polynomial=None,
    plot_range=(-20, 220),
    t_a_start_exponent=-3,
    all_t_a=None,
    segs_to_keep=None,
    T1=None,
    T2=None,
    autosave=False,
    quiet=False,
    correct_baseline=True,
    multiref=None,
):
    """
    End-to-end analysis pipeline for flash DSC data at a single annealing temperature.

    Returns
    -------
    summary : dict
        {"Ta": T_a, "Ta_corr": Ta_corrected, "Tfq": Tfq, "all_ta": all_t_a, "all_Tf": all_Tf}
    """

    # 1) Load curves (and optional background)
    data, background = _load_curves(data_path, bg_path)

    # 2) Temperature correction
    data, background = _apply_temp_polynomial(data, background, temp_polynomial)

    # 3) Slice unwanted initial points
    data = _slice_curves(data, cut_idx)

    # 4) Background subtraction
    data = _subtract_background_with_optional_plots(data, background, quiet)

    # 5) Show candidate reference segments, then build reference from selection
    _plot_reference_segments_for_selection(data, ref_segs, quiet)
    ref, data, multiref_data = _build_reference_curve(data, ref_segs, segs_to_keep, multiref)

    # 6) Choose a "sample" curve for plotting (largest segment number)
    sample_key = sorted(data.keys())[-1]
    sample = data[sample_key]

    # 7) Plot ref+sample to help choose T1/T2, then choose them
    _plot_reference_and_sample_for_ranges(ref, sample, plot_range, quiet)
    T1, T2 = _choose_T_ranges(T1, T2, quiet)

    # 8) Baseline correction for sample and all curves
    if correct_baseline:
        sample_corrected, data = _baseline_correct_all(data, ref, sample, T1, T2, quiet, multiref_data)
    else:
        sample_corrected = sample

    # 9) Plot fictive construction + corrected curves
    _plot_fictive_example(sample_corrected, sample, T1, T2, quiet)
    _plot_all_corrected(data, quiet)

    # 10) Aging times t_a
    if all_t_a is None:
        all_t_a = _generate_default_ta(len(data), t_a_start_exponent)

    # 11) Compute fictive temperatures for reference (Tfq) and all curves
    Tfq = fictive_temperature(ref, T1, T2)["Tf"]
    all_Tf = _compute_all_Tf(data, T1, T2)

    # 12) Plot Tf vs ta
    _plot_Tf_vs_ta(all_t_a, all_Tf, quiet)

    # 13) Correct Ta. Uses temp polynomial if static isn´t given. If neither is given, do nothing
    if static_polynomial is None:
        static_polynomial = temp_polynomial
    Ta_corrected = _compute_Ta_corrected(T_a, static_polynomial)

    if not quiet:
        print(f"\nTf min is {np.min(all_Tf):.2f} °C")
        print(f"Ta corrected is {Ta_corrected:.2f} °C")

    # 14) Build summary and maybe save
    summary = {
        "Ta": T_a,
        "Ta_corr": Ta_corrected,
        "Tfq": Tfq,
        "all_ta": all_t_a,
        "all_Tf": all_Tf,
        "ref": ref,
        "data": data
    }

    _maybe_save_summary(summary, data_path, T_a, autosave)

    return summary
