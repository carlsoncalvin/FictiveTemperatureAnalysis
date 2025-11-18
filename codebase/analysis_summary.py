import numpy as np
import matplotlib.pyplot as plt
import pickle

from data_parser import parse_mettler_dsc_txt_file, extract_T_and_dsc
from fictive_temp_functions import baseline_correction, fictive_temperature
from utils import average_curves, subtract_background, slice_curve_idx

def summarize_flash_data(T_a,
                         data_path,
                         ref_segs,
                         *,
                         bg_path=None,
                         cut_idx=200,
                         temp_polynomial=None,
                         plot_range=(-20, 220),
                         t_a_start_exponent=-3,
                         all_t_a=None,
                         segs_to_keep=None,
                         T1=None, T2=None,
                         autosave=False,
                         quiet=False,):

    if bg_path is not None:
        # get background curve
        bg_results = parse_mettler_dsc_txt_file(bg_path)
        bg_data = extract_T_and_dsc(bg_results)
        background = average_curves(bg_data)

    # get data
    results = parse_mettler_dsc_txt_file(data_path)

    # get only necessary data for analysis
    data = extract_T_and_dsc(results)

    # correct temperature
    # T_corrected - T_exp = a + b * T_exp
    if temp_polynomial is not None:
        a, b = temp_polynomial
        for curve in data.values():
            curve["T"] = a + b*curve["T"] + curve["T"]
        if bg_path is not None:
            background["T"] = a + b*background["T"] + background["T"]

    # slice unwated data
    for key in data:
        data[key] = slice_curve_idx(data[key], i_start=cut_idx)

    if quiet:
        if bg_path is not None:
            data = subtract_background(data, background)
    elif not quiet and bg_path is not None:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
        for curve in data.values():
            ax1.plot(curve["T"], curve["dsc"])

        # subtract background
        data = subtract_background(data, background)
        for curve in data.values():
            ax2.plot(curve["T"], curve["dsc"])
        ax1.title.set_text("Raw data")
        ax2.title.set_text("Background subtracted")
        plt.show()

        plt.figure()
        for seg in ref_segs:
            plt.plot(data[seg]["T"], data[seg]["dsc"], label=f"Segment {seg}")
        plt.legend()
        plt.title("Reference curves")
        plt.show()

    # logic to determine reference segments to use
    if not segs_to_keep:
        while True:
            keep_segs = input("Keep all reference segments? (y/n) ")
            if keep_segs.lower() in ["y", "yes"]:
                segs_to_keep = ref_segs
                break
            elif keep_segs.lower() in ["n", "no"]:
                while True:
                    segs_to_keep = input("Enter segment numbers to keep separated by spaces: ")
                    segs_to_keep = segs_to_keep.split()
                    try:
                        segs_to_keep = [int(seg) for seg in segs_to_keep]
                    except ValueError:
                        print("Invalid input. Please enter integer segment numbers separated by spaces.")
                        continue
                    if all(seg in ref_segs for seg in segs_to_keep):
                        break
                    else:
                        print(f"Invalid input. Valid segment numbers are {ref_segs}.")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    selected = {key: data[key] for key in segs_to_keep}

    # average reference segments to get 1 curve
    ref = average_curves(selected)

    # remove all references from data dict
    for seg in ref_segs:
        data.pop(seg)

    # take last segment for plotting
    sample = data[sorted(data.keys())[-1]]

    # plot curves to find good T1 and T2
    if not quiet:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        ax1, ax2 = axs
        for ax in axs:
            ax.plot(ref["T"], ref["dsc"], label="Reference")
            ax.set_xlabel("T /$\degree$C")
            ax.grid(axis="x")

        ax1.plot(sample["T"], sample["dsc"], label="Sample")
        ax1.set_ylabel("Heat Flow / mW")

        ax2.plot(sample["T"], sample["dsc"], label="Sample")
        x1, x2 = plot_range
        ax2.set_xlim(x1, x2)
        ax2.set_xticks(range(x1, x2, 20))

        fig.tight_layout()
        plt.show()

    # input logic for getting T1 and T2
    if not T1 or not T2:
        while True:
            keep = input("Keep default T1 and T2? (-40, 20), (140, 220) (y/n) ")
            if keep.lower() in ["y", "yes"]:
                T1 = (-40, 20)
                T2 = (140, 220)
                break
            elif keep.lower() in ["n", "no"]:
                while True:
                    T_ranges = input("Enter T1 and T2 ranges separated by spaces: ")
                    try:
                        T_ranges = [int(T) for T in T_ranges.split()]
                    except ValueError:
                        print("Invalid input. Please enter integer temperatures separated by spaces.")
                        continue
                    if len(T_ranges) == 4:
                       T1 = (T_ranges[0], T_ranges[1])
                       T2 = (T_ranges[2], T_ranges[3])
                       break
                    else:
                        print("Invalid input. Please enter 4 integers separated by spaces.")
                break

    # compute baseline-corrected curve and save
    dsc_corrected = baseline_correction(sample, ref, T1, T2)

    sample_corrected = sample.copy()
    sample_corrected["dsc"] = dsc_corrected

    # plot raw and corrected curves
    if not quiet:
        fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        ax1, ax2 = axs
        for ax in axs:
            ax.plot(ref["T"], ref["dsc"], label="Reference")
            ax.set_xlabel("T /$\degree$C")

        ax1.plot(sample["T"], sample["dsc"], label="Sample")
        ax1.set_title("Sample and Reference Raw")
        ax1.set_ylabel("Heat Flow / mW")

        ax2.plot(sample_corrected["T"], sample_corrected["dsc"], label="Corrected Sample")
        ax2.set_title("Sample and Reference Corrected")

        fig.tight_layout()
        for ax in axs:
            ax.legend()
        plt.show()

    # correct all data
    for curve in data.values():
        curve["dsc"] = baseline_correction(curve, ref, T1, T2)

    # compute and plot fictive temperature example
    if not quiet:
        Tf_results = fictive_temperature(sample_corrected, T1, T2)

        plt.plot(sample_corrected["T"], sample_corrected["dsc"])

        T_slice = Tf_results["T_slice"]

        plt.plot(T_slice, Tf_results["glass_line"], label="Glass line")
        plt.plot(T_slice, Tf_results["liquid_line"], label="Liquid line")
        plt.vlines(Tf_results["Tf"], min(sample["dsc"]), max(sample["dsc"]), color="k", linestyles="--",
                   label=f"T$_f$={Tf_results["Tf"]}")
        plt.xlabel("T /$\degree$C")
        plt.ylabel("Heat Flow / mW")
        plt.legend()
        plt.show()

        # plot all corrected curves
        plt.figure()
        for curve in data.values():
            plt.plot(curve["T"], curve["dsc"])
        plt.xlabel("T /$\degree$C")
        plt.ylabel("Heat Flow / mW")
        plt.show()

    # get aging times
    if all_t_a:
        pass
    else:
        # compute aging times from default pattern
        all_t_a = np.empty_like(list(data.keys()), dtype=float)

        i = 0
        for e in range(t_a_start_exponent, 6):
            for p in [1, 2, 5]:
                if i == len(all_t_a):
                    break
                all_t_a[i] = p * 10 ** e
                i += 1
        if all_t_a[-1] == 100000:
            all_t_a[-1] = 90000

    all_Tf = np.empty_like(all_t_a)

    for i, key in enumerate(sorted(data)):
        Tf_results = fictive_temperature(data[key], T1, T2)
        all_Tf[i] = Tf_results["Tf"]

    if not quiet:
        plt.figure()
        plt.scatter(all_t_a, all_Tf)
        plt.xscale("log")
        plt.xlabel("t$_a$ / s")
        plt.ylabel("T$_f$ / $\degree$ C")
        plt.show()

    Ta_corrected = a + b * T_a + T_a
    if not quiet:
        print(f"\nTf min is {min(all_Tf)}")
        print(f"Ta corrected is {Ta_corrected}")

    while True:
        save = autosave or input("Save data? (y/n) ")
        if autosave or save.lower() in ["y", "yes"]:
            dname = autosave or input(f"Use default name? [{T_a}] (y/n) : ")
            if autosave or dname.lower() in ["y", "yes"]:
                name = T_a
            elif dname.lower() in ["n", "no"]:
                name = input("Enter file name (.pkl is automatically appended): ")
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
                continue
            data_branch = "/".join(data_path.split("/")[:-1])
            summary = {"Ta": T_a, "Ta_corr": Ta_corrected, "all_ta": all_t_a, "all_Tf": all_Tf}
            # serialize data
            with open(f'{data_branch}/{name}.pkl', 'wb') as handle:
                pickle.dump(summary, handle)
            break
        elif save.lower() in ["n", "no"]:
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

