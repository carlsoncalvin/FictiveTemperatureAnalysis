import re
from typing import Dict, List, Tuple, Optional
import numpy as np

# --- regex helpers ------------------------------------------------------------

# finds first integer anywhere in the curve-name line
FIRST_INT = re.compile(r"(\d+)")

# numeric row splitter based on spaces
ROW_SPLIT = re.compile(r"\s+")


# --- core parsing -------------------------------------------------------------

def parse_segment_number(curve_name_line: str) -> Optional[int]:
    """
    Extract the segment number from the 'Curve Name:' value line
    by taking the first integer we see. Returns None if nothing is found.
    """
    m = FIRST_INT.search(curve_name_line)
    return int(m.group(1)) if m else None


def _next_nonempty(lines: List[str], i: int) -> Tuple[Optional[str], int]:
    """
    Return (line, new_index) at the first non-empty line starting at i.
    If none, returns (None, len(lines)).
    """
    n = len(lines)
    while i < n:
        if lines[i].strip():
            return lines[i], i
        i += 1
    return None, n


def _looks_like_values_header(line: str) -> bool:
    """
    Rough check that we're at the 'Curve Values:' header. Tolerant to variation.
    """
    return "Curve Values" in line


def _parse_units_line(line: str) -> Dict[str, str]:
    """
    Parse something like '[s] [°C] [°C] [mW]' aligned under columns.
    Map units to expected columns. Missing units become ''.
    """
    units = re.findall(r"\[([^\]]*)\]", line)
    # map onto expected column order
    expected_cols = ["Index", "t", "Ts", "Tr", "Value"]
    out = {c: "" for c in expected_cols}
    if not units:
        raise Exception(f"Could not parse units line: {line}")
    else:
        # Try to align from the right (Index lacks a unit)
        for col, unit in zip(expected_cols[::-1], units[::-1]):
            if col == "Index":
                continue
            out[col] = unit
        return out


def _parse_values_table(lines: List[str],
                        i: int) -> Tuple[Dict[str, np.ndarray], Dict[str, str], int]:
    """
    Parse the 'Curve Values' block starting at index i (which should point to the line
    AFTER the line containing 'Curve Values:').
    Returns (data_dict, units_dict, new_index).
    Stops when hitting a blank/non-numeric row or EOF.
    """
    n = len(lines)
    # check header line is correct
    expected_cols = ["Index", "t", "Ts", "Tr", "Value"]
    if lines[i].split() != expected_cols:
        raise Exception(f"Line {i}: Expected 'Index', 't', 'Ts', 'Tr', 'Value' columns, "
                        f"got: {lines[i]}")
    i += 1

    # parse the units line just below header
    maybe_units_line = lines[i]
    units = _parse_units_line(maybe_units_line)
    i += 1

    # prepare holders
    idxs: List[int] = []
    t: List[float] = []
    Ts: List[float] = []
    Tr: List[float] = []
    val: List[float] = []

    # consume numeric rows
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            break  # end of table on blank line

        # split row
        parts = ROW_SPLIT.split(stripped)

        # try to parse data
        try:
            idx = int(parts[0])
            tt = float(parts[1])
            ts = float(parts[2])
            tr = float(parts[3])
            v = float(parts[4])
        except ValueError:
            break

        idxs.append(idx)
        t.append(tt)
        Ts.append(ts)
        Tr.append(tr)
        val.append(v)
        i += 1

    data = {
        "Index": np.asarray(idxs, dtype=int),
        "t": np.asarray(t, dtype=float),
        "Ts": np.asarray(Ts, dtype=float),
        "Tr": np.asarray(Tr, dtype=float),
        "Value": np.asarray(val, dtype=float),
    }
    return data, units, i


def parse_mettler_dsc_txt(text: str) -> Dict[int|str, Dict]:
    """
    Parse a METTLER DSC export that contains multiple segments in a single .txt file.
    Scan for 'Curve Name:' then 'Curve Values:' blocks and extract:
      - segment_number (int)
      - raw_curve_name (str)
      - units (dict)
      - data: (dict)          # dict of numpy arrays: Index, t, Ts, Tr, Value
      - Sample: (str)
      - Method: (str)
      - Sample Holder: (str)
      - dTs(T): (str)         # temperature calibration polynomial

    Return:
      {
        <segment_number>: {
            'segment_number': int,
            'raw_curve_name': str,
            'units': {'Index': '', 't': 's', 'Ts': '°C', 'Tr': '°C', 'Value': 'mW'},
            'data': {'Index': np.array, 't': np.array, 'Ts': np.array,
                     'Tr': np.array, 'Value': np.array},
            'Sample': str,
            'Method': str,
            'Sample Holder': str,
            dTs(T): str,
        },
        ...
      }

    Notes:
    - If the same segment number appears twice, later ones will get suffixed with
      _dup1, _dup2 in the returned dict keys (to avoid overwriting).
    """
    lines = text.splitlines()
    i = 0
    n = len(lines)

    results: Dict[int|str, Dict] = {}
    dup_counts: Dict[int, int] = {}

    # initialize segment key holder
    current_seg_key = None

    while i < n:
        line = lines[i].strip()

        # Segment start
        if line.startswith("Curve Name:"):
            i += 1
            name_line, i = _next_nonempty(lines, i)
            if name_line is None:
                break

            seg_no = parse_segment_number(name_line)
            if seg_no is None:
                raise Exception(f"No segment number found in 'Curve Name:' line: {name_line}")

            # Find the next 'Curve Values:' block
            found_values = False
            while i < n:
                if _looks_like_values_header(lines[i]):
                    found_values = True
                    i += 1
                    break
                i += 1

            if not found_values:
                raise Exception(f"No 'Curve Values:' block found after 'Curve Name:' at line {i}")

            data, units, i = _parse_values_table(lines, i)

            # define segment key and check for duplicates
            seg_key = seg_no
            if seg_key in results:
                dup_counts[seg_key] = dup_counts.get(seg_key, 0) + 1
                seg_key = f"{seg_no}_dup{dup_counts[seg_no]}"

            current_seg_key = seg_key

            results[seg_key] = {
                "segment_number": seg_no,
                "raw_curve_name": name_line.strip(),  # keep junk for transparency
                "units": units,
                "data": data,
            }
            continue

        # capture metadata
        if line.startswith("Sample:"):
            i += 1
            val, i = _next_nonempty(lines, i)
            if val is not None:
                sample = val.strip()
                results[current_seg_key]["sample"] = sample
            continue

        if line.startswith("Method:"):
            i += 1
            val, i = _next_nonempty(lines, i)
            if val is not None:
                method = val.strip()
                results[current_seg_key]["method"] = method
            continue

        if line.startswith("dTs(T)"):
            temp_poly = line.strip().split(":")[1]
            results[current_seg_key]["dTs(T)"] = temp_poly
            i += 1
            continue

        if line.startswith("Sample Holder"):
            i += 1
            val, i = _next_nonempty(lines, i)
            if val is not None:
                holder = val.strip()
                results[current_seg_key]["sample_holder"] = holder
            continue

        i += 1

    return results


# --- Convenience I/O ----------------------------------------------------------

def parse_mettler_dsc_txt_file(path: str) -> Dict[int|str, Dict]:
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        return parse_mettler_dsc_txt(f.read())

def extract_T_and_dsc(parsed_dict: Dict[int|str, Dict],
                      flip_dsc: bool = True) -> Dict[int|str, Dict[str, np.ndarray] ]:
    """
    Extracts only the temperature (Ts) and DSC signal (Value) from a parsed DSC dataset,
    renaming them to 'T' and 'dsc' respectively.

    Parameters
    ----------
    parsed_dict : dict
        Dictionary returned by the DSC parser, keyed by segment number.
    flip_dsc : bool
        Optional, default True. If True, flip the sign of the DSC signal

    Returns
    -------
    dict
        Dictionary of the same structure, but each segment contains only:
        {
            "T": ndarray,
            "dsc": ndarray
        }
    """
    result = {}
    for seg_num, seg_data in parsed_dict.items():
        data = seg_data.get("data", {})
        if "Ts" in data and "Value" in data:
            if flip_dsc:
                result[seg_num] = {
                    "T": data["Ts"],
                    "dsc": -data["Value"]
                }
            else:
                result[seg_num] = {
                    "T": data["Ts"],
                    "dsc": data["Value"]
                }
    return result