import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


Number = Union[int, float]


@dataclass(frozen=True)
class BuildCsvConfig:
    """
    Configuration for building the wide CSV.
    - include_tf: whether to include Tf_n columns
    - t_key: input key for temperature-like series
    - hf_key: input key for HF-like series (called dsc in your input)
    - tf_key: input key for Tf series (optional)
    - passthrough_keys: global (non-segment) keys to include as columns (scalar or series)
    """
    include_tf: bool = False
    t_key: str = "T"
    hf_key: str = "dsc"
    tf_key: str = "Tf"
    passthrough_keys: Tuple[str, ...] = ()


def _is_sequence_like(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _get_series(record: Any, key: str) -> Optional[Sequence[Any]]:
    """
    Extract a series from a segment record.

    Supports:
      - dict-like: record[key]
      - object with attribute: record.key
    """
    if record is None:
        return None

    # dict-like
    if isinstance(record, Mapping) and key in record:
        return record[key]

    # attribute-like
    if hasattr(record, key):
        return getattr(record, key)

    return None


def _normalize_series(series: Any) -> Sequence[Any]:
    """
    Normalize a series-like value into a sequence.
    - If scalar -> treat as length-1
    - If list/tuple -> use as-is
    - If pandas Series / numpy array -> attempt to convert via list()
    """
    if series is None:
        return []
    if _is_sequence_like(series):
        return series
    # try iterable conversion (handles numpy/pandas)
    try:
        return list(series)
    except TypeError:
        return [series]


def _max_len(seqs: Iterable[Sequence[Any]]) -> int:
    m = 0
    for s in seqs:
        if len(s) > m:
            m = len(s)
    return m


def build_wide_rows(
    data_by_segment: Mapping[int, Any],
    config: BuildCsvConfig,
    *,
    global_data: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Build header + rows for a wide CSV.

    Parameters
    ----------
    data_by_segment:
        Dict keyed by segment number. Each value holds at least:
          - T series at config.t_key
          - HF series at config.hf_key (input name is 'dsc' by default)
          - optional Tf series at config.tf_key, included only if config.include_tf is True

        Each segment record may contain other keys; you can add handling if you want
        per-segment extra columns.

    config:
        BuildCsvConfig

    global_data:
        Optional dict of additional (non-segment) information.
        Keys listed in config.passthrough_keys will be added as columns.
        Each passthrough value may be scalar or a series; scalars are repeated for all rows.

    Returns
    -------
    header: list of column names
    rows: list of dict rows (ready for csv.DictWriter)
    """
    if not data_by_segment:
        raise ValueError("data_by_segment is empty.")

    segments = sorted(data_by_segment.keys())

    # Extract series per segment
    seg_T: Dict[int, Sequence[Any]] = {}
    seg_HF: Dict[int, Sequence[Any]] = {}
    seg_Tf: Dict[int, Sequence[Any]] = {}

    for seg in segments:
        rec = data_by_segment[seg]

        T = _normalize_series(_get_series(rec, config.t_key))
        HF = _normalize_series(_get_series(rec, config.hf_key))  # input is 'dsc'
        Tf = _normalize_series(_get_series(rec, config.tf_key)) if config.include_tf else []

        seg_T[seg] = T
        seg_HF[seg] = HF
        if config.include_tf:
            seg_Tf[seg] = Tf

    # Determine output row count (max across all included series)
    all_series: List[Sequence[Any]] = []
    all_series.extend(seg_T.values())
    all_series.extend(seg_HF.values())
    if config.include_tf:
        all_series.extend(seg_Tf.values())

    # Include passthrough series in row count if provided
    global_data = global_data or {}
    passthrough_series: Dict[str, Sequence[Any]] = {}
    for k in config.passthrough_keys:
        v = global_data.get(k)
        passthrough_series[k] = _normalize_series(v)
        all_series.append(passthrough_series[k])

    n_rows = _max_len(all_series)
    if n_rows == 0:
        raise ValueError("No data found to write (all series empty).")

    # Build header: passthrough columns first, then per-segment repeated columns
    header: List[str] = []
    header.extend(list(config.passthrough_keys))

    for seg in segments:
        header.append(f"T_{seg}")
        header.append(f"HF_{seg}")  # output name HF, input key dsc
        if config.include_tf:
            header.append(f"Tf_{seg}")

    # Build rows
    rows: List[Dict[str, Any]] = []
    for i in range(n_rows):
        row: Dict[str, Any] = {}

        # passthrough values
        for k in config.passthrough_keys:
            seq = passthrough_series[k]
            if len(seq) == 1 and n_rows > 1:
                row[k] = seq[0]  # scalar repeated
            else:
                row[k] = seq[i] if i < len(seq) else None

        # segment values
        for seg in segments:
            T = seg_T[seg]
            HF = seg_HF[seg]

            row[f"T_{seg}"] = T[i] if i < len(T) else None
            row[f"HF_{seg}"] = HF[i] if i < len(HF) else None

            if config.include_tf:
                Tf = seg_Tf[seg]
                row[f"Tf_{seg}"] = Tf[i] if i < len(Tf) else None

        rows.append(row)

    return header, rows


def write_single_csv(
    data_by_segment: Mapping[int, Any],
    output_csv_path: Union[str, Path],
    *,
    include_tf: bool = False,
    global_data: Optional[Mapping[str, Any]] = None,
    passthrough_keys: Sequence[str] = (),
    t_key: str = "T",
    hf_input_key: str = "dsc",
    tf_key: str = "Tf",
) -> Path:
    """
    User-facing function.

    - include_tf toggles whether Tf_n columns appear.
    - hf_input_key defaults to 'dsc' (mapped to output column prefix 'HF_').
    """
    config = BuildCsvConfig(
        include_tf=include_tf,
        t_key=t_key,
        hf_key=hf_input_key,
        tf_key=tf_key,
        passthrough_keys=tuple(passthrough_keys),
    )

    header, rows = build_wide_rows(
        data_by_segment=data_by_segment,
        config=config,
        global_data=global_data,
    )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return output_csv_path
