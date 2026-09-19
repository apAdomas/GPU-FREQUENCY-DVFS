#!/usr/bin/env python3
"""Switch-aware kernel grouping illustration from measured clocks.

In:  auto_time_s summary + selected/effective clock pair per kernel.
Out: fig_grouping_dependencies.*, fig_grouping_schedule_timeline.*, grouping_schedule.csv

Dependencies follow real benchmark groups when available. The schedule is a
greedy, dependency-preserving grouping; clock switches are drawn as overlay,
not idle time.
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, ConnectionPatch
from matplotlib.ticker import MaxNLocator

WORKLOAD_COLORS = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#B07AA1",  # purple
    "#76B900",  # NVIDIA green
    "#76B7B2",  # cyan
    "#EDC948",  # yellow
]

CORE_GREENS = [
    "#76B900",  # NVIDIA green
    "#2E7D32",  # dark green
    "#9CCC65",  # light green
]

BRIGHT_COLORS = [
    "#4C9BE8",  # bright blue
    "#F2A341",  # warm amber
    "#9B6FC4",  # purple
    "#3FB5C4",  # cyan
    "#F2C14E",  # yellow
    "#E879A6",  # pink
    "#FF7A5C",  # coral
    "#5BC8AF",  # mint
    "#7E57C2",  # violet
    "#1F9ED1",  # azure
    "#C0CA33",  # lime
    "#EF6C7A",  # rose
    "#26A69A",  # teal
    "#FFB300",  # gold
    "#8D6E63",  # warm taupe
    "#5C8AE6",  # periwinkle
    "#AB47BC",  # orchid
]


def pick_clock_columns(sel: pd.DataFrame):
    candidates = [
        ("selected_effective_core_clock", "selected_effective_mem_clock"),
        ("effective_core_clock", "effective_mem_clock"),
        ("oracle_effective_core_clock", "oracle_effective_mem_clock"),
    ]

    for core_col, mem_col in candidates:
        if core_col in sel.columns and mem_col in sel.columns:
            return core_col, mem_col

    raise ValueError(
        "Could not find clock columns. Expected one of: "
        "selected_effective_core_clock/selected_effective_mem_clock, "
        "effective_core_clock/effective_mem_clock, or "
        "oracle_effective_core_clock/oracle_effective_mem_clock."
    )


def clock_label(value) -> str:
    s = str(value).strip()
    if s.lower() == "auto":
        return "auto"

    try:
        return str(int(round(float(s))))
    except (TypeError, ValueError):
        return s


def savefig_both(fig, out_pdf: Path):
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")


def load_data(summary_csv, selection_csv, constraint=None, feature_set=None, model=None):
    df = pd.read_csv(summary_csv)
    sel = pd.read_csv(selection_csv)

    if constraint and "constraint_name" in sel.columns:
        sel = sel[sel["constraint_name"].astype(str) == str(constraint)]

    if feature_set and "feature_set" in sel.columns:
        sel = sel[sel["feature_set"].astype(str) == str(feature_set)]

    if model and "model" in sel.columns:
        sel = sel[sel["model"].astype(str) == str(model)]

    core_col, mem_col = pick_clock_columns(sel)

    sel = sel[["kernel_id", core_col, mem_col]].drop_duplicates("kernel_id")
    sel = sel.rename(columns={core_col: "pref_core", mem_col: "pref_mem"})

    out = df.merge(sel, on="kernel_id", how="inner")

    out["auto_time_ms"] = out["auto_time_s"].astype(float) * 1000.0

    if "auto_energy_mJ" in out.columns:
        out["auto_energy_mJ"] = out["auto_energy_mJ"].astype(float)
    elif "auto_energy_j" in out.columns:
        out["auto_energy_mJ"] = out["auto_energy_j"].astype(float) * 1000.0
    elif "auto_energy_J" in out.columns:
        out["auto_energy_mJ"] = out["auto_energy_J"].astype(float) * 1000.0
    else:
        raise ValueError(
            "summary CSV must contain auto energy column: "
            "auto_energy_mJ, auto_energy_j, or auto_energy_J"
        )

    out["pref_core"] = out["pref_core"].map(clock_label)
    out["pref_mem"] = out["pref_mem"].map(clock_label)
    out["pref_pair"] = out["pref_core"] + "/" + out["pref_mem"]

    return out


def make_constructed_traces_all_kernels(
    df: pd.DataFrame,
    trace_lengths=None,
    max_nodes=None,
    max_kernel_ms=None,
):
    """
    Construct synthetic dependency traces from measured kernels.

    Each node is a real measured kernel with measured runtime and selected
    core-memory clock pair. The dependency edges are synthetic and only impose
    an order within each constructed workload.
    """
    d = df.copy()

    if max_kernel_ms is not None and max_kernel_ms > 0:
        d = d[d["auto_time_ms"] <= max_kernel_ms].copy()

    if d.empty:
        raise ValueError("No kernels left after filtering.")

    # Use all kernels unless capped.
    d = d.sort_values(["auto_time_ms", "kernel_id"], ascending=[False, True]).reset_index(drop=True)

    if max_nodes is not None and max_nodes > 0:
        d = d.head(max_nodes).copy()

    n = len(d)

    if trace_lengths is None:
        if n >= 65:
            trace_lengths = [14, 12, 11, 10, 9, 9]
        else:
            base = n // 6
            rem = n % 6
            trace_lengths = [base + (1 if i < rem else 0) for i in range(6)]
            trace_lengths = [x for x in trace_lengths if x > 0]

    if sum(trace_lengths) != n:
        trace_lengths = list(trace_lengths)
        trace_lengths[-1] += n - sum(trace_lengths)

    traces = [[] for _ in trace_lengths]

    # Distribute long kernels across traces first, so one trace does not get all large kernels.
    node_id = 1
    positions = [0 for _ in trace_lengths]

    for _, r in d.iterrows():
        # Put next kernel into the currently shortest total-runtime trace that still has capacity.
        candidates = [
            i for i in range(len(traces))
            if positions[i] < trace_lengths[i]
        ]

        if not candidates:
            break

        target_i = min(
            candidates,
            key=lambda i: sum(k["auto_time_ms"] for k in traces[i])
        )

        workload_idx = target_i + 1

        traces[target_i].append({
            "node": node_id,
            "workload_idx": workload_idx,
            "workload": r["workload"],
            "kernel": r["kernel"],
            "kernel_id": r["kernel_id"],
            "auto_time_ms": float(r["auto_time_ms"]),
            "auto_energy_mJ": float(r["auto_energy_mJ"]),
            "pref_core": r["pref_core"],
            "pref_mem": r["pref_mem"],
            "pref_pair": r["pref_pair"],
        })

        positions[target_i] += 1
        node_id += 1

    #  within each trace, mix long and short kernels.
    for i, trace in enumerate(traces):
        trace_sorted = sorted(trace, key=lambda k: k["auto_time_ms"], reverse=True)

        mixed = []
        left = 0
        right = len(trace_sorted) - 1
        take_short = True

        while left <= right:
            if take_short:
                mixed.append(trace_sorted[right])
                right -= 1
            else:
                mixed.append(trace_sorted[left])
                left += 1
            take_short = not take_short

        traces[i] = mixed

    return [t for t in traces if len(t) > 0]


def kernel_order_key(row):
    """
    Deterministic ordering inside a benchmark workload.
    """
    k = str(row["kernel"]).lower()
    kid = str(row["kernel_id"]).lower()

    patterns = [
        r"kernel[_-]?(\d+)",
        r"step[_-]?(\d+)",
        r"shared[_-]?(\d+)",
        r"mm\d?_kernel(\d+)",
    ]

    for p in patterns:
        m = re.search(p, k)
        if m:
            return (0, int(m.group(1)), kid)

    manual = {
        # Particlefilter
        "likelihood": 1,
        "sum": 2,
        "normalize_weights": 3,
        "find_index": 4,

        # SRAD v1
        "extract": 1,
        "prepare": 2,
        "reduce": 3,
        "srad": 4,
        "srad2": 5,
        "compress": 6,

        # CFD / Euler
        "compute_step_factor": 1,
        "compute_flux": 2,
        "compute_flux_contributions": 3,
        "time_step": 4,

        # Hotspot / misc
        "mean_kernel": 1,
        "std_kernel": 2,
        "reduce_kernel": 3,
        "corr_kernel": 4,
        "covar_kernel": 3,
    }

    return (0, manual.get(k, 999), kid)


def make_real_workload_traces(df: pd.DataFrame):
    """
    Use real benchmark workloads as dependency chains.

    Each workload is independent. Kernels inside each workload preserve a
    deterministic benchmark-local order. Single-kernel workloads remain valid
    one-node independent workloads.
    """
    traces = []
    node_id = 1

    stats = (
        df.groupby("workload")
        .agg(
            n_kernels=("kernel_id", "count"),
            total_time_ms=("auto_time_ms", "sum"),
        )
        .reset_index()
        .sort_values(
            ["n_kernels", "total_time_ms", "workload"],
            ascending=[False, False, True],
        )
    )

    for workload_idx, workload in enumerate(stats["workload"].tolist(), start=1):
        g = df[df["workload"] == workload].copy()
        g["_order_key"] = g.apply(kernel_order_key, axis=1)
        g = g.sort_values("_order_key")

        chain = []

        for _, r in g.iterrows():
            chain.append({
                "node": node_id,
                "workload_idx": workload_idx,
                "workload": r["workload"],
                "kernel": r["kernel"],
                "kernel_id": r["kernel_id"],
                "auto_time_ms": float(r["auto_time_ms"]),
                "auto_energy_mJ": float(r["auto_energy_mJ"]),
                "pref_core": r["pref_core"],
                "pref_mem": r["pref_mem"],
                "pref_pair": r["pref_pair"],
            })
            node_id += 1

        if chain:
            traces.append(chain)

    return traces


def clock_key(kernel) -> str:
    return kernel["pref_pair"]


def parse_pair(pair: str):
    core, mem = pair.split("/")
    return float(core), float(mem)


def clock_distance(pair_a: str, pair_b: str) -> float:
    """
    Normalized distance between two core-memory clock pairs.
    Used only for grouping fallback when no exact compatible ready kernel exists.
    """
    ca, ma = parse_pair(pair_a)
    cb, mb = parse_pair(pair_b)

    # Normalize so memory MHz does not dominate purely because numbers are larger.
    return abs(ca - cb) / 2000.0 + abs(ma - mb) / 10000.0


def schedule_ready_set(traces, switch_ms=1.0):
    """
    Greedy dependency-preserving grouping using exact core-memory clock pairs first.

    Each group selects one core-memory clock pair. The scheduler first packs
    ready kernels whose preferred pair matches the selected pair. If the group
    is still shorter than the assumed switching interval S, it adds ready
    kernels with the closest preferred pair, rather than starting another
    clock group immediately.
    """
    positions = [0 for _ in traces]
    groups = []
    group_id = 1

    while True:
        ready = []
        for wi, chain in enumerate(traces):
            if positions[wi] < len(chain):
                ready.append((wi, chain[positions[wi]]))

        if not ready:
            break

        # Pick target pair from ready kernels: largest ready runtime mass.
        mass_by_pair = {}
        example_by_pair = {}

        for wi, k in ready:
            key = k["pref_pair"]
            mass_by_pair[key] = mass_by_pair.get(key, 0.0) + k["auto_time_ms"]
            example_by_pair[key] = k

        target_pair = max(mass_by_pair.items(), key=lambda x: x[1])[0]
        target_example = example_by_pair[target_pair]

        group = []
        group_runtime = 0.0

        while group_runtime < switch_ms:
            ready = []
            for wi, chain in enumerate(traces):
                if positions[wi] < len(chain):
                    ready.append((wi, chain[positions[wi]]))

            if not ready:
                break

            # First prefer exact clock-pair matches.
            exact = [(wi, k) for wi, k in ready if k["pref_pair"] == target_pair]

            if exact:
                candidates = exact
            else:
                # If no exact match is ready, use the closest available pair.
                candidates = sorted(
                    ready,
                    key=lambda x: (
                        clock_distance(x[1]["pref_pair"], target_pair),
                        -x[1]["auto_time_ms"],
                    ),
                )

            wi, k = candidates[0]

            group.append(k)
            group_runtime += k["auto_time_ms"]
            positions[wi] += 1

        groups.append({
            "group": group_id,
            "runtime_ms": group_runtime,
            "clock_label": target_pair,
            "clock_key": target_pair,
            "underfilled_switch_window": group_runtime < switch_ms,
            "kernels": group,
        })

        group_id += 1

    return groups


def merge_last_underfilled_group(groups, switch_ms=1.0):
    """
    If the last group is shorter than S, do not make a separate clock change
    for it. Append it to the previous group so it runs under the previous
    group's clock pair.
    """
    if len(groups) <= 1:
        return groups

    last = groups[-1]

    if last["runtime_ms"] < switch_ms:
        prev = groups[-2]
        prev["kernels"].extend(last["kernels"])
        prev["runtime_ms"] += last["runtime_ms"]
        prev["merged_last_underfilled_group"] = True
        groups = groups[:-1]

    for new_id, g in enumerate(groups, start=1):
        g["group"] = new_id
        g["underfilled_switch_window"] = g["runtime_ms"] < switch_ms

    return groups


def infer_measured_columns(meas: pd.DataFrame):
    time_candidates = [
        "time_per_launch_mean_ms",
        "time_per_launch_ms",
        "runtime_per_launch_ms",
        "mean_time_ms",
    ]
    energy_candidates = [
        "energy_per_launch_mean_mJ",
        "energy_per_launch_mJ",
        "energy_mJ",
        "mean_energy_mJ",
    ]

    time_col = next((c for c in time_candidates if c in meas.columns), None)
    energy_col = next((c for c in energy_candidates if c in meas.columns), None)

    if time_col is None:
        if "time_per_launch_s" in meas.columns:
            meas["time_per_launch_ms"] = meas["time_per_launch_s"].astype(float) * 1000.0
            time_col = "time_per_launch_ms"
        else:
            raise ValueError("Could not find measured time column.")

    if energy_col is None:
        if "energy_per_launch_J" in meas.columns:
            meas["energy_per_launch_mJ"] = meas["energy_per_launch_J"].astype(float) * 1000.0
            energy_col = "energy_per_launch_mJ"
        elif "energy_per_launch_j" in meas.columns:
            meas["energy_per_launch_mJ"] = meas["energy_per_launch_j"].astype(float) * 1000.0
            energy_col = "energy_per_launch_mJ"
        else:
            raise ValueError("Could not find measured energy column.")

    return meas, time_col, energy_col


def load_measured_clock_table(measured_csv: Path, id_map: pd.DataFrame = None):
    """
    Load full measured kernel-clock table.

    Fixed-clock rows come from measured_csv. Automatic-clock rows are added
    from id_map, so group selection always has a feasible baseline candidate.
    """
    meas = pd.read_csv(measured_csv)

    # Keep only fixed-clock rows from the measured file.
    if "is_auto_baseline" in meas.columns:
        meas = meas[meas["is_auto_baseline"] != 1].copy()

    meas, time_col, energy_col = infer_measured_columns(meas)

    for c in ("effective_core_clock", "effective_mem_clock"):
        if c not in meas.columns:
            raise ValueError(f"Measured CSV missing column: {c}")

    meas = meas.dropna(subset=["effective_core_clock", "effective_mem_clock"]).copy()

    if "kernel_id" not in meas.columns:
        if id_map is None:
            raise ValueError(
                "Measured CSV has no kernel_id column; provide id_map with "
                "columns [workload, kernel, kernel_id]."
            )
        meas = meas.merge(
            id_map[["workload", "kernel", "kernel_id"]].drop_duplicates(),
            on=["workload", "kernel"],
            how="inner",
        )

    meas["effective_core_clock"] = meas["effective_core_clock"].map(clock_label)
    meas["effective_mem_clock"] = meas["effective_mem_clock"].map(clock_label)
    meas["pair"] = meas["effective_core_clock"] + "/" + meas["effective_mem_clock"]

    fixed = (
        meas[["kernel_id", "pair", time_col, energy_col]]
        .rename(columns={time_col: "measured_time_ms", energy_col: "measured_energy_mJ"})
        .groupby(["kernel_id", "pair"], as_index=False)
        .mean()
    )

    # Add automatic-clock baseline as a valid candidate.
    if id_map is None or not {"kernel_id", "auto_time_ms", "auto_energy_mJ"}.issubset(id_map.columns):
        raise ValueError(
            "id_map must include kernel_id, auto_time_ms, and auto_energy_mJ "
            "so automatic clocks can be added as a feasible baseline."
        )

    auto = (
        id_map[["kernel_id", "auto_time_ms", "auto_energy_mJ"]]
        .drop_duplicates("kernel_id")
        .rename(columns={
            "auto_time_ms": "measured_time_ms",
            "auto_energy_mJ": "measured_energy_mJ",
        })
    )
    auto["pair"] = "auto"

    out = pd.concat(
        [fixed, auto[["kernel_id", "pair", "measured_time_ms", "measured_energy_mJ"]]],
        ignore_index=True,
    )

    return out


def get_measurement(measured_map, kernel_id, pair):
    try:
        row = measured_map.loc[(kernel_id, pair)]
    except KeyError:
        raise KeyError(f"Missing measured row for kernel_id={kernel_id}, pair={pair}")

    return float(row["measured_time_ms"]), float(row["measured_energy_mJ"])


def select_group_clocks_global_from_measured(groups, meas: pd.DataFrame, epsilon=0.0):
    """
    Select one clock pair per group using a global runtime constraint.

    This is the group-level version of the global oracle:
      minimize total measured energy across groups
      subject to total grouped runtime <= total automatic runtime * (1 + epsilon)

    Groups are fixed before this step.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError as e:
        raise RuntimeError("scipy.optimize.milp is required for global group selection.") from e

    measured_map = meas.set_index(["kernel_id", "pair"])
    all_pairs = sorted(meas["pair"].unique())

    rows = []

    for gi, g in enumerate(groups):
        kernels = g["kernels"]
        group_auto_time = sum(k["auto_time_ms"] for k in kernels)
        group_auto_energy = sum(k["auto_energy_mJ"] for k in kernels)

        for pair in all_pairs:
            total_time = 0.0
            total_energy = 0.0
            ok = True

            for k in kernels:
                try:
                    t_ms, e_mJ = get_measurement(measured_map, k["kernel_id"], pair)
                except KeyError:
                    ok = False
                    break

                total_time += t_ms
                total_energy += e_mJ

            if not ok:
                continue

            rows.append({
                "group_index": gi,
                "group_id": g["group"],
                "pair": pair,
                "group_time_ms": total_time,
                "group_energy_mJ": total_energy,
                "group_auto_time_ms": group_auto_time,
                "group_auto_energy_mJ": group_auto_energy,
            })

    c = pd.DataFrame(rows).reset_index(drop=True)

    if c.empty:
        raise RuntimeError("No group-clock candidates were constructed.")

    n_vars = len(c)
    group_indices = sorted(c["group_index"].unique())

    objective = c["group_energy_mJ"].to_numpy(dtype=float)
    constraints = []

    # Select exactly one pair per group.
    for gi in group_indices:
        row = np.zeros(n_vars)
        row[c.index[c["group_index"] == gi].to_numpy()] = 1.0
        constraints.append(LinearConstraint(row, lb=1.0, ub=1.0))

    # Global grouped runtime constraint.
    auto_total_time = (
        c.drop_duplicates("group_index")["group_auto_time_ms"].sum()
    )
    allowed_total_time = (1.0 + epsilon) * auto_total_time

    time_row = c["group_time_ms"].to_numpy(dtype=float)
    constraints.append(LinearConstraint(time_row, lb=-np.inf, ub=allowed_total_time))

    result = milp(
        c=objective,
        integrality=np.ones(n_vars),
        bounds=Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars)),
        constraints=constraints,
        options={
            "time_limit": 300,
            "mip_rel_gap": 0.0,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(f"Group MILP failed: {result.message}")

    x = np.asarray(result.x)
    selected_idx = np.where(x > 0.5)[0]
    selected = c.iloc[selected_idx].copy()

    if selected["group_index"].nunique() != len(group_indices):
        raise RuntimeError("Solver did not select exactly one clock pair per group.")

    selected_by_group = selected.set_index("group_index")

    for gi, g in enumerate(groups):
        row = selected_by_group.loc[gi]

        g["clock_key"] = row["pair"]
        g["clock_label"] = row["pair"]
        g["selected_group_time_ms"] = float(row["group_time_ms"])
        g["selected_group_energy_mJ"] = float(row["group_energy_mJ"])

    return groups


def evaluate_grouping(groups, meas: pd.DataFrame, out_csv: Path):
    """
    Compare three policies on measured values:
      auto baseline, ungrouped preferred clocks, grouped selected clocks.
    """
    measured_map = meas.set_index(["kernel_id", "pair"])

    rows = []

    for g in groups:
        group_pair = g["clock_key"]

        for k in g["kernels"]:
            pref_pair = k["pref_pair"]

            group_time_ms, group_energy_mJ = get_measurement(
                measured_map,
                k["kernel_id"],
                group_pair,
            )

            pref_time_ms, pref_energy_mJ = get_measurement(
                measured_map,
                k["kernel_id"],
                pref_pair,
            )

            rows.append({
                "group": g["group"],
                "group_pair": group_pair,
                "kernel_id": k["kernel_id"],
                "workload": k["workload"],
                "kernel": k["kernel"],

                "auto_time_ms": k["auto_time_ms"],
                "auto_energy_mJ": k["auto_energy_mJ"],

                "pref_pair": pref_pair,
                "pref_time_ms": pref_time_ms,
                "pref_energy_mJ": pref_energy_mJ,

                "group_time_ms": group_time_ms,
                "group_energy_mJ": group_energy_mJ,

                "uses_preferred_pair": group_pair == pref_pair,
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    auto_time = out["auto_time_ms"].sum()
    auto_energy = out["auto_energy_mJ"].sum()

    pref_time = out["pref_time_ms"].sum()
    pref_energy = out["pref_energy_mJ"].sum()

    group_time = out["group_time_ms"].sum()
    group_energy = out["group_energy_mJ"].sum()

    pref_saving = 100.0 * (1.0 - pref_energy / auto_energy)
    group_saving = 100.0 * (1.0 - group_energy / auto_energy)

    pref_runtime_change = 100.0 * (pref_time / auto_time - 1.0)
    group_runtime_change = 100.0 * (group_time / auto_time - 1.0)

    summary = {
        "kernels": len(out),
        "groups": out["group"].nunique(),
        "clock_changes_ungrouped": len(out) - 1,
        "clock_changes_grouped": out["group"].nunique() - 1,
        "kernels_using_preferred_pair": int(out["uses_preferred_pair"].sum()),
        "kernels_forced_to_group_pair": int((~out["uses_preferred_pair"]).sum()),

        "auto_time_ms": auto_time,
        "auto_energy_mJ": auto_energy,

        "ungrouped_time_ms": pref_time,
        "ungrouped_energy_mJ": pref_energy,
        "ungrouped_energy_saving_pct": pref_saving,
        "ungrouped_runtime_change_pct": pref_runtime_change,

        "grouped_time_ms": group_time,
        "grouped_energy_mJ": group_energy,
        "grouped_energy_saving_pct": group_saving,
        "grouped_runtime_change_pct": group_runtime_change,

        "grouping_loss_percentage_points": pref_saving - group_saving,
    }

    print("\nGrouping evaluation")
    print("-------------------")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    pd.DataFrame([summary]).to_csv(out_csv.with_name("grouping_summary.csv"), index=False)

    return summary


def infer_epsilon(constraint_name, explicit_epsilon=None):
    if explicit_epsilon is not None:
        return explicit_epsilon

    s = str(constraint_name).strip().lower()

    if s.endswith("pct"):
        return float(s.replace("pct", "")) / 100.0

    if s.endswith("%"):
        return float(s.replace("%", "")) / 100.0

    return float(s)


def plot_dependencies(traces, out_pdf: Path):
    max_len = max(len(w) for w in traces)
    fig_w = max(7.0, 1.15 * max_len + 2.0)
    fig_h = max(3.5, 0.9 * len(traces) + 0.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")

    x0 = 1.25
    dx = 1.15
    radius = 0.23

    for row_idx, chain in enumerate(traces):
        y = len(traces) - row_idx
        workload_idx = chain[0]["workload_idx"]
        color = WORKLOAD_COLORS[(workload_idx - 1) % len(WORKLOAD_COLORS)]

        ax.text(
            0.25,
            y,
            f"W{workload_idx}",
            fontsize=11,
            va="center",
            ha="center",
        )

        for i, k in enumerate(chain):
            x = x0 + i * dx

            circ = Circle(
                (x, y),
                radius,
                facecolor=color,
                edgecolor="black",
                alpha=0.85,
                linewidth=1.4,
            )
            ax.add_patch(circ)

            ax.text(
                x,
                y,
                str(k["node"]),
                ha="center",
                va="center",
                fontsize=9,
            )

            if i < len(chain) - 1:
                x_next = x0 + (i + 1) * dx
                arrow = FancyArrowPatch(
                    (x + radius, y),
                    (x_next - radius, y),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.2,
                    color="black",
                )
                ax.add_patch(arrow)

    ax.set_xlim(-0.15, x0 + max_len * dx)
    ax.set_ylim(0.4, len(traces) + 0.6)

    fig.tight_layout(pad=0.15)
    savefig_both(fig, out_pdf)
    plt.close(fig)


def build_timeline(groups):
    timeline = []
    t = 0.0

    for g in groups:
        group_start = t
        kt = group_start
        kernels = []

        for k in g["kernels"]:
            k_start = kt
            k_end = kt + k["auto_time_ms"]
            kernels.append((k_start, k_end, k))
            kt = k_end

        group_end = kt
        timeline.append((group_start, group_end, g, kernels))
        t = group_end

    return timeline, t


def plot_schedule_overview_with_zooms(
    groups,
    out_pdf: Path,
    zoom_windows,
):
    """
    Paper figure:
      - top axis: full schedule overview in grey
      - lower axes: two expanded regions with kernel IDs and clock-pair labels
    """
    timeline, total_time = build_timeline(groups)

    n_zooms = len(zoom_windows)

    height_ratios = [0.55] + [1.2] * n_zooms

    fig, axes = plt.subplots(
        1 + n_zooms,
        1,
        figsize=(6.5, 0.9 + 1.6 * n_zooms),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.7},
    )
    if n_zooms == 0:
        axes = [axes]

    ax_full = axes[0]
    panels = [(axes[i + 1], window, "below") for i, window in enumerate(zoom_windows)]

    y = 0.0
    h = 1.0

    grey_shades = ["0.68", "0.84"]
    for gi, (group_start, group_end, g, kernels) in enumerate(timeline):
        ax_full.add_patch(
            Rectangle(
                (group_start, y),
                max(group_end - group_start, 1e-9),
                h,
                facecolor=grey_shades[gi % len(grey_shades)],
                edgecolor="0.35",
                linewidth=0.35,
            )
        )

    ax_full.set_xlim(0, total_time)
    ax_full.set_ylim(-0.08, 1.08)
    ax_full.set_yticks([])
    ax_full.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax_full.tick_params(axis="x", labelsize=9)

    for spine in ("top", "left", "right"):
        ax_full.spines[spine].set_visible(False)

    # ---------- zoom panels ----------
    for ax_zoom, (zoom_start_ms, zoom_end_ms), placement in panels:
        vis = [
            (k_start, k_end, k)
            for _, _, _, kernels in timeline
            for (k_start, k_end, k) in kernels
            if k_end > zoom_start_ms and k_start < zoom_end_ms
        ]

        if vis:
            draw_start = max(zoom_start_ms, min(v[0] for v in vis))
            draw_end = min(zoom_end_ms, max(v[1] for v in vis))
        else:
            draw_start, draw_end = zoom_start_ms, zoom_end_ms

        span = max(draw_end - draw_start, 1e-9)

        vis_duration = {}
        for k_start, k_end, k in vis:
            d = min(k_end, draw_end) - max(k_start, draw_start)
            if d > 0:
                vis_duration[k["workload_idx"]] = vis_duration.get(k["workload_idx"], 0.0) + d
        panel_workloads = sorted(vis_duration, key=lambda wi: vis_duration[wi], reverse=True)
        palette = CORE_GREENS + BRIGHT_COLORS
        color_of = {wi: palette[i % len(palette)] for i, wi in enumerate(panel_workloads)}

        # Highlight zoom window on full overview.
        ax_full.add_patch(
            Rectangle(
                (draw_start, 0.0),
                draw_end - draw_start,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
            )
        )

        if placement == "above":
            y_full, y_zoom = 1.0, -0.05
        else:
            y_full, y_zoom = 0.0, 1.0

        con_left = ConnectionPatch(
            xyA=(draw_start, y_full),
            coordsA=ax_full.transData,
            xyB=(draw_start, y_zoom),
            coordsB=ax_zoom.transData,
            color="0.45",
            linewidth=0.75,
            alpha=0.5,
        )
        con_right = ConnectionPatch(
            xyA=(draw_end, y_full),
            coordsA=ax_full.transData,
            xyB=(draw_end, y_zoom),
            coordsB=ax_zoom.transData,
            color="0.45",
            linewidth=0.75,
            alpha=0.5,
        )
        fig.add_artist(con_left)
        fig.add_artist(con_right)

        y = 0.0
        h = 1.0
        visible_workloads = set()


        axis_y = y + h + 0.08
        tick_len = 0.10
        axis_lw = 0.8

        # Draw group boundaries, kernels, and group labels.
        for group_start, group_end, g, kernels in timeline:
            if group_end <= draw_start or group_start >= draw_end:
                continue

            # Frequency-switch tick on the switch axis line (outward, like axis ticks).
            if draw_start <= group_start < draw_end:
                ax_zoom.plot(
                    [group_start, group_start],
                    [axis_y, axis_y + tick_len],
                    linewidth=axis_lw,
                    color="black",
                )

            for k_start, k_end, k in kernels:
                if k_end <= draw_start or k_start >= draw_end:
                    continue

                cs = max(k_start, draw_start)
                ce = min(k_end, draw_end)
                width = max(ce - cs, span * 0.0005)

                color = color_of[k["workload_idx"]]
                visible_workloads.add(k["workload_idx"])

                ax_zoom.add_patch(
                    Rectangle(
                        (cs, y),
                        width,
                        h,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=0.65,
                        alpha=0.85,
                    )
                )

            # Clock-pair label for every group, angled so narrow groups still fit.
            vs = max(group_start, draw_start)
            ve = min(group_end, draw_end)
            visible_width = ve - vs

            if visible_width > span * 0.10:
                ax_zoom.text(
                    vs + visible_width / 2,
                    axis_y + tick_len + 0.04,
                    g["clock_label"],
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax_zoom.add_patch(
            Rectangle(
                (draw_start, y),
                span,
                h,
                fill=False,
                edgecolor="black",
                linewidth=1.2,
            )
        )

        # Frequency-switch axis: horizontal line with a closing tick at the right edge.
        ax_zoom.plot(
            [draw_start, draw_end],
            [axis_y, axis_y],
            linewidth=axis_lw,
            color="black",
        )
        ax_zoom.plot(
            [draw_end, draw_end],
            [axis_y, axis_y + tick_len],
            linewidth=axis_lw,
            color="black",
        )

        # Axis label for the frequency-switch row (mirrors the bottom time axis).
        ax_zoom.text(
            (draw_start + draw_end) / 2,
            1.7,
            "core/memory clock pair (MHz)",
            ha="center",
            va="center",
            fontsize=10,
        )

        ax_zoom.set_xlim(draw_start, draw_end)
        ax_zoom.set_ylim(-0.08, 2.0)
        ax_zoom.set_yticks([])
        ax_zoom.set_xlabel("Time (ms)", labelpad=2, fontsize=10)
        ax_zoom.tick_params(axis="x", labelsize=9)
        ax_zoom.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))

        for spine in ("top", "left", "right"):
            ax_zoom.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.045, right=0.99, top=0.96, bottom=0.08)

    savefig_both(fig, out_pdf)
    plt.close(fig)


def save_schedule_csv(groups, out_csv: Path):
    rows = []
    t = 0.0

    for g in groups:
        group_start = t
        group_end = t + g["runtime_ms"]

        kt = group_start

        for k in g["kernels"]:
            kernel_start = kt
            kernel_end = kt + k["auto_time_ms"]
            kt = kernel_end

            rows.append({
                "group": g["group"],
                "group_start_ms": group_start,
                "group_end_ms": group_end,
                "group_runtime_ms": g["runtime_ms"],
                "group_selected_time_ms": g.get("selected_group_time_ms"),
                "group_selected_energy_mJ": g.get("selected_group_energy_mJ"),
                "clock_label": g["clock_label"],
                "clock_key": g["clock_key"],
                "node": k["node"],
                "workload_idx": k["workload_idx"],
                "workload": k["workload"],
                "kernel": k["kernel"],
                "kernel_id": k["kernel_id"],
                "kernel_start_ms": kernel_start,
                "kernel_end_ms": kernel_end,
                "kernel_auto_time_ms": k["auto_time_ms"],
                "kernel_auto_energy_mJ": k["auto_energy_mJ"],
                "kernel_pref_core": k["pref_core"],
                "kernel_pref_mem": k["pref_mem"],
                "kernel_pref_pair": k["pref_pair"],
            })

        # Non-blocking switching means no added time gap.
        t = group_end

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def find_zoom_windows(schedule_csv: Path, window_ms=20.0, n_windows=3):
    """
    Pick zoom windows where grouping behavior is visible.

    Prefer windows with:
      - many visible kernels,
      - several group boundaries,
      - multiple workloads,
      - no single kernel dominating the window.
    """
    df = pd.read_csv(schedule_csv)

    if len(df) == 0:
        return [(0.0, window_ms)]

    starts = df["kernel_start_ms"].astype(float)
    ends = df["kernel_end_ms"].astype(float)

    candidates = sorted(set(starts.tolist() + ends.tolist()))
    scored = []

    for s in candidates:
        e = s + window_ms

        overlap = (starts < e) & (ends > s)
        if overlap.sum() == 0:
            continue

        visible = df.loc[overlap].copy()

        # Actual visible duration inside the window, not full kernel duration.
        visible["visible_ms"] = (
            visible["kernel_end_ms"].clip(upper=e)
            - visible["kernel_start_ms"].clip(lower=s)
        ).clip(lower=0)

        total_visible = visible["visible_ms"].sum()
        if total_visible <= 0:
            continue

        longest_share = visible["visible_ms"].max() / total_visible
        kernel_count = len(visible)
        group_count = visible["group"].nunique()
        workload_count = visible["workload_idx"].nunique()

        # Reject windows that are basically one huge kernel.
        if longest_share > 0.65:
            continue

        score = (
            2.0 * kernel_count
            + 1.5 * group_count
            + 1.0 * workload_count
            - 8.0 * longest_share
        )

        scored.append((score, s, e, kernel_count, group_count, workload_count, longest_share))

    scored = sorted(scored, reverse=True)

    windows = []
    for _, s, e, _, _, _, _ in scored:
        overlaps_existing = any(not (e <= ws or s >= we) for ws, we in windows)
        if overlaps_existing:
            continue

        windows.append((s, e))

        if len(windows) >= n_windows:
            break

    if not windows:
        windows = [(0.0, window_ms)]

    return sorted(windows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--summary-csv",
        default=str(Path.home() / "thesis/figures/fig_kernel_time_energy_ranges_data.csv"),
    )
    ap.add_argument(
        "--selection-csv",
        default=str(
            Path.home()
            / "thesis/results/master/ptx_launch_runtime_normalized_models/measured_global_oracle_selections.csv"
        ),
    )
    ap.add_argument(
        "--measured-csv",
        default=str(Path.home() / "thesis/results/master/aggregated_all_kernels_effective.csv"),
        help="Full measured kernel-clock CSV with effective clocks, runtime, and energy.",
    )
    ap.add_argument("--out-dir", default=str(Path.home() / "thesis/figures/grouping"))

    ap.add_argument("--constraint", default="0pct")
    ap.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Slowdown bound as decimal. If omitted, inferred from --constraint, e.g. 0pct -> 0.0.",
    )
    ap.add_argument("--feature-set", default=None)
    ap.add_argument("--model", default=None)

    ap.add_argument("--switch-ms", type=float, default=1.0)
    ap.add_argument("--n-workloads", type=int, default=6)
    ap.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="0 = use all kernels. Otherwise cap to the N longest kernels.",
    )
    ap.add_argument(
        "--viz-max-kernel-ms",
        type=float,
        default=0.0,
        help="0 = no filter (all kernels). Otherwise exclude kernels longer than this (ms).",
    )
    ap.add_argument("--zoom-window-ms", type=float, default=20.0)
    ap.add_argument("--n-zoom-windows", type=int, default=3)

    ap.add_argument(
        "--trace-mode",
        choices=["real", "constructed"],
        default="real",
        help="real = benchmark workloads as dependency chains; constructed = synthetic chains from measured kernels.",
    )

    ap.add_argument("--dependency-max-workloads", type=int, default=12)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(
        args.summary_csv,
        args.selection_csv,
        constraint=args.constraint,
        feature_set=args.feature_set,
        model=args.model,
    )

    if args.trace_mode == "real":
        traces = make_real_workload_traces(df)
    else:
        traces = make_constructed_traces_all_kernels(
            df,
            max_nodes=(args.max_nodes if args.max_nodes > 0 else None),
            max_kernel_ms=(args.viz_max_kernel_ms if args.viz_max_kernel_ms > 0 else None),
        )

    groups_before_merge = schedule_ready_set(
        traces,
        switch_ms=args.switch_ms,
    )

    groups = merge_last_underfilled_group(
        groups_before_merge,
        switch_ms=args.switch_ms,
    )

    epsilon = infer_epsilon(args.constraint, args.epsilon)

    meas = load_measured_clock_table(
        Path(args.measured_csv),
        id_map=df[["workload", "kernel", "kernel_id", "auto_time_ms", "auto_energy_mJ"]].drop_duplicates(),
    )

    groups = select_group_clocks_global_from_measured(
        groups,
        meas,
        epsilon=epsilon,
    )

    schedule_csv = out_dir / "grouping_schedule.csv"
    save_schedule_csv(groups, schedule_csv)

    summary = evaluate_grouping(
        groups,
        meas,
        out_dir / "grouping_evaluation.csv",
    )

    dependency_traces = [t for t in traces if len(t) >= 2][:args.dependency_max_workloads]
    plot_dependencies(dependency_traces, out_dir / "fig_grouping_dependencies.pdf")

    zoom_windows = find_zoom_windows(
        schedule_csv,
        window_ms=args.zoom_window_ms,
        n_windows=1,
    )

    plot_schedule_overview_with_zooms(
        groups,
        out_dir / "fig_grouping_schedule_overview_zooms.pdf",
        zoom_windows=zoom_windows,
    )

    print(f"Loaded kernels after selection merge: {len(df)}")
    print(f"Constructed traces: {len(traces)}")
    print(f"Kernels in constructed trace: {sum(len(t) for t in traces)}")
    print(f"Groups before merging last short group: {len(groups_before_merge)}")
    print(f"Groups after merging last short group: {len(groups)}")
    print(f"Clock changes after grouping: {max(len(groups) - 1, 0)}")
    print(f"Wrote {out_dir / 'fig_grouping_dependencies.pdf'} (+ .png)")
    for i, (zs, ze) in enumerate(zoom_windows, start=1):
        print(f"Wrote zoom {i}: {zs:.3f}--{ze:.3f} ms")
    print(f"Wrote {out_dir / 'fig_grouping_schedule_overview_zooms.pdf'}")
    print(f"Wrote {schedule_csv}")


if __name__ == "__main__":
    main()