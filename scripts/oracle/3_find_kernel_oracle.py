#!/usr/bin/env python3
from pathlib import Path
import csv, sys
from collections import defaultdict

if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 3_find_kernel_oracle_auto.py <results_dir>")

base = Path(sys.argv[1]).expanduser()
runs = sorted([p for p in base.iterdir() if p.is_dir()])
if not runs:
    raise SystemExit(f"No run folders found in {base}")
latest = runs[-1]
in_csv = latest / "aggregated_by_kernel_clock.csv"
out_candidates = latest / "oracle_candidates_by_kernel.csv"
out_summary = latest / "oracle_summary_by_kernel.csv"
if not in_csv.exists():
    raise SystemExit(f"Missing {in_csv}")

NUMS = [
    "applied_core_clock", "applied_mem_clock", "n",
    "measured_launches_mean", "measured_launches_std",
    "total_time_mean_ms", "total_time_std_ms",
    "total_energy_mean_mJ", "total_energy_std_mJ",
    "avg_power_mean_W", "avg_power_std_W",
    "time_per_launch_mean_ms", "time_per_launch_std_ms",
    "energy_per_launch_mean_mJ", "energy_per_launch_std_mJ",
]

def to_num(row):
    r = dict(row)
    for k in NUMS:
        if k in r and r[k] != "":
            r[k] = float(r[k])
    r["applied_core_clock"] = int(r["applied_core_clock"])
    r["applied_mem_clock"] = int(r["applied_mem_clock"])
    r["n"] = int(r["n"])
    if "requested_pair" not in r:
        r["requested_pair"] = r.get("requested_pairs", "")
    if "is_auto_baseline" not in r:
        r["is_auto_baseline"] = int(r.get("requested_pair", "") == "auto/auto")
    else:
        r["is_auto_baseline"] = int(float(r["is_auto_baseline"]))
    return r

rows = []
with open(in_csv, newline="") as fh:
    for row in csv.DictReader(fh):
        rows.append(to_num(row))
if not rows:
    raise SystemExit("No rows in input")

by_kernel = defaultdict(list)
for r in rows:
    by_kernel[r["kernel"]].append(r)

candidate_rows = []
summary_rows = []
BOUNDS = [0, 1, 2, 5]

def add_summary(kernel, policy, chosen, baseline):
    time = chosen["time_per_launch_mean_ms"]
    energy = chosen["energy_per_launch_mean_mJ"]
    bt = baseline["time_per_launch_mean_ms"]
    be = baseline["energy_per_launch_mean_mJ"]
    out = dict(chosen)
    out.update({
        "kernel": kernel,
        "policy": policy,
        "baseline_time_per_launch_ms": bt,
        "baseline_energy_per_launch_mJ": be,
        "slowdown_pct_vs_baseline": ((time - bt) / bt) * 100.0 if bt else 0.0,
        "energy_saving_pct_vs_baseline": ((be - energy) / be) * 100.0 if be else 0.0,
    })
    summary_rows.append(out)

for kernel, kr in sorted(by_kernel.items()):
    auto_rows = [r for r in kr if int(r.get("is_auto_baseline", 0)) == 1]
    auto = min(auto_rows, key=lambda r: r["time_per_launch_mean_ms"]) if auto_rows else None

    fixed_kr = [
        r for r in kr
        if str(r["requested_core_clock"]) != "auto"
        and str(r["requested_mem_clock"]) != "auto"
    ]

    if not fixed_kr:
        continue

    fastest = min(fixed_kr, key=lambda r: r["time_per_launch_mean_ms"])
    search_space = fixed_kr

    if auto:
        search_space_with_auto = fixed_kr + [auto]
    else:
        search_space_with_auto = fixed_kr

    for r in kr:
        c = dict(r)
        c["slowdown_pct_vs_fastest"] = ((r["time_per_launch_mean_ms"] - fastest["time_per_launch_mean_ms"]) / fastest["time_per_launch_mean_ms"]) * 100.0
        c["energy_saving_pct_vs_fastest"] = ((fastest["energy_per_launch_mean_mJ"] - r["energy_per_launch_mean_mJ"]) / fastest["energy_per_launch_mean_mJ"]) * 100.0
        if auto:
            c["slowdown_pct_vs_auto"] = ((r["time_per_launch_mean_ms"] - auto["time_per_launch_mean_ms"]) / auto["time_per_launch_mean_ms"]) * 100.0
            c["energy_saving_pct_vs_auto"] = ((auto["energy_per_launch_mean_mJ"] - r["energy_per_launch_mean_mJ"]) / auto["energy_per_launch_mean_mJ"]) * 100.0
        else:
            c["slowdown_pct_vs_auto"] = ""
            c["energy_saving_pct_vs_auto"] = ""
        candidate_rows.append(c)

    add_summary(kernel, "baseline_fastest", fastest, fastest)
    if auto:
        add_summary(kernel, "baseline_auto", auto, auto)

    for b in BOUNDS:
        fcands = [r for r in search_space if r["time_per_launch_mean_ms"] <= fastest["time_per_launch_mean_ms"] * (1 + b/100.0) + 1e-15]
        if fcands:
            add_summary(kernel, f"oracle_vs_fastest_le_{b}pct", min(fcands, key=lambda r: r["energy_per_launch_mean_mJ"]), fastest)

        if auto:
            acands = [
                r for r in search_space_with_auto
                if r["time_per_launch_mean_ms"] <= auto["time_per_launch_mean_ms"] * (1 + b/100.0) + 1e-15
            ]
            if acands:
                add_summary(
                    kernel,
                    f"oracle_vs_auto_le_{b}pct",
                    min(acands, key=lambda r: r["energy_per_launch_mean_mJ"]),
                    auto,
                )

    min_energy = min(search_space, key=lambda r: r["energy_per_launch_mean_mJ"])
    add_summary(kernel, "min_energy_overall_vs_fastest", min_energy, fastest)
    if auto:
        add_summary(kernel, "min_energy_overall_vs_auto", min_energy, auto)

base_fields = [
    "kernel", "policy", "requested_core_clock", "requested_mem_clock", "requested_pair", "is_auto_baseline",
    "applied_core_clock", "applied_mem_clock", "n",
    "measured_launches_mean", "measured_launches_std",
    "total_time_mean_ms", "total_time_std_ms", "total_energy_mean_mJ", "total_energy_std_mJ",
    "avg_power_mean_W", "avg_power_std_W",
    "time_per_launch_mean_ms", "time_per_launch_std_ms", "energy_per_launch_mean_mJ", "energy_per_launch_std_mJ",
    "baseline_time_per_launch_ms", "baseline_energy_per_launch_mJ",
    "slowdown_pct_vs_baseline", "energy_saving_pct_vs_baseline",
]
def unique_fields(rows, preferred=None):
    fields = []
    if preferred:
        for k in preferred:
            if k not in fields:
                fields.append(k)
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    return fields

all_sum_fields = unique_fields(summary_rows, base_fields)
all_cand_fields = unique_fields(candidate_rows)

def write(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

write(out_candidates, candidate_rows, all_cand_fields)
write(out_summary, summary_rows, all_sum_fields)
print(f"Wrote {out_candidates}")
print(f"Wrote {out_summary}")
print(f"Latest run folder: {latest}")
