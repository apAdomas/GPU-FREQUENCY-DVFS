#!/usr/bin/env python3
from pathlib import Path
import csv, math, sys
from collections import defaultdict

if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 2_aggregate_kernel_logs_clean.py <results_dir>")

base = Path(sys.argv[1]).expanduser()
runs = sorted([p for p in base.iterdir() if p.is_dir()])
if not runs:
    raise SystemExit(f"No run folders found in {base}")
latest = runs[-1]
in_csv = latest / "merged_summary.csv"
out_csv = latest / "aggregated_by_kernel_clock.csv"
if not in_csv.exists():
    raise SystemExit(f"Missing {in_csv}")

def mean(xs): return sum(xs) / len(xs)
def stddev(xs):
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

def f(row, key): return float(str(row[key]).strip())
def i(row, key): return int(float(str(row[key]).strip()))

def is_auto_pair(rc, rm):
    return str(rc).strip() == "auto" and str(rm).strip() == "auto"

groups = defaultdict(list)
with open(in_csv, newline="") as fh:
    for row in csv.DictReader(fh):
        kernel = row.get("kernel", "").strip()
        ac = row.get("applied_core_clock", "").strip()
        am = row.get("applied_mem_clock", "").strip()
        rc = row.get("requested_core_clock", "").strip()
        rm = row.get("requested_mem_clock", "").strip()
        if not kernel or not ac or not am or not rc or not rm:
            continue
        try:
            measured_launches = i(row, "measured_launches")
            if measured_launches <= 0: continue
            total_time_ms = f(row, "measured_cuda_time_ms")
            total_energy_mj = f(row, "measured_energy_mj")
            avg_power_w = f(row, "average_power_w")
        except Exception:
            continue

        key = (kernel, rc, rm, int(float(ac)), int(float(am)))
        groups[key].append({
            "measured_launches": measured_launches,
            "total_time_ms": total_time_ms,
            "total_energy_mJ": total_energy_mj,
            "avg_power_W": avg_power_w,
            "time_per_launch_ms": total_time_ms / measured_launches,
            "energy_per_launch_mJ": total_energy_mj / measured_launches,
        })

rows = []
for (kernel, rc, rm, ac, am), vals in sorted(groups.items()):
    launch_vals = [v["measured_launches"] for v in vals]
    total_time_vals = [v["total_time_ms"] for v in vals]
    total_energy_vals = [v["total_energy_mJ"] for v in vals]
    power_vals = [v["avg_power_W"] for v in vals]
    tpl_vals = [v["time_per_launch_ms"] for v in vals]
    epl_vals = [v["energy_per_launch_mJ"] for v in vals]
    rows.append({
        "kernel": kernel,
        "requested_core_clock": rc,
        "requested_mem_clock": rm,
        "requested_pair": f"{rc}/{rm}",
        "is_auto_baseline": int(is_auto_pair(rc, rm)),
        "applied_core_clock": ac,
        "applied_mem_clock": am,
        "n": len(vals),
        "measured_launches_mean": mean(launch_vals),
        "measured_launches_std": stddev(launch_vals),
        "total_time_mean_ms": mean(total_time_vals),
        "total_time_std_ms": stddev(total_time_vals),
        "total_energy_mean_mJ": mean(total_energy_vals),
        "total_energy_std_mJ": stddev(total_energy_vals),
        "avg_power_mean_W": mean(power_vals),
        "avg_power_std_W": stddev(power_vals),
        "time_per_launch_mean_ms": mean(tpl_vals),
        "time_per_launch_std_ms": stddev(tpl_vals),
        "energy_per_launch_mean_mJ": mean(epl_vals),
        "energy_per_launch_std_mJ": stddev(epl_vals),
    })

if not rows:
    raise SystemExit("No aggregated rows produced")

fields = list(rows[0].keys())
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
print(f"Wrote {out_csv}")
print(f"Rows: {len(rows)}")
print(f"Latest run folder: {latest}")
