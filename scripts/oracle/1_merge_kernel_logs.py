from pathlib import Path
import csv
import re
import sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 merge_kernel_logs.py <results_dir>")

base = Path(sys.argv[1]).expanduser()
if not base.exists():
    raise SystemExit(f"Results dir does not exist: {base}")

runs = sorted([p for p in base.iterdir() if p.is_dir()])
if not runs:
    raise SystemExit("No run folders found")

latest = runs[-1]
out_csv = latest / "merged_summary.csv"

rows = []

def parse_applied_clocks(s: str):
    m = re.match(r"\s*(\d+)\s*MHz,\s*(\d+)\s*MHz\s*$", s)
    if not m:
        return "", ""
    return m.group(1), m.group(2)

for kernel_dir in sorted(latest.iterdir()):
    if not kernel_dir.is_dir():
        continue

    for run_dir in sorted(kernel_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        summary = run_dir / "summary.log"
        stdout = run_dir / "program_stdout.txt"

        if not summary.exists():
            continue

        data = {}
        for line in summary.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()

        data["kernel_dir"] = kernel_dir.name
        data["run_dir"] = run_dir.name

        if base.name == "jacobi1d_kernel_energy":
            if data.get("kernel") == "step1":
                data["kernel"] = "kernel1"
            elif data.get("kernel") == "step2":
                data["kernel"] = "kernel2"

            if data.get("kernel_dir") == "step1":
                data["kernel_dir"] = "kernel1"
            elif data.get("kernel_dir") == "step2":
                data["kernel_dir"] = "kernel2"

        applied_core, applied_mem = parse_applied_clocks(data.get("applied_clocks", ""))
        data["applied_core_clock"] = applied_core
        data["applied_mem_clock"] = applied_mem

        if stdout.exists():
            data["program_stdout"] = stdout.read_text().strip().replace("\n", " | ")
        else:
            data["program_stdout"] = ""

        rows.append(data)

if not rows:
    raise SystemExit("No summary.log files found")

fieldnames = sorted({k for row in rows for k in row.keys()})

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {out_csv}")
print(f"Rows: {len(rows)}")
print(f"Latest run folder: {latest}")
