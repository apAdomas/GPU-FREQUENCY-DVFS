#!/usr/bin/env python3
from pathlib import Path
import subprocess

POLY_ROOT = Path.home() / "polybenchGpu" / "CUDA"
RODINIA_ROOT = Path.home() / "rodinia_3.1" / "cuda"

ROOTS = [
    ("polybench", POLY_ROOT),
    ("rodinia", RODINIA_ROOT),
]


def is_polybench_benchmark_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    name = p.name
    if name.startswith("_") or name.startswith("."):
        return False
    if name in {"scripts", "common"}:
        return False
    return any(c.isupper() for c in name)


def is_isolated_kernel_file(p: Path) -> bool:
    return (
        p.is_file()
        and p.suffix == ".cu"
        and "_repeat" in p.stem
        and p.name[:2].isdigit()
    )


def polybench_files(root: Path):
    for bench_dir in sorted(root.iterdir()):
        if not is_polybench_benchmark_dir(bench_dir):
            continue
        for cu in sorted(bench_dir.iterdir()):
            if is_isolated_kernel_file(cu):
                yield cu


def rodinia_files(root: Path):
    for cu in sorted(root.rglob("*_repeat.cu")):
        if is_isolated_kernel_file(cu):
            yield cu


def source_files():
    for kind, root in ROOTS:
        if not root.exists():
            print(f"SKIP missing root: {root}")
            continue

        if kind == "polybench":
            yield from polybench_files(root)
        elif kind == "rodinia":
            yield from rodinia_files(root)


def run(cmd, cwd: Path):
    print(" ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main():
    generated = 0
    skipped = 0
    failed = 0

    files = list(source_files())

    print(f"Found isolated CUDA files: {len(files)}")

    for cu in files:
        out = cu.with_suffix(".ptx")

        rel = cu
        try:
            if str(cu).startswith(str(POLY_ROOT)):
                rel = cu.relative_to(POLY_ROOT)
            elif str(cu).startswith(str(RODINIA_ROOT)):
                rel = cu.relative_to(RODINIA_ROOT)
        except Exception:
            pass

        print(f"\n=== {rel} ===")

        if out.exists() and out.stat().st_mtime >= cu.stat().st_mtime:
            print(f"SKIP up-to-date: {out.name}")
            skipped += 1
            continue

        cmd = [
            "nvcc",
            "-ptx",
            "-arch=sm_86",
            "-I", str(POLY_ROOT / "scripts"),
            "-I", str(POLY_ROOT / "common"),
            "-I", str(RODINIA_ROOT),
            "-I", str(cu.parent),
            cu.name,
            "-o",
            out.name,
        ]

        res = run(cmd, cu.parent)

        if res.returncode != 0:
            failed += 1
            print(f"FAILED: {cu}")
            print(res.stderr[-4000:])
            continue

        generated += 1
        print(f"WROTE: {out}")

    print("\nDone.")
    print(f"Generated: {generated}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
