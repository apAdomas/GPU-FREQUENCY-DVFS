#!/usr/bin/env python3
from pathlib import Path
import csv
import re
import math

POLY_ROOT = Path.home() / "polybenchGpu" / "CUDA"
RODINIA_ROOT = Path.home() / "rodinia_3.1" / "cuda"
OUT_CSV = Path.home() / "thesis/results/launch_features_all.csv"


def is_polybench_benchmark_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    n = p.name
    if n.startswith("_") or n.startswith("."):
        return False
    if n in {"scripts", "common"}:
        return False
    return any(c.isupper() for c in n)


def is_isolated_kernel_file(p: Path) -> bool:
    return (
        p.is_file()
        and p.suffix == ".cu"
        and "_repeat" in p.stem
        and p.name[:2].isdigit()
    )


def normalize_workload(folder_name: str) -> str:
    return str(folder_name).lower().replace("-", "").replace("_", "")


def rodinia_source_folder(cu: Path) -> str:
    stem = cu.stem

    if "_pre_euler3d_" in stem:
        return "pre_euler3d"
    if "_euler3d_" in stem:
        return "euler3d"

    parts = cu.relative_to(RODINIA_ROOT).parts

    if "srad_v1" in parts:
        return "srad_v1"

    parent = cu.parent.name
    aliases = {
        "hotspot3D": "hotspot3d",
        "lavaMD": "lavamd",
    }
    return aliases.get(parent, parent)


def source_files():
    # PolyBench one-level benchmark folders.
    for bench in sorted(POLY_ROOT.iterdir()):
        if not is_polybench_benchmark_dir(bench):
            continue
        for cu in sorted(bench.iterdir()):
            if is_isolated_kernel_file(cu):
                yield bench.name, normalize_workload(bench.name), cu

    # Rodinia recursive.
    for cu in sorted(RODINIA_ROOT.rglob("*_repeat.cu")):
        if not is_isolated_kernel_file(cu):
            continue
        source_folder = rodinia_source_folder(cu)
        yield source_folder, normalize_workload(source_folder), cu


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def read_with_local_includes(cu: Path) -> str:
    chunks = []
    seen = set()

    def add_file(p: Path):
        p = p.resolve()
        if p in seen:
            return
        if not p.exists() or not p.is_file():
            return
        seen.add(p)
        try:
            chunks.append(f"\n\n/* FILE: {p} */\n")
            chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass

    add_file(cu)

    raw = cu.read_text(encoding="utf-8", errors="ignore")

    # Direct quoted includes.
    for m in re.finditer(r'#include\s+"([^"]+)"', raw):
        inc = m.group(1)
        candidates = [
            cu.parent / inc,
            POLY_ROOT / "scripts" / inc,
            POLY_ROOT / "common" / inc,
            RODINIA_ROOT / inc,
        ]
        for inc_path in candidates:
            add_file(inc_path)

    # Local headers and local original benchmark files.
    for p in sorted(cu.parent.glob("*.h")):
        add_file(p)
    for p in sorted(cu.parent.glob("*.hpp")):
        add_file(p)
    for p in sorted(cu.parent.glob("*.cuh")):
        add_file(p)

    for p in sorted(cu.parent.glob("*.cu")):
        if "_repeat" not in p.stem:
            add_file(p)

    return "\n".join(chunks)


def split_args(argstr: str):
    args = []
    cur = []
    depth = 0

    for ch in argstr:
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue

        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1

        cur.append(ch)

    if cur:
        args.append("".join(cur).strip())

    return args


def clean_expr(expr: str) -> str:
    expr = expr.strip().replace(";", "")

    cast_types = [
        "size_t",
        "int",
        "unsigned int",
        "unsigned",
        "long",
        "unsigned long",
        "float",
        "double",
    ]

    for t in cast_types:
        expr = re.sub(rf"\(\s*{re.escape(t)}\s*\)", "", expr)

    expr = re.sub(r"\s+", " ", expr)
    return expr


def replace_constants(expr: str, constants: dict) -> str:
    for k, v in sorted(constants.items(), key=lambda x: -len(x[0])):
        # Dotted names like block.x do not behave nicely with pure word-boundary regex.
        if "." in k:
            expr = expr.replace(k, str(v))
        else:
            expr = re.sub(rf"\b{re.escape(k)}\b", str(v), expr)
    return expr


def safe_eval_int(expr: str):
    allowed = {
        "__builtins__": {},
        "ceil": math.ceil,
        "floor": math.floor,
        "sqrt": math.sqrt,
    }

    expr = expr.strip()

    # Remove harmless C/CUDA numeric suffixes.
    expr = re.sub(r"(?<=\d)[uUlLfF]+\b", "", expr)

    tmp = expr
    for fn in ["ceil", "floor", "sqrt"]:
        tmp = tmp.replace(fn, "")

    if re.search(r"[A-Za-z_]", tmp):
        return None

    # After constants are replaced, only arithmetic and the allowed math calls remain.
    if re.search(r"[^0-9+\-*/%<>&|()., \tceilforsq]", expr):
        return None

    try:
        return int(eval(expr, allowed, {}))
    except Exception:
        return None


def parse_constant_expr(expr: str, constants: dict) -> int | None:
    expr = clean_expr(expr)
    expr = re.sub(r"POLYBENCH_LOOP_BOUND\s*\(([^,]+),([^)]+)\)", r"\1", expr)
    expr = replace_constants(expr, constants)
    return safe_eval_int(expr)


def parse_int_expr(expr: str, constants: dict) -> int | None:
    expr = clean_expr(expr)
    expr = replace_constants(expr, constants)
    return safe_eval_int(expr)


def extract_constants(text: str) -> dict:
    constants = {}
    raw_defs = {}

    define_pat = re.compile(
        r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$",
        flags=re.M,
    )

    for m in define_pat.finditer(text):
        name = m.group(1)
        value = m.group(2).strip()

        if "(" in name:
            continue

        value = re.sub(r"//.*$", "", value).strip()
        value = re.sub(r"/\*.*?\*/", "", value).strip()

        raw_defs[name] = value

    for pat in [
        r"const\s+int\s+([A-Za-z_]\w*)\s*=\s*([^;]+)\s*;",
        r"int\s+([A-Za-z_]\w*)\s*=\s*([^;]+)\s*;",
        r"unsigned\s+int\s+([A-Za-z_]\w*)\s*=\s*([^;]+)\s*;",
        r"long\s+([A-Za-z_]\w*)\s*=\s*([^;]+)\s*;",
        r"unsigned\s+long\s+([A-Za-z_]\w*)\s*=\s*([^;]+)\s*;",
        r"size_t\s+([A-Za-z_]\w*)\s*=\s*([^;]+)\s*;",
    ]:
        for m in re.finditer(pat, text):
            raw_defs[m.group(1)] = m.group(2).strip()

    changed = True
    while changed:
        changed = False

        for name, expr in list(raw_defs.items()):
            if name in constants:
                continue

            val = parse_constant_expr(expr, constants)
            if val is not None:
                constants[name] = val
                changed = True

    return constants


def parse_dim3_decls(text: str, constants: dict) -> tuple[dict, dict]:
    dims = {}
    dim_exprs = {}

    # Constructor form: dim3 block(256, 1, 1);
    pat = re.compile(r"dim3\s+(\w+)\s*\((.*?)\)\s*;", flags=re.S)

    for m in pat.finditer(text):
        name = m.group(1)
        args = split_args(m.group(2))
        dim_exprs[name] = args

        vals = []
        for i in range(3):
            if i < len(args):
                vals.append(parse_int_expr(args[i], constants))
            else:
                vals.append(1)

        dims[name] = tuple(vals)

    # Empty declaration form: dim3 block;
    empty_pat = re.compile(r"dim3\s+(\w+)\s*;", flags=re.S)
    for m in empty_pat.finditer(text):
        name = m.group(1)
        if name not in dims:
            dims[name] = (None, None, None)
            dim_exprs[name] = []

    # Assignment form:
    # block.x = 256;
    # block.y = 1;
    # block.z = 1;
    assign_pat = re.compile(r"(\w+)\s*\.\s*([xyz])\s*=\s*([^;]+)\s*;")
    idx = {"x": 0, "y": 1, "z": 2}

    for m in assign_pat.finditer(text):
        name = m.group(1)
        axis = m.group(2)
        expr = m.group(3)

        if name not in dims:
            continue

        vals = list(dims[name])
        val = parse_int_expr(expr, constants)
        vals[idx[axis]] = val
        dims[name] = tuple(vals)

    # Fill omitted y/z as 1 when x is known, then expose dim.x/y/z constants.
    for name, vals in list(dims.items()):
        vals = list(vals)
        if vals[0] is not None:
            if vals[1] is None:
                vals[1] = 1
            if vals[2] is None:
                vals[2] = 1
        dims[name] = tuple(vals)

        labels = ["x", "y", "z"]
        for label, val in zip(labels, dims[name]):
            if val is not None:
                constants[f"{name}.{label}"] = val

    # Second pass: constructor expressions like grid(ceil(N/block.x), ...)
    # may depend on block.x/block.y, which only become constants after
    # the first pass.
    for name, args in list(dim_exprs.items()):
        if not args:
            continue

        vals = []
        for i in range(3):
            if i < len(args):
                vals.append(parse_int_expr(args[i], constants))
            else:
                vals.append(1)

        old_vals = dims.get(name, (None, None, None))
        vals = [
            vals[i] if vals[i] is not None else old_vals[i]
            for i in range(3)
        ]

        if vals[0] is not None:
            if vals[1] is None:
                vals[1] = 1
            if vals[2] is None:
                vals[2] = 1

        dims[name] = tuple(vals)

        labels = ["x", "y", "z"]
        for label, val in zip(labels, dims[name]):
            if val is not None:
                constants[f"{name}.{label}"] = val

    return dims, dim_exprs


def find_launches(text: str):
    launches = []
    pat = re.compile(r"(\w+)\s*<<<\s*(.*?)\s*>>>\s*\(", flags=re.S)

    for m in pat.finditer(text):
        kernel_func = m.group(1)
        config = split_args(m.group(2))
        launches.append((kernel_func, config))

    return launches


def dim_from_config_arg(arg: str, dims: dict, constants: dict):
    arg = arg.strip()

    if arg in dims:
        return dims[arg]

    m = re.match(r"dim3\s*\((.*?)\)", arg, flags=re.S)
    if m:
        args = split_args(m.group(1))
        vals = []
        for i in range(3):
            vals.append(parse_int_expr(args[i], constants) if i < len(args) else 1)
        return tuple(vals)

    val = parse_int_expr(arg, constants)
    if val is not None:
        return (val, 1, 1)

    return (None, None, None)


def prod(vals):
    if any(v is None for v in vals):
        return None
    out = 1
    for v in vals:
        out *= int(v)
    return out



def apply_manual_launch_fallbacks(rows):
    """
    Fill exact Rodinia launch dimensions from measurement stdout/config logs.
    This is only used where grid size is input-dependent and not statically
    recoverable from the isolated CUDA file alone.
    """

    manual = {
        # source_folder, source_cu:
        #   grid_x, grid_y, grid_z, block_x, block_y, block_z, kernel_func
        ("myocyte", "01_myocyte_solver_2_repeat.cu"):
            (8, 1, 1, 32, 1, 1, "solver_2"),

        ("nn", "01_nn_euclid_repeat.cu"):
            (39063, 1, 1, 256, 1, 1, "euclid"),

        ("lavamd", "01_lavamd_kernel_gpu_cuda_repeat.cu"):
            (1000, 1, 1, 128, 1, 1, "kernel_gpu_cuda"),

        ("nw", "01_nw_shared_1_repeat.cu"):
            (256, 1, 1, 16, 1, 1, "needle_cuda_shared_1"),

        ("nw", "02_nw_shared_2_repeat.cu"):
            (255, 1, 1, 16, 1, 1, "needle_cuda_shared_2"),

        ("particlefilter", "01_particlefilter_likelihood_repeat.cu"):
            (196, 1, 1, 128, 1, 1, "likelihood_kernel"),

        ("particlefilter", "02_particlefilter_sum_repeat.cu"):
            (196, 1, 1, 128, 1, 1, "sum_kernel"),

        ("particlefilter", "03_particlefilter_normalize_weights_repeat.cu"):
            (196, 1, 1, 128, 1, 1, "normalize_weights_kernel"),

        ("particlefilter", "04_particlefilter_find_index_repeat.cu"):
            (196, 1, 1, 128, 1, 1, "find_index_kernel"),

        ("pathfinder", "01_pathfinder_dynproc_repeat.cu"):
            (463, 1, 1, 256, 1, 1, "dynproc_kernel"),

        ("srad_v1", "01_srad_extract_repeat.cu"):
            (8192, 1, 1, 512, 1, 1, "extract"),

        ("srad_v1", "02_srad_prepare_repeat.cu"):
            (8192, 1, 1, 512, 1, 1, "prepare"),

        ("srad_v1", "03_srad_reduce_repeat.cu"):
            (8192, 1, 1, 512, 1, 1, "reduce"),

        ("srad_v1", "04_srad_srad_repeat.cu"):
            (8192, 1, 1, 512, 1, 1, "srad"),

        ("srad_v1", "05_srad_srad2_repeat.cu"):
            (8192, 1, 1, 512, 1, 1, "srad2"),

        ("srad_v1", "06_srad_compress_repeat.cu"):
            (8192, 1, 1, 512, 1, 1, "compress"),

        ("streamcluster", "01_streamcluster_compute_cost_repeat.cu"):
            (1954, 1, 1, 512, 1, 1, "kernel_compute_cost"),

        # From Hotspot config: 512 x 512 grid, block tile 16 x 16.
        ("hotspot", "01_hotspot_calculate_temp_repeat.cu"):
            (32, 32, 1, 16, 16, 1, "calculate_temp"),

        # From Hotspot3D config: 512 x 512 x 8 cells, WG size 64.
        # Treat as one-dimensional 64-thread CTAs over all cells.
        ("hotspot3d", "01_hotspot3d_hotspotOpt1_repeat.cu"):
            (32768, 1, 1, 64, 1, 1, "hotspotOpt1"),
    }

    for r in rows:
        key = (r["source_folder"], r["source_cu"])
        if key not in manual:
            continue

        gx, gy, gz, bx, by, bz, kernel_func = manual[key]

        r["kernel_func_launched"] = kernel_func

        r["grid_dim_x"] = gx
        r["grid_dim_y"] = gy
        r["grid_dim_z"] = gz
        r["block_dim_x"] = bx
        r["block_dim_y"] = by
        r["block_dim_z"] = bz

        ctas = gx * gy * gz
        threads_per_cta = bx * by * bz
        warps_per_cta = math.ceil(threads_per_cta / 32)

        r["ctas"] = ctas
        r["threads_per_cta"] = threads_per_cta
        r["warps_per_cta"] = warps_per_cta
        r["total_threads"] = ctas * threads_per_cta
        r["total_warps"] = ctas * warps_per_cta



def main():
    rows = []
    files = list(source_files())
    print(f"Found isolated CUDA files: {len(files)}")

    for source_folder, workload_norm, cu in files:
        raw_main = cu.read_text(encoding="utf-8", errors="ignore")
        text_main = strip_comments(raw_main)

        raw_constants = read_with_local_includes(cu)
        text_constants = strip_comments(raw_constants)

        constants = extract_constants(text_constants)

        dims, dim_exprs = parse_dim3_decls(text_main, constants)
        launches = find_launches(text_main)

        chosen = launches[0] if launches else (None, [])
        kernel_func, config = chosen

        grid = (None, None, None)
        block = (None, None, None)
        dynamic_smem_bytes = 0

        if len(config) >= 1:
            grid = dim_from_config_arg(config[0], dims, constants)
        if len(config) >= 2:
            block = dim_from_config_arg(config[1], dims, constants)
        if len(config) >= 3:
            dynamic_smem_bytes = parse_int_expr(config[2], constants)
            if dynamic_smem_bytes is None:
                dynamic_smem_bytes = 0

        bx, by, bz = block
        gx, gy, gz = grid

        # Manual fallback for Rodinia NN:
        # euclid<<<gridDim, threadsPerBlock>>> where threadsPerBlock defaults to 256.
        if source_folder == "nn" and cu.name == "01_nn_euclid_repeat.cu":
            if bx is None:
                bx = 256
            if by is None:
                by = 1
            if bz is None:
                bz = 1
            block = (bx, by, bz)

        threads_per_cta = prod([bx, by, bz])
        ctas = prod([gx, gy, gz])

        warps_per_cta = None
        if threads_per_cta:
            warps_per_cta = math.ceil(threads_per_cta / 32)

        total_threads = None
        if ctas is not None and threads_per_cta is not None:
            total_threads = ctas * threads_per_cta

        total_warps = None
        if ctas is not None and warps_per_cta is not None:
            total_warps = ctas * warps_per_cta

        rows.append({
            "source_folder": source_folder,
            "source_cu": cu.name,
            "workload": workload_norm,
            "kernel_func_launched": kernel_func or "",

            "grid_dim_x": gx if gx is not None else "",
            "grid_dim_y": gy if gy is not None else "",
            "grid_dim_z": gz if gz is not None else "",
            "block_dim_x": bx if bx is not None else "",
            "block_dim_y": by if by is not None else "",
            "block_dim_z": bz if bz is not None else "",

            "ctas": ctas if ctas is not None else "",
            "threads_per_cta": threads_per_cta if threads_per_cta is not None else "",
            "warps_per_cta": warps_per_cta if warps_per_cta is not None else "",
            "total_threads": total_threads if total_threads is not None else "",
            "total_warps": total_warps if total_warps is not None else "",
            "dynamic_smem_bytes": dynamic_smem_bytes,

            "num_launches_found": len(launches),
            "num_dim3_decls_found": len(dims),
            "num_constants_found": len(constants),
            "dim3_exprs": str(dim_exprs),
            "constants_keys": ",".join(sorted(constants.keys())),
        })

    if not rows:
        raise SystemExit("No launch rows found")

    apply_manual_launch_fallbacks(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT_CSV}")
    print(f"Rows: {len(rows)}")

    missing = [r for r in rows if r["threads_per_cta"] == "" or r["ctas"] == ""]
    print(f"Rows missing launch dimensions: {len(missing)}")

    if missing:
        print("Missing examples:")
        for r in missing[:50]:
            print(
                r["source_folder"],
                r["source_cu"],
                "launches:", r["num_launches_found"],
                "dim3:", r["num_dim3_decls_found"],
                "constants:", r["num_constants_found"],
            )


if __name__ == "__main__":
    main()
