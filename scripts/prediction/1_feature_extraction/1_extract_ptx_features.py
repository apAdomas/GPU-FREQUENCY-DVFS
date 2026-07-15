#!/usr/bin/env python3
from pathlib import Path
import csv
import re
from collections import Counter

POLY_ROOT = Path.home() / "polybenchGpu" / "CUDA"
RODINIA_ROOT = Path.home() / "rodinia_3.1" / "cuda"
OUT_CSV = Path.home() / "thesis/results/ptx_features_all.csv"

ENTRY_PAT = re.compile(r"\.visible\s+\.entry\s+([^(]+)\(")
COMMENT_PAT = re.compile(r"//.*$")
LABEL_PAT = re.compile(r"^\s*\$?[\w.$]+:\s*$")
DIRECTIVE_PAT = re.compile(r"^\s*\.")
INSTR_PAT = re.compile(r"^\s*(?:@!?%?[\w.]+\s+)?([a-zA-Z][\w.]*)")
REG_DECL_PAT = re.compile(r"\.reg\s+\.([a-zA-Z0-9]+)\s+%[a-zA-Z_][\w$]*<(\d+)>;")

DTYPE_SET = {
    "f16", "f32", "f64",
    "s8", "s16", "s32", "s64",
    "u8", "u16", "u32", "u64",
    "b8", "b16", "b32", "b64",
    "pred"
}

MEMSPACE_SET = {"global", "shared", "local", "param", "const", "reg"}

ARITH_OPS = {
    "add", "sub", "mul", "mad", "fma", "div", "rem",
    "sqrt", "rsqrt", "rcp", "abs", "neg", "min", "max"
}
INT_OPS = {
    "and", "or", "xor", "not", "shl", "shr", "cnot",
    "popc", "clz", "bfind", "brev"
}
CONTROL_OPS = {"bra", "call", "ret", "exit", "trap", "brkpt"}
PRED_OPS = {"setp", "selp", "slct", "testp"}
LOAD_OPS = {"ld", "ldu", "prefetch"}
STORE_OPS = {"st"}
MEMORY_OPS = LOAD_OPS | STORE_OPS | {"atom", "red", "suld", "sust", "sured"}


def is_polybench_benchmark_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    n = p.name
    if n.startswith("_") or n.startswith("."):
        return False
    if n in {"scripts", "common"}:
        return False
    return any(c.isupper() for c in n)


def is_isolated_ptx_file(p: Path) -> bool:
    return (
        p.is_file()
        and p.suffix == ".ptx"
        and "_repeat" in p.stem
        and p.name[:2].isdigit()
    )


def rodinia_source_folder(ptx: Path) -> str:
    stem = ptx.stem

    if "_pre_euler3d_" in stem:
        return "pre_euler3d"
    if "_euler3d_" in stem:
        return "euler3d"

    parts = ptx.relative_to(RODINIA_ROOT).parts

    if "srad_v1" in parts:
        return "srad_v1"

    parent = ptx.parent.name
    aliases = {
        "hotspot3D": "hotspot3d",
        "lavaMD": "lavamd",
    }
    return aliases.get(parent, parent)


def ptx_files():
    # PolyBench: one-level benchmark folders.
    for bench in sorted(POLY_ROOT.iterdir()):
        if not is_polybench_benchmark_dir(bench):
            continue
        for p in sorted(bench.glob("*_repeat.ptx")):
            if is_isolated_ptx_file(p):
                yield bench.name, normalize_workload(bench.name), p

    # Rodinia: recursive.
    for p in sorted(RODINIA_ROOT.rglob("*_repeat.ptx")):
        if not is_isolated_ptx_file(p):
            continue
        source_folder = rodinia_source_folder(p)
        yield source_folder, normalize_workload(source_folder), p


def clean_line(line: str) -> str:
    return COMMENT_PAT.sub("", line).strip()


def is_instruction(line: str) -> bool:
    if not line:
        return False
    if LABEL_PAT.match(line):
        return False
    if DIRECTIVE_PAT.match(line):
        return False
    return INSTR_PAT.match(line) is not None


def parse_op(line: str):
    m = INSTR_PAT.match(line)
    if not m:
        return None

    full = m.group(1)
    parts = full.split(".")
    base = parts[0]
    qualifiers = parts[1:]

    dtypes = [q for q in qualifiers if q in DTYPE_SET]
    memspaces = [q for q in qualifiers if q in MEMSPACE_SET]

    instr_form = base
    if base in MEMORY_OPS and memspaces:
        instr_form = f"{base}.{memspaces[0]}"

    return full, base, instr_form, dtypes, memspaces


def dtype_bytes(dtype: str) -> int:
    if dtype in {"f64", "s64", "u64", "b64"}:
        return 8
    if dtype in {"f32", "s32", "u32", "b32"}:
        return 4
    if dtype in {"f16", "s16", "u16", "b16"}:
        return 2
    if dtype in {"s8", "u8", "b8"}:
        return 1
    return 0


def estimate_memory_bytes(base: str, dtypes, memspaces) -> int:
    if base not in MEMORY_OPS:
        return 0
    if not dtypes:
        return 0
    if not memspaces:
        return 0
    return dtype_bytes(dtypes[0])


def normalize_workload(folder_name: str) -> str:
    return str(folder_name).lower().replace("-", "").replace("_", "")


def target_tokens_from_ptx(ptx_name: str, source_folder: str):
    stem = Path(str(ptx_name)).stem
    stem = re.sub(r"^\d+_", "", stem)
    stem = stem.replace("_repeat", "")

    folder = str(source_folder).lower()

    prefixes = [
        "pre_euler3d_",
        "euler3d_",
        "hotspot3d_",
        "hotspot_",
        "kmeans_",
        "lavamd_",
        "myocyte_",
        "nn_",
        "nw_",
        "particlefilter_",
        "pathfinder_",
        "streamcluster_",
        "srad_",
    ]

    target = stem
    for pref in prefixes:
        if target.startswith(pref):
            target = target[len(pref):]
            break

    # Specific Rodinia mappings where entry names differ from file names.
    special = {
        "calculate_temp": ["calculate_temp"],
        "hotspotopt1": ["hotspotopt1"],
        "kmeanspoint": ["kmeanspoint"],
        "kernel_gpu_cuda": ["kernel_gpu_cuda"],
        "solver_2": ["solver_2"],
        "euclid": ["euclid"],
        "shared_1": ["shared_1"],
        "shared_2": ["shared_2"],
        "likelihood": ["likelihood"],
        "sum": ["sum"],
        "normalize_weights": ["normalize_weights"],
        "find_index": ["find_index"],
        "dynproc": ["dynproc"],
        "extract": ["extract"],
        "prepare": ["prepare"],
        "reduce": ["reduce"],
        "srad": ["srad"],
        "srad2": ["srad2"],
        "compress": ["compress"],
        "compute_step_factor": ["cuda_compute_step_factor", "compute_step_factor", "compute_step"],
        "compute_flux_contributions": ["cuda_compute_flux_contributions", "compute_flux_contributions"],
        "compute_flux": ["cuda_compute_fluxi", "compute_flux"],
        "time_step": ["cuda_time_step", "time_step"],
        "compute_cost": ["compute_cost"],
    }

    key = target.lower()
    return special.get(key, [key])


def select_target_entries(entries, text: str, ptx: Path, source_folder: str):
    if len(entries) <= 1:
        return entries

    tokens = target_tokens_from_ptx(ptx.name, source_folder)
    # Prefer earlier tokens. This matters for names like compute_flux,
    # which is also contained in compute_flux_contributions.
    for tok in tokens:
        selected = []
        tok_l = tok.lower()
        for m in entries:
            entry = m.group(1).strip().lower()
            if tok_l in entry:
                selected.append(m)

        if len(selected) == 1:
            return selected

    selected = []
    for m in entries:
        entry = m.group(1).strip().lower()
        if any(tok.lower() in entry for tok in tokens):
            selected.append(m)

    if len(selected) == 1:
        return selected

    if len(selected) > 1:
        print(f"AMBIGUOUS ENTRY: {ptx} tokens={tokens}")
        for m in selected:
            print("  ", m.group(1).strip())
        return selected[:1]

    print(f"NO TARGET ENTRY MATCH: {ptx} tokens={tokens}")
    print("Entries:")
    for m in entries:
        print("  ", m.group(1).strip())

    return []



def main():
    rows_raw = []
    all_ops, all_forms, all_dtypes, all_mems = set(), set(), set(), set()

    files = list(ptx_files())
    print(f"Found PTX files: {len(files)}")

    for source_folder, workload_norm, ptx in files:
        text = ptx.read_text(encoding="utf-8", errors="ignore")
        entries = list(ENTRY_PAT.finditer(text))

        if not entries:
            print(f"NO ENTRY: {ptx}")
            continue

        selected_entries = select_target_entries(entries, text, ptx, source_folder)
        if not selected_entries:
            continue

        for idx, m in enumerate(selected_entries):
            kernel_entry = m.group(1).strip()
            start = m.start()
            # Use original PTX entry boundaries, not selected-entry boundaries.
            original_idx = entries.index(m)
            end = entries[original_idx + 1].start() if original_idx + 1 < len(entries) else len(text)
            body = text[start:end]

            op_counts = Counter()
            form_counts = Counter()
            dtype_counts = Counter()
            mem_counts = Counter()
            mem_byte_counts = Counter()

            reg_counts = Counter()
            for rm in REG_DECL_PAT.finditer(body):
                reg_dtype = rm.group(1)
                reg_count = int(rm.group(2))
                reg_counts[reg_dtype] += reg_count

            registers_per_thread_approx = sum(reg_counts.values())

            for raw in body.splitlines():
                line = clean_line(raw)
                if not is_instruction(line):
                    continue

                parsed = parse_op(line)
                if not parsed:
                    continue

                full, base, form, dtypes, memspaces = parsed
                op_counts[base] += 1
                form_counts[form] += 1
                all_ops.add(base)
                all_forms.add(form)

                for dt in dtypes:
                    dtype_counts[dt] += 1
                    all_dtypes.add(dt)

                for ms in memspaces:
                    mem_counts[ms] += 1
                    all_mems.add(ms)

                b = estimate_memory_bytes(base, dtypes, memspaces)
                for ms in memspaces:
                    mem_byte_counts[ms] += b

            rows_raw.append({
                "source_folder": source_folder,
                "source_ptx": ptx.name,
                "workload": workload_norm,
                "kernel_entry": kernel_entry,
                "op_counts": op_counts,
                "form_counts": form_counts,
                "dtype_counts": dtype_counts,
                "mem_counts": mem_counts,
                "mem_byte_counts": mem_byte_counts,
                "reg_counts": reg_counts,
                "registers_per_thread_approx": registers_per_thread_approx,
            })

    if not rows_raw:
        raise SystemExit("No PTX rows collected")

    all_ops = sorted(all_ops)
    all_forms = sorted(all_forms)
    all_dtypes = sorted(all_dtypes)
    all_mems = sorted(all_mems)

    rows = []

    for item in rows_raw:
        op = item["op_counts"]
        forms = item["form_counts"]
        dtypes = item["dtype_counts"]
        mems = item["mem_counts"]
        mem_bytes = item["mem_byte_counts"]
        regs = item["reg_counts"]

        opcode_total = sum(op.values())
        form_total = sum(forms.values())
        dtype_total = sum(dtypes.values())
        mem_total = sum(mems.values())

        arith_total = sum(op[o] for o in ARITH_OPS)
        int_total = sum(op[o] for o in INT_OPS)
        control_total = sum(op[o] for o in CONTROL_OPS)
        predicate_total = sum(op[o] for o in PRED_OPS)
        load_total = sum(op[o] for o in LOAD_OPS)
        store_total = sum(op[o] for o in STORE_OPS)
        memory_total = sum(op[o] for o in MEMORY_OPS)

        global_mem_total = mems["global"]
        shared_mem_total = mems["shared"]
        local_mem_total = mems["local"]
        param_mem_total = mems["param"]

        estimated_global_mem_bytes = mem_bytes["global"]
        estimated_shared_mem_bytes = mem_bytes["shared"]
        estimated_local_mem_bytes = mem_bytes["local"]
        estimated_param_mem_bytes = mem_bytes["param"]
        estimated_const_mem_bytes = mem_bytes["const"]
        estimated_total_mem_bytes = sum(mem_bytes.values())

        float_total = sum(dtypes[x] for x in ["f16", "f32", "f64"])
        dtype_int_total = sum(dtypes[x] for x in [
            "s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64"
        ])

        compute_instr_total = arith_total + int_total

        arithmetic_intensity_global = (
            compute_instr_total / estimated_global_mem_bytes
            if estimated_global_mem_bytes else 0.0
        )

        arithmetic_intensity_total = (
            compute_instr_total / estimated_total_mem_bytes
            if estimated_total_mem_bytes else 0.0
        )

        row = {
            "source_folder": item["source_folder"],
            "source_ptx": item["source_ptx"],
            "workload": item["workload"],
            "kernel_entry": item["kernel_entry"],

            "opcode_total": opcode_total,
            "instr_form_total": form_total,
            "dtype_total": dtype_total,
            "memspace_total": mem_total,

            "arith_total": arith_total,
            "int_op_total": int_total,
            "control_total": control_total,
            "predicate_total": predicate_total,
            "load_total": load_total,
            "store_total": store_total,
            "memory_total": memory_total,

            "global_mem_total": global_mem_total,
            "shared_mem_total": shared_mem_total,
            "local_mem_total": local_mem_total,
            "param_mem_total": param_mem_total,

            "estimated_global_mem_bytes": estimated_global_mem_bytes,
            "estimated_shared_mem_bytes": estimated_shared_mem_bytes,
            "estimated_local_mem_bytes": estimated_local_mem_bytes,
            "estimated_param_mem_bytes": estimated_param_mem_bytes,
            "estimated_const_mem_bytes": estimated_const_mem_bytes,
            "estimated_total_mem_bytes": estimated_total_mem_bytes,

            "float_dtype_total": float_total,
            "int_dtype_total": dtype_int_total,

            "compute_instr_total": compute_instr_total,

            "registers_per_thread_approx": item["registers_per_thread_approx"],
            "reg_pred_count": regs["pred"],
            "reg_f16_count": regs["f16"],
            "reg_f32_count": regs["f32"],
            "reg_f64_count": regs["f64"],
            "reg_b16_count": regs["b16"],
            "reg_b32_count": regs["b32"],
            "reg_b64_count": regs["b64"],
            "reg_s32_count": regs["s32"],
            "reg_s64_count": regs["s64"],
            "reg_u32_count": regs["u32"],
            "reg_u64_count": regs["u64"],

            "arithmetic_intensity_global": arithmetic_intensity_global,
            "arithmetic_intensity_total": arithmetic_intensity_total,

            "fma_total": op["fma"],
            "mul_total": op["mul"],
            "add_total": op["add"],
            "div_total": op["div"],
            "bra_total": op["bra"],
            "setp_total": op["setp"],

            "arith_ratio": arith_total / opcode_total if opcode_total else 0.0,
            "memory_ratio": memory_total / opcode_total if opcode_total else 0.0,
            "load_ratio": load_total / opcode_total if opcode_total else 0.0,
            "store_ratio": store_total / opcode_total if opcode_total else 0.0,
            "control_ratio": control_total / opcode_total if opcode_total else 0.0,
            "predicate_ratio": predicate_total / opcode_total if opcode_total else 0.0,
            "global_mem_ratio": global_mem_total / mem_total if mem_total else 0.0,
            "shared_mem_ratio": shared_mem_total / mem_total if mem_total else 0.0,
            "local_mem_ratio": local_mem_total / mem_total if mem_total else 0.0,
            "param_mem_ratio": param_mem_total / mem_total if mem_total else 0.0,
            "float_dtype_ratio": float_total / dtype_total if dtype_total else 0.0,
            "int_dtype_ratio": dtype_int_total / dtype_total if dtype_total else 0.0,
            "mem_to_arith_ratio": memory_total / (arith_total + 1e-9),
        }

        for name in all_ops:
            c = op[name]
            row[f"op_{name}"] = c
            row[f"op_{name}_norm"] = c / opcode_total if opcode_total else 0.0

        for name in all_forms:
            safe = name.replace(".", "_")
            c = forms[name]
            row[f"iform_{safe}"] = c
            row[f"iform_{safe}_norm"] = c / form_total if form_total else 0.0

        for name in all_dtypes:
            c = dtypes[name]
            row[f"dtype_{name}"] = c
            row[f"dtype_{name}_norm"] = c / dtype_total if dtype_total else 0.0

        for name in all_mems:
            c = mems[name]
            row[f"mem_{name}"] = c
            row[f"mem_{name}_norm"] = c / mem_total if mem_total else 0.0

        for name in all_mems:
            b = mem_bytes[name]
            row[f"mem_bytes_{name}"] = b
            row[f"mem_bytes_{name}_norm"] = b / estimated_total_mem_bytes if estimated_total_mem_bytes else 0.0

        rows.append(row)

    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT_CSV}")
    print(f"Rows: {len(rows)}")
    print(f"Unique ops: {len(all_ops)}")
    print(f"Unique instruction forms: {len(all_forms)}")
    print(f"Unique dtypes: {len(all_dtypes)}")
    print(f"Unique memory spaces: {len(all_mems)}")


if __name__ == "__main__":
    main()
