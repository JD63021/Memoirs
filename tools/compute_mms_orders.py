#!/usr/bin/env python3
import argparse
import glob
import math
import re
from pathlib import Path


def get_float(text, key):
    m = re.search(rf"{re.escape(key)}\s*=\s*([-+0-9.eE]+)", text)
    return float(m.group(1)) if m else None


def get_int(text, key):
    x = get_float(text, key)
    return int(x) if x is not None else None


def parse_family(log_dir: Path, family: str, prec: str):
    rows = []
    for fn in sorted(glob.glob(str(log_dir / f"{family}_*_{prec}_mms.log"))):
        path = Path(fn)
        txt = path.read_text(errors="replace")
        mN = re.search(rf"{re.escape(family)}_(\d+)_{re.escape(prec)}_mms\.log$", path.name)
        if not mN:
            continue
        l2 = get_float(txt, "L2Error")
        h1 = get_float(txt, "H1SemiError")
        it = get_int(txt, "iterations")
        res = get_float(txt, "finalRelativeResidual")
        if l2 is None or h1 is None or l2 < 0 or h1 < 0:
            continue
        rows.append({"N": int(mN.group(1)), "L2": l2, "H1": h1, "it": it, "res": res})
    rows.sort(key=lambda r: r["N"])
    return rows


def report(rows, family, prec):
    print(f"\n==== {prec.upper()} {family} MMS order ====")
    if not rows:
        print("No valid error logs found.")
        return 1
    print("N        L2Error          H1SemiError       it   finalRel")
    for r in rows:
        res = "None" if r["res"] is None else f"{r['res']:.3e}"
        print(f"{r['N']:<8d} {r['L2']:<16.9e} {r['H1']:<16.9e} {r['it']}   {res}")
    print("\nOrders:")
    for a, b in zip(rows, rows[1:]):
        pL2 = math.log(a["L2"] / b["L2"]) / math.log(b["N"] / a["N"])
        pH1 = math.log(a["H1"] / b["H1"]) / math.log(b["N"] / a["N"])
        print(f"{a['N']} -> {b['N']}:  L2 order = {pL2:.3f}, H1 order = {pH1:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=Path)
    args = ap.parse_args()
    status = 0
    for prec in ["dp", "sp"]:
        for family in ["hex", "tet"]:
            rows = parse_family(args.log_dir, family, prec)
            status |= report(rows, family, prec)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
