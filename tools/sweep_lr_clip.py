#!/usr/bin/env python3
"""
LOGOS SHM optimizer sweep: LOGOS_LR x LOGOS_CLIP  (fresh runs, koi checkpoint/resume nahi).

Kaise kaam karta hai
  * Har config ek alag run hai (alag cwd) -> checkpoints/vocab aapas me nahi takraate.
  * LOGOS_STEPS = production horizon (default 500000) rakha jaata hai taaki optimizer ka
    T/alpha_H cosine anneal aur warmup REAL run ke shuruaati phase jaisa rahe; script step N
    pe process rok deta hai (--steps N). Chhota LOGOS_STEPS dene se schedule compress ho jaata.
  * Har run me dropout seed fixed (LOGOS_DROPOUT_SEED) aur data order deterministic -> fair compare.
  * Checkpoint har run me band (LOGOS_CKPT_FREQ=50000); run ke baad bhaari files delete.
  * Results sweep_out/results.json me har run ke baad save hote hain -> Kaggle session toote
    to dobara chalao, ho chuke runs skip ho jaate hain.
  * Pehla run vocab.bin + pre-tokenized .tok.bin banata hai; baaki runs reuse karte hain.

Stages (pehle lr, phir clip):
  python tools/sweep_lr_clip.py --binary build/logos_gpu --dataset /kaggle/working/dataset.txt --stage lr
  python tools/sweep_lr_clip.py --binary build/logos_gpu --dataset /kaggle/working/dataset.txt --stage clip
  python tools/sweep_lr_clip.py ... --stage grid          # poora LR x CLIP (mehenga)

Model size/seq env (LOGOS_D_MODEL, LOGOS_N_HEADS, LOGOS_N_LAYERS, LOGOS_SEQ_LEN, LOGOS_GRAD_ACCUM)
environment se inherit hota hai (notebook CELL 1C jaisa), ya --env KEY=VAL do.
"""
import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

HEAVY_EXT = (".bin", ".optstate", ".trainstate", ".tmp")


def parse_row(line):
    """Training log row: Step | F | Train_CE | Val_CE | Entropy | GNorm | aH | aL | T | LRx | Status"""
    if "|" not in line:
        return None
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 6:
        return None
    try:
        step = int(parts[0])
    except ValueError:
        return None

    def f(x):
        try:
            return float(x)
        except ValueError:
            return None

    return dict(
        step=step,
        F=f(parts[1]),
        ce=f(parts[2]),
        val=None if parts[3] == "N/A" else f(parts[3]),
        gnorm=f(parts[5]),
        status=parts[-1],
    )


def finite(x):
    return x is not None and math.isfinite(x)


def run_key(lr, clip, steps):
    return f"lr={lr:g}_clip={clip:g}_steps={steps}"


def summarize(rows, clip, overflow, reason, steps_reached, elapsed):
    ces = [r["ce"] for r in rows if finite(r["ce"])]
    vals = [r["val"] for r in rows if finite(r["val"])]
    gns = [r["gnorm"] for r in rows if finite(r["gnorm"])]
    diverged = reason.startswith("diverged") or not ces
    k = 3
    return dict(
        reason=reason,
        diverged=diverged,
        steps_reached=steps_reached,
        first_ce=ces[0] if ces else None,
        final_train_ce=statistics.mean(ces[-k:]) if ces else None,
        final_val_ce=statistics.mean(vals[-2:]) if vals else None,
        median_gnorm=statistics.median(gns) if gns else None,
        clip_active_frac=(sum(1 for g in gns if g > clip) / len(gns)) if gns else None,
        amp_overflows=overflow,
        minutes=round(elapsed / 60.0, 1),
    )


def cleanup(workdir):
    for name in os.listdir(workdir):
        if name == "vocab.bin" or name.endswith(".log"):
            continue
        if name.endswith(HEAVY_EXT):
            try:
                os.remove(os.path.join(workdir, name))
            except OSError:
                pass


def run_one(args, lr, clip, shared_vocab):
    key = run_key(lr, clip, args.steps)
    workdir = os.path.join(args.out, key)
    os.makedirs(workdir, exist_ok=True)
    if os.path.exists(shared_vocab):
        shutil.copy2(shared_vocab, os.path.join(workdir, "vocab.bin"))

    env = os.environ.copy()
    for k in ("LOGOS_CKPT", "LOGOS_START_STEP", "LOGOS_BEST_F"):   # hamesha fresh
        env.pop(k, None)
    env.update(
        {
            "LOGOS_LR": f"{lr:g}",
            "LOGOS_CLIP": f"{clip:g}",
            "LOGOS_STEPS": str(args.horizon),
            "LOGOS_LOG_FREQ": str(args.log_freq),
            "LOGOS_VAL_BATCHES": str(args.val_batches),
            "LOGOS_CKPT_FREQ": "50000",
            "LOGOS_WARMUP": str(args.warmup),
            "LOGOS_DROPOUT_SEED": str(args.seed),
        }
    )
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    log_path = os.path.join(workdir, "train.log")
    rows, overflow, reason = [], 0, "exited"
    t0 = time.time()
    proc = subprocess.Popen(
        [args.binary, "--train", args.dataset],
        cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    with open(log_path, "w") as lf:
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            if args.verbose:
                print(line, end="")
            if "OVERFLOW" in line:
                overflow += 1
            row = parse_row(line)
            if row:
                rows.append(row)
                if not finite(row["F"]) or not finite(row["ce"]) or row["ce"] > args.diverge_ce:
                    reason = "diverged (nan/inf/ce too high)"
                    break
                if row["step"] >= args.steps:
                    reason = "done"
                    break
            if time.time() - t0 > args.timeout * 60:
                reason = "timeout"
                break

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    elif reason == "exited" and proc.returncode not in (0, None):
        reason = f"crashed (exit={proc.returncode})"

    if not os.path.exists(shared_vocab) and os.path.exists(os.path.join(workdir, "vocab.bin")):
        shutil.copy2(os.path.join(workdir, "vocab.bin"), shared_vocab)   # agle runs reuse karenge
    cleanup(workdir)

    steps_reached = rows[-1]["step"] if rows else 0
    return key, summarize(rows, clip, overflow, reason, steps_reached, time.time() - t0)


def rank(results):
    def score(item):
        r = item[1]
        primary = r["final_val_ce"] if finite(r["final_val_ce"]) else r["final_train_ce"]
        return (r["diverged"] or primary is None, primary if primary is not None else 1e9)
    return sorted(results.items(), key=score)


def print_table(results):
    hdr = f"{'config':<34} {'val_CE':>8} {'train_CE':>9} {'gnorm_med':>10} {'clip%':>6} {'ovf':>4} {'min':>5}  status"
    print("\n" + hdr)
    print("-" * len(hdr))
    for key, r in rank(results):
        fmt = lambda x, p=4: "   n/a" if x is None else f"{x:.{p}f}"
        clip_pct = "n/a" if r["clip_active_frac"] is None else f"{100*r['clip_active_frac']:.0f}"
        print(f"{key:<34} {fmt(r['final_val_ce']):>8} {fmt(r['final_train_ce']):>9} "
              f"{fmt(r['median_gnorm'], 3):>10} {clip_pct:>6} {r['amp_overflows']:>4} {r['minutes']:>5}  {r['reason']}")


def main():
    ap = argparse.ArgumentParser(description="LOGOS SHM optimizer LR x CLIP sweep")
    ap.add_argument("--binary", required=True, help="logos_gpu binary path")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="sweep_out")
    ap.add_argument("--stage", choices=["lr", "clip", "grid"], default="lr")
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-4, 1e-3, 2e-3, 4e-3, 8e-3])
    ap.add_argument("--clips", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0])
    ap.add_argument("--lr", type=float, default=None, help="clip stage me fixed LR (default: lr-stage ka best)")
    ap.add_argument("--clip", type=float, default=0.3, help="lr stage me fixed CLIP")
    ap.add_argument("--steps", type=int, default=400, help="har run itne steps pe roka jaata hai")
    ap.add_argument("--horizon", type=int, default=500000, help="LOGOS_STEPS (schedule horizon, production wala)")
    ap.add_argument("--warmup", type=int, default=50, help="LOGOS_WARMUP (short sweep ke liye; prod default 100)")
    ap.add_argument("--log-freq", type=int, default=25)
    ap.add_argument("--val-batches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--diverge-ce", type=float, default=12.0)
    ap.add_argument("--timeout", type=float, default=90.0, help="minutes per run")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    shared_vocab = os.path.join(args.out, "vocab.bin")

    if args.stage == "lr":
        grid = [(lr, args.clip) for lr in args.lrs]
    elif args.stage == "clip":
        lr = args.lr
        if lr is None:
            cand = {k: v for k, v in results.items() if not v["diverged"] and f"_clip={args.clip:g}_" in k}
            if not cand:
                sys.exit("clip stage ke liye --lr do, ya pehle --stage lr chalao")
            best_key = rank(cand)[0][0]
            lr = float(best_key.split("_")[0].split("=")[1])
            print(f"[sweep] lr stage ka best LR = {lr:g} ({best_key})")
        grid = [(lr, c) for c in args.clips]
    else:
        grid = [(lr, c) for lr in args.lrs for c in args.clips]

    print(f"[sweep] {len(grid)} runs | steps={args.steps} | horizon={args.horizon} | warmup={args.warmup}")
    for i, (lr, clip) in enumerate(grid, 1):
        key = run_key(lr, clip, args.steps)
        if key in results and results[key]["reason"] in ("done", "diverged (nan/inf/ce too high)"):
            print(f"[{i}/{len(grid)}] skip (already done): {key}")
            continue
        print(f"[{i}/{len(grid)}] run: {key}", flush=True)
        key, res = run_one(args, lr, clip, shared_vocab)
        results[key] = res
        json.dump(results, open(results_path, "w"), indent=2)
        print(f"    -> {res['reason']} | val_CE={res['final_val_ce']} train_CE={res['final_train_ce']} "
              f"gnorm_med={res['median_gnorm']} ({res['minutes']} min)", flush=True)

    print_table(results)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config"] + list(next(iter(results.values())).keys()) if results else ["config"])
        for key, r in rank(results):
            w.writerow([key] + list(r.values()))

    ok = [(k, r) for k, r in rank(results) if not r["diverged"]]
    if ok:
        best_key, best = ok[0]
        lr_s = best_key.split("_")[0].split("=")[1]
        clip_s = best_key.split("_")[1].split("=")[1]
        print(f"\nBEST: LOGOS_LR={lr_s} LOGOS_CLIP={clip_s}  "
              f"(val_CE={best['final_val_ce']}, train_CE={best['final_train_ce']})")
        print("Tip: best LR agar grid ke kinare (sabse chhota/bada) pe hai to range badhao; "
              "clip_active_frac ~100% matlab clip hi effective LR tay kar raha hai.")


if __name__ == "__main__":
    main()
