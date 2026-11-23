#!/usr/bin/env python3
"""LOGOS mutation testing (dependency-free, GCC/Clang only; replaces the old Mull job).

For every mutant: copy include/, apply ONE source edit, rebuild the affected CPU tests and run
them. A mutant is
  KILLED    - a test fails, crashes or times out (good: the suite noticed the change)
  SURVIVED  - every affected test still passes (bad: untested behaviour, or an equivalent mutant)
  INVALID   - the mutant does not compile (ignored in the score)
  STALE     - the 'old' text no longer occurs exactly once in the file (mutant list needs updating)

Usage:  python3 tools/mutation_test.py [--min-score 0.80] [--jobs N] [--only SUBSTR] [--list]
Exit:   0 ok | 1 score below --min-score | 2 baseline failed or STALE mutants
"""
import argparse, concurrent.futures as cf, json, os, pathlib, shutil, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
CXX = os.environ.get("CXX", "g++")

# cheap tests first so killed mutants stop early
TESTS = {
    "physics":     "tests/cpu/test_physics_cpu.cpp",
    "tokenizer":   "tests/cpu/test_tokenizer.cpp",
    "integration": "tests/cpu/test_integration.cpp",
    "fuzz":        "tests/cpu/test_fuzz.cpp",
}
AFFECTED = {
    "Tokenizer.hpp":    ["tokenizer", "integration", "fuzz"],
    "PathIntegral.hpp": ["physics", "fuzz"],
    "TokenBin.hpp":     ["integration", "fuzz"],
}

# (name, header, old, new)  — 'old' must occur EXACTLY once in the header
MUTANTS = [
    # ---- Tokenizer.hpp ----
    ("tok: tie-break reversed",        "Tokenizer.hpp", "kv.first < best_key", "kv.first > best_key"),
    ("tok: pick least frequent pair",  "Tokenizer.hpp", "if (kv.second > best_cnt ||", "if (kv.second < best_cnt ||"),
    ("tok: min pair freq 2 -> 3",      "Tokenizer.hpp", "if (best_cnt < 2) break;", "if (best_cnt < 3) break;"),
    ("tok: piece cap > -> >=",         "Tokenizer.hpp", "while (len > MAX_PIECE_BYTES) {", "while (len >= MAX_PIECE_BYTES) {"),
    ("tok: tab not whitespace",        "Tokenizer.hpp", "c == '\\r' || c == '\\t' || c == '\\v'", "c == '\\r' || c == '\\v'"),
    ("tok: no trailing-space glue",    "Tokenizer.hpp", "if (j < n && s[j - 1] == ' ' && (j - i) > 1) --j;", "if (false) --j;"),
    ("tok: encode room off by one",    "Tokenizer.hpp", "const int64_t room = (int64_t)max_len - 2;", "const int64_t room = (int64_t)max_len - 1;"),
    ("tok: encode wrong BOS",          "Tokenizer.hpp", "ids.push_back(TOKEN_BOS);", "ids.push_back(TOKEN_UNK);"),
    ("tok: decode keeps PAD",          "Tokenizer.hpp", "if (id == TOKEN_BOS || id == TOKEN_EOS || id == TOKEN_PAD) continue;", "if (id == TOKEN_BOS || id == TOKEN_EOS) continue;"),
    ("tok: decode bounds <=",          "Tokenizer.hpp", "if (id >= 0 && id < (int)id_to_token.size())", "if (id >= 0 && id <= (int)id_to_token.size())"),
    ("tok: accept any vocab version",  "Tokenizer.hpp", "if (!f || ver != VOCAB_VERSION) {", "if (!f || ver == VOCAB_VERSION) {"),
    ("tok: accept n==0",               "Tokenizer.hpp", "if (n <= 0 || n > MAX_VOCAB_SIZE) {", "if (n < 0 || n > MAX_VOCAB_SIZE) {"),
    ("tok: max_tokens<=0 unchecked",   "Tokenizer.hpp", "if (max_tokens <= 0 || text.empty()) return ids;", "if (text.empty()) return ids;"),
    # ---- PathIntegral.hpp (physics) ----
    ("phys: amplitude sign flipped",   "PathIntegral.hpp", "-= action / hbar;", "+= action / hbar;"),
    ("phys: negative loss allowed",    "PathIntegral.hpp", "const float action = std::max(loss, 0.0f) * std::max(step_norm, 0.0f);", "const float action = loss * step_norm;"),
    ("phys: negative norm allowed",    "PathIntegral.hpp", "* std::max(step_norm, 0.0f);", "* step_norm;"),
    ("phys: no finite guard",          "PathIntegral.hpp", "if (!std::isfinite(action)) return;", "if (false) return;"),
    ("phys: ema coefficients",         "PathIntegral.hpp", "0.98f * ema_action + 0.02f * action", "0.90f * ema_action + 0.10f * action"),
    ("phys: lr scale formula",         "PathIntegral.hpp", "return lr_min_frac + (1.0f - lr_min_frac) * rel;", "return lr_min_frac * (1.0f - lr_min_frac) * rel;"),
    ("phys: lr rel not clipped",       "PathIntegral.hpp", "const float rel = std::min(1.0f, std::exp(-(last_action - ema_action) / (ema_action + 1e-8f)));", "const float rel = std::exp(-(last_action - ema_action) / (ema_action + 1e-8f));"),
    ("phys: hbar not clamped",         "PathIntegral.hpp", "hbar(std::max(1e-6f, hbar_))", "hbar(hbar_)"),
    ("phys: best-amplitude sentinel",  "PathIntegral.hpp", "if (best_log_amplitude <= -1e29f) return 1.0f;", "if (best_log_amplitude >= -1e29f) return 1.0f;"),
    ("phys: amplitude underflow clamp","PathIntegral.hpp", "return std::exp(std::max(log_amplitude, -88.0f)); }", "return std::exp(log_amplitude); }"),
    # ---- TokenBin.hpp ----
    ("bin: fingerprint check inverted","TokenBin.hpp", "if (info.fingerprint != tok.fingerprint()) return false;", "if (info.fingerprint == tok.fingerprint()) return false;"),
    ("bin: src size check inverted",   "TokenBin.hpp", "if (info.src_size != src_size) return false;", "if (info.src_size == src_size) return false;"),
    ("bin: payload size formula",      "TokenBin.hpp", "return (int64_t)fsz == TOKBIN_HEADER_BYTES + 2 * info.n_tokens;", "return (int64_t)fsz == TOKBIN_HEADER_BYTES + info.n_tokens;"),
    ("bin: loader drops last window",  "TokenBin.hpp", "if (pos_ + seq_len + 1 > range_) {", "if (pos_ + seq_len + 1 >= range_) {"),
    ("bin: target not shifted",        "TokenBin.hpp", "target_ids[(size_t)i] = tmp_[(size_t)i + 1];", "target_ids[(size_t)i] = tmp_[(size_t)i];"),
    ("bin: truncated file accepted",   "TokenBin.hpp", "(int64_t)fsz < TOKBIN_HEADER_BYTES + 2 * info.n_tokens", "(int64_t)fsz > TOKBIN_HEADER_BYTES + 2 * info.n_tokens"),
    ("bin: chunk cut ignores \\v \\f",  "TokenBin.hpp", "inline bool is_ws(unsigned char c) { return Tokenizer::is_ws_byte(c); }", "inline bool is_ws(unsigned char c) { return c == ' ' || c == '\\n' || c == '\\r' || c == '\\t'; }"),
]


def build_and_run(src: pathlib.Path, inc_dir: pathlib.Path, work: pathlib.Path, name: str, timeout: int):
    """returns ('PASS'|'FAIL'|'TIMEOUT'|'COMPILE_ERROR', detail)"""
    exe = work / (name + ".exe")
    cmd = [CXX, "-std=c++20", "-O1", "-pthread", "-w", f"-I{inc_dir}", f"-I{ROOT / 'tests' / 'cpu'}",
           str(src), "-o", str(exe)]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        return "COMPILE_ERROR", cp.stderr[-400:]
    env = dict(os.environ, LOGOS_FUZZ_ITERS="500", TMPDIR=str(work), TMP=str(work), TEMP=str(work))
    try:
        rp = subprocess.run([str(exe)], capture_output=True, text=True, timeout=timeout, cwd=work, env=env)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    return ("PASS" if rp.returncode == 0 else "FAIL"), (rp.stderr or rp.stdout)[-300:]


def run_one(idx_mutant, timeout):
    idx, (name, header, old, new) = idx_mutant
    work = pathlib.Path(tempfile.mkdtemp(prefix=f"logos_mut_{idx}_"))
    try:
        inc = work / "include"
        shutil.copytree(ROOT / "include", inc)
        path = inc / header
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            return dict(name=name, status="STALE", detail=f"'old' occurs {text.count(old)}x in {header}")
        path.write_text(text.replace(old, new), encoding="utf-8")
        for t in AFFECTED[header]:
            res, detail = build_and_run(ROOT / TESTS[t], inc, work, t, timeout)
            if res == "COMPILE_ERROR":
                return dict(name=name, status="INVALID", detail=detail)
            if res in ("FAIL", "TIMEOUT"):
                return dict(name=name, status="KILLED", killed_by=t, detail=res)
        return dict(name=name, status="SURVIVED", detail="all affected tests passed")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def baseline(timeout):
    work = pathlib.Path(tempfile.mkdtemp(prefix="logos_mut_base_"))
    try:
        ok = True
        for t, rel in TESTS.items():
            res, detail = build_and_run(ROOT / rel, ROOT / "include", work, t, timeout)
            print(f"  baseline {t:12s} {res}")
            if res != "PASS":
                print(detail)
                ok = False
        return ok
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=0.80)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--only", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", default="mutation_report.json")
    a = ap.parse_args()

    muts = [m for m in MUTANTS if a.only in m[0]]
    if a.list:
        for m in muts: print(m[0])
        return 0

    print(f"LOGOS mutation test: {len(muts)} mutants, compiler={CXX}, jobs={a.jobs}")
    t0 = time.time()
    if not baseline(a.timeout):
        print("BASELINE FAILED — fix the unmutated suite first")
        return 2

    results = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(lambda im: run_one(im, a.timeout), list(enumerate(muts))):
            results.append(r)
            tag = r["status"] + (f" ({r['killed_by']})" if r.get("killed_by") else "")
            print(f"  {tag:26s} {r['name']}")

    killed = sum(r["status"] == "KILLED" for r in results)
    survived = [r for r in results if r["status"] == "SURVIVED"]
    stale = [r for r in results if r["status"] == "STALE"]
    invalid = sum(r["status"] == "INVALID" for r in results)
    valid = killed + len(survived)
    score = killed / valid if valid else 0.0

    print(f"\nkilled={killed} survived={len(survived)} invalid={invalid} stale={len(stale)} "
          f"score={score:.2%} (min {a.min_score:.0%}) in {time.time() - t0:.0f}s")
    for r in survived: print("  SURVIVED:", r["name"])
    for r in stale:    print("  STALE:   ", r["name"], "-", r["detail"])

    pathlib.Path(a.report).write_text(json.dumps(
        dict(score=score, killed=killed, survived=len(survived), invalid=invalid, stale=len(stale),
             mutants=results), indent=2), encoding="utf-8")

    if stale: return 2
    return 0 if score >= a.min_score else 1


if __name__ == "__main__":
    sys.exit(main())
