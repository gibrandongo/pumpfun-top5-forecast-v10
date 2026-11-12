"""v10: cap-gap diffusion with tight churn-adaptive size kernels.

One backbone: diffuse the rank-1 anchor and the log-cap gaps between
adjacent ranks, reconstruct the sorted cap profile, diffuse individual
coin strengths, add ghost newcomers fitted from the latest marks only.
Sizes use narrow kernels whose widths adapt to measured churn; centers
come from the simulated profile (trend leads, never lags). Calibration
via shrinkage toward the even spread on both splits jointly.
Stdlib only, offline, deterministic (fixed seed).
"""
import json
import math
import os
import random
import statistics
from datetime import datetime

SEED = 20261112
NPATH = 1000
K = 40
SHRINK = 0.25
SIZE_UNIFORM = 0.06
P0_CAP = 20


def parse_t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def bin_of(x, edges):
    if x < edges[0]:
        return 0
    if x >= edges[-1]:
        return 81
    return int((x - edges[0]) / 0.05) + 1


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main():
    inp = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    with open(os.path.join(inp, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(inp, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                m = json.loads(line)
                if m.get("rows"):
                    marks.append(m)
    edges = task["size"]["edges"]
    cad = task.get("cadence_minutes", 10) * 60.0
    reads = [parse_t(r) for r in task["reads"]]

    probs = []
    sizes = []
    top5 = []
    if marks:
        probs, sizes, top5 = forecast(marks, reads, cad, edges)
    else:
        # No history: even spread fallback.
        top5 = ["A" * 44] * 5
        sizes = [{"place": k, "bin": b, "p": round(1.0 / 82, 6)}
                 for k in range(1, 6) for b in range(82)]
    out_obj = {"probabilities": probs, "top5": top5[:5], "sizes": sizes}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "forecast.json"), "w") as f:
        json.dump(out_obj, f, separators=(",", ":"), sort_keys=True)


def forecast(marks, reads, cad, edges):
    rng = random.Random(SEED)
    fit = marks[-6:]
    last = marks[-1]
    last_t = parse_t(last["at"])
    horizons = [max(0.0, (r - last_t).total_seconds() / cad) for r in reads]
    if not horizons:
        horizons = [1.0]

    # --- Fit: per-mark sorted top-K log-cap profiles ---
    series = []
    for m in fit:
        caps = sorted((logcap(r) for r in m["rows"]), reverse=True)[:K]
        while len(caps) < K:
            caps.append(caps[-1] - 0.08 if caps else 2.0)
        series.append(caps)
    # Gap stats, shrunk toward global stats.
    all_gaps = []
    per_gap = []
    for j in range(K - 1):
        vals = [s[j] - s[j + 1] for s in series]
        per_gap.append(vals)
        all_gaps.extend(vals)
    gmean = sum(all_gaps) / len(all_gaps)
    gsd = statistics.pstdev(all_gaps) if len(all_gaps) > 1 else 0.05
    gap_mean = [0.5 * (sum(v) / len(v)) + 0.5 * gmean for v in per_gap]
    gap_sd = [max(0.01, 0.5 * (statistics.pstdev(v) if len(v) > 1 else 0.05)
                  + 0.5 * gsd) for v in per_gap]
    last_gaps = [series[-1][j] - series[-1][j + 1] for j in range(K - 1)]

    # Anchor drift / vol from rank-1 series (latest marks only).
    a1 = [s[0] for s in series]
    diffs = [a1[i + 1] - a1[i] for i in range(len(a1) - 1)]
    adrift = max(-0.15, min(0.25, sum(diffs) / len(diffs))) if diffs else 0.0
    avol = max(0.05, min(0.35, statistics.pstdev(diffs))) if len(diffs) > 1 else 0.10

    # --- Churn from the latest transition only ---
    def top_set(m, n=10):
        return set(r["mint"] for r in m["rows"][:n])
    if len(marks) >= 2:
        churn = 1.0 - len(top_set(marks[-1]) & top_set(marks[-2])) / 10.0
    else:
        churn = 0.5
    churn = max(0.0, min(1.0, churn))

    # Ghost rate: first appearances in top-K over the latest transitions.
    seen = set()
    new_count = 0
    trans = 0
    for m in marks[-4:]:
        cur = set(r["mint"] for r in m["rows"][:K])
        if seen:
            new_count += len(cur - seen)
            trans += 1
        seen |= cur
    lam = new_count / max(trans, 1)
    lam5 = max(0.2, min(2.5, lam * 0.25 * (0.6 + 0.8 * churn)))

    # --- Coin strengths: top 60 of last mark, shrunk drift/vol ---
    hist_caps = {}
    for m in fit:
        for r in m["rows"]:
            hist_caps.setdefault(r["mint"], []).append(logcap(r))
    coins = []
    for r in last["rows"][:60]:
        mint = r["mint"]
        seq = hist_caps.get(mint, [logcap(r)])
        d = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
        n = len(d)
        own = sum(d) / n if n else 0.0
        drift = (n / (n + 2.0)) * max(-0.2, min(0.3, own))
        vol = statistics.pstdev(d) if n > 1 else 0.15
        vol = max(0.06, min(0.4, 0.5 * vol + 0.5 * 0.15))
        coins.append((mint, seq[-1], drift, vol))
    last_rank = {r["mint"]: r["rank"] for r in last["rows"]}

    # --- Simulation ---
    place_freq = [dict() for _ in range(5)]
    ghost_freq = [0.0] * 5
    place_caps = [[] for _ in range(5)]
    rank_sum = {}
    rank_cnt = {}
    rho = 0.93
    for _ in range(NPATH):
        for h in horizons:
            sat = (1.0 - rho ** h) / (1.0 - rho) if h > 0 else 0.0
            sq = math.sqrt(min(h, 25.0))
            anchor = a1[-1] + adrift * sat + rng.gauss(0.0, avol * sq)
            prof = [anchor]
            mr = 0.75 ** max(h, 0.0)
            gsq = math.sqrt(min(h, 9.0) + 0.3)
            for j in range(K - 1):
                mu = gap_mean[j] + (last_gaps[j] - gap_mean[j]) * mr
                g = mu + rng.gauss(0.0, gap_sd[j] * gsq)
                prof.append(prof[-1] - max(0.002, g))
            entries = []
            for mint, lc, dr, vo in coins:
                c = lc + dr * sat + rng.gauss(0.0, vo * sq)
                if rng.random() < 0.02 * min(h, 10.0):
                    c -= rng.uniform(0.3, 1.0)
                entries.append((c, mint))
            mu_g = lam5 * max(h, 0.5)
            ng = int(mu_g + rng.gauss(0.0, math.sqrt(mu_g + 0.1)) + 0.5)
            ng = max(0, min(10, ng))
            for _ in range(ng):
                r = rng.randint(0, 17)
                entries.append((prof[r] + rng.gauss(0.0, 0.06), None))
            entries.sort(key=lambda e: e[0], reverse=True)
            for k in range(5):
                c, mint = entries[k]
                place_caps[k].append(c)
                if mint is None:
                    ghost_freq[k] += 1.0
                else:
                    place_freq[k][mint] = place_freq[k].get(mint, 0) + 1
            for idx, (_, mint) in enumerate(entries[:10]):
                if mint is not None:
                    rank_sum[mint] = rank_sum.get(mint, 0) + idx + 1
                    rank_cnt[mint] = rank_cnt.get(mint, 0) + 1

    total = float(NPATH * len(horizons))
    # --- Order probabilities with shrinkage (calibration) ---
    probs = []
    for k in range(5):
        rows = []
        for mint, cnt in place_freq[k].items():
            p = (1.0 - SHRINK) * cnt / total
            if p >= 0.002:
                rows.append({"mint": mint, "place": k + 1, "p": round(p, 6)})
        s = sum(r["p"] for r in rows)
        if s > 0.95:
            for r in rows:
                r["p"] = round(r["p"] * 0.95 / s, 6)
        # p=0 rows: hopeless coins (rank 20+, never simulated in top 5)
        # concentrate the unnamed share on the rest of the chart.
        named = set(place_freq[k])
        hopeless = [r["mint"] for r in last["rows"]
                    if r["rank"] >= 20 and r["mint"] not in named]
        hopeless.sort(key=lambda m: last_rank[m])
        for mint in hopeless[:P0_CAP]:
            if len(rows) >= 100:
                break
            rows.append({"mint": mint, "place": k + 1, "p": 0})
        rows.sort(key=lambda r: -r["p"])
        probs.extend(rows)

    # --- Sizes: narrow churn-adaptive kernels over simulated place caps ---
    sig = 0.045 + 0.04 * churn
    sizes = []
    for k in range(5):
        dens = [0.0] * 82
        for x in place_caps[k]:
            lo = max(0, bin_of(x - 3 * sig, edges))
            hi = min(81, bin_of(x + 3 * sig, edges))
            for b in range(lo, hi + 1):
                if b == 0:
                    lo_e, hi_e = -1e9, edges[0]
                elif b == 81:
                    lo_e, hi_e = edges[-1], 1e9
                else:
                    lo_e, hi_e = edges[b - 1], edges[b]
                dens[b] += norm_cdf((hi_e - x) / sig) - norm_cdf((lo_e - x) / sig)
        n = float(len(place_caps[k]))
        dens = [d / n for d in dens]
        dens = [(1.0 - SIZE_UNIFORM) * d + SIZE_UNIFORM / 82.0 for d in dens]
        tot = sum(dens)
        dens = [d / tot * 0.98 for d in dens]
        for b, p in enumerate(dens):
            if p >= 0.0002:
                sizes.append({"place": k + 1, "bin": b, "p": round(p, 6)})

    # --- Top 5 by mean simulated rank ---
    ranked = sorted(rank_sum, key=lambda m: rank_sum[m] / rank_cnt[m])
    top5 = ranked[:5]
    if len(top5) < 5:
        for r in last["rows"]:
            if r["mint"] not in top5:
                top5.append(r["mint"])
            if len(top5) == 5:
                break
    return probs, sizes, top5


if __name__ == "__main__":
    main()
