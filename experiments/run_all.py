"""Run every experiment of the manuscript and write CSV files to results/.

Usage:  python -I experiments/run_all.py [--quick] [--out results]

E1  Static risk landscape: exact R(a, w) versus the additive prediction.
E2  Non-identifiability of concordance: iso-concordance curves and the risk
    along them.
E3  Incorporation bias of report-referenced AI performance.
E4  Monte Carlo time course: five drift scenarios x six monitoring strategies.
E5  Audit design: detection of algorithmic and behavioural drift as a
    function of audit fraction and sentinel rate.
E6  One-way sensitivity of the interaction share to model parameters.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from cdrift import Params, SCENARIOS  # noqa: E402
from cdrift import risk as rk  # noqa: E402
from cdrift import sim  # noqa: E402

MASKED_WORKLOAD = 0.34   # workload effect at which concordance stays flat under coupled drift

SCEN = {
    "S1_algorithmic": (SCENARIOS["S1_algorithmic"], {}),
    "S2_behavioural": (SCENARIOS["S2_behavioural"], {}),
    "S3_coupled": (SCENARIOS["S3_coupled"], {}),
    "S4_masked": (SCENARIOS["S3_coupled"], {"workload_effect": MASKED_WORKLOAD}),
}
STRATEGIES = {
    "none": ("none", {}),
    "report_referenced": ("report_referenced", {}),
    "concordance": ("concordance", {}),
    "delayed_truth": ("delayed_truth", {}),
    "dual_audit_3pct_sent2": ("dual_audit", {"audit_fraction": 0.03, "sentinels_per_day": 2.0}),
    "dual_audit_10pct_sent2": ("dual_audit", {"audit_fraction": 0.10, "sentinels_per_day": 2.0}),
    "dual_truth_sent2": ("dual_truth_sentinel", {"sentinels_per_day": 2.0}),
}


def static_state(M: sim.Model, a, w):
    """Cell state at drift level a and workload w with trust and skill held at baseline."""
    a = np.atleast_1d(np.asarray(a, float))
    w = np.atleast_1d(np.asarray(w, float))
    T = np.full(a.shape, M.T_eq)
    U = np.full(a.shape, M.u_eq)
    return M.cell_state(a, w, T, U)


# ----------------------------------------------------------------------------
def e1_landscape(p: Params, out: str, n=41):
    M = sim.Model(p)
    cells = M.cells
    eb, cb, ob, _, _ = M.baseline_state()
    A, W = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    e, c, o, _, v = static_state(M, A.ravel(), W.ravel())
    dec = rk.decompose(cells, eb, cb, ob, e, c, o)
    df = pd.DataFrame({
        "a": A.ravel(), "w": W.ravel(), "v": v[:, 0],
        "R": dec["R1"] * 1e3, "R_additive": (dec["R0"] + dec["algorithmic"] + dec["behavioural"]) * 1e3,
        "interaction": dec["interaction"] * 1e3, "residual": dec["residual"],
    })
    df["interaction_share"] = np.where(df.R - dec["R0"] * 1e3 > 1e-9,
                                       df.interaction / (df.R - dec["R0"] * 1e3), 0.0)
    # pgfplots expects x to vary fastest within each scan line
    df.sort_values(["w", "a"]).to_csv(os.path.join(out, "e1_landscape.csv"), index=False)
    return {"R0_per_1000": float(dec["R0"][0] * 1e3), "max_abs_residual": float(np.abs(dec["residual"]).max()),
            "R_corner": float(df.R.iloc[-1]), "R_additive_corner": float(df.R_additive.iloc[-1])}


# ----------------------------------------------------------------------------
def e2_identifiability(p: Params, out: str, n=101):
    """Iso-concordance curves in the (drift level a, verification v) plane."""
    M = sim.Model(p)
    cells = M.cells

    def state_av(a, v):
        a = np.atleast_1d(a)
        e = np.clip(M.e0 + a[:, None] * M.de, 0, 1)
        h = np.broadcast_to(M.h0, e.shape)
        vv = np.broadcast_to(np.atleast_1d(v)[:, None], e.shape)
        c, o = rk.interception(vv, h, p.rho_r, p.rho_w)
        return e, c, o

    # heat map
    A, V = np.meshgrid(np.linspace(0, 1, 51), np.linspace(0.05, 0.95, 51), indexing="ij")
    e, c, o = state_av(A.ravel(), V.ravel())
    grid = pd.DataFrame({"a": A.ravel(), "v": V.ravel(),
                         "R": rk.risk(cells, e, c, o) * 1e3,
                         "conc": rk.concordance(cells, e, c, o)})
    grid.sort_values(["v", "a"]).to_csv(os.path.join(out, "e2_grid.csv"), index=False)

    # iso-concordance curves through baseline (v_eq) at a = 0 and two other levels
    e0, c0, o0 = state_av(np.array([0.0]), np.array([M.v_eq]))
    k0 = float(rk.concordance(cells, e0, c0, o0)[0])
    levels = [k0 - 0.01, k0, k0 + 0.01]
    rows = []
    for lev in levels:
        for a in np.linspace(0, 1, n):
            lo, hi = 0.0, 1.0           # concordance decreases with v
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                ee, cc, oo = state_av(np.array([a]), np.array([mid]))
                if rk.concordance(cells, ee, cc, oo)[0] > lev:
                    lo = mid
                else:
                    hi = mid
            vv = 0.5 * (lo + hi)
            ee, cc, oo = state_av(np.array([a]), np.array([vv]))
            if not (0.001 < vv < 0.999):
                continue
            rows.append({"level": round(lev, 4), "a": a, "v": vv,
                         "R": float(rk.risk(cells, ee, cc, oo)[0] * 1e3),
                         "true_ai_error": float(rk.true_ai_error(cells, ee)[0]),
                         "override": float(rk.disagreement(cells, ee, cc, oo)[0])})
    iso = pd.DataFrame(rows)
    iso["R0"] = float(rk.risk(cells, e0, c0, o0)[0] * 1e3)
    iso.to_csv(os.path.join(out, "e2_iso_concordance.csv"), index=False)
    for i, lev in enumerate(levels):
        iso[iso.level == round(lev, 4)].to_csv(os.path.join(out, f"e2_iso_level{i}.csv"), index=False)
    base = iso[(iso.level == round(k0, 4))]
    return {"baseline_concordance": k0, "risk_at_a0": float(base.R.iloc[0]),
            "risk_at_max_a_same_concordance": float(base.R.iloc[-1]),
            "a_max_same_concordance": float(base.a.iloc[-1])}


# ----------------------------------------------------------------------------
def e3_incorporation_bias(p: Params, out: str):
    M = sim.Model(p)
    cells = M.cells
    rows = []
    for a in (0.0, 0.5, 1.0):
        for v in np.linspace(0.0, 1.0, 51):
            e = np.clip(M.e0 + a * M.de, 0, 1)[None, :]
            c, o = rk.interception(np.full(e.shape, v), np.broadcast_to(M.h0, e.shape), p.rho_r, p.rho_w)
            rows.append({"a": a, "v": v,
                         "true_sens": float(rk.true_sensitivity(cells, e)[0]),
                         "report_sens": float(rk.report_referenced_sensitivity(cells, e, c, o)[0]),
                         "true_err": float(rk.true_ai_error(cells, e)[0]),
                         "report_err": float(rk.report_referenced_error(cells, e, c, o)[0]),
                         "R": float(rk.risk(cells, e, c, o)[0] * 1e3)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e3_incorporation_bias.csv"), index=False)
    d1 = df[df.a == 1.0]
    return {"true_sens_drifted": float(d1.true_sens.iloc[0]),
            "report_sens_drifted_at_v_eq": float(np.interp(M.v_eq, d1.v, d1.report_sens)),
            "report_sens_drifted_at_v_0.2": float(np.interp(0.2, d1.v, d1.report_sens)),
            "true_sens_baseline": float(df[df.a == 0.0].true_sens.iloc[0])}


# ----------------------------------------------------------------------------
def _summ(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(x, [50, 25, 75]))


def e4_timecourse(p0: Params, out: str, R: int, R_cal: int, seed=2026):
    rows, traj = [], {}
    thr_cache = {}
    for sname, (sc, sov) in SCEN.items():
        for gname, (mon, gov) in STRATEGIES.items():
            p = p0.with_(**gov)
            key = (mon, tuple(sorted(gov.items())))
            if key not in thr_cache:
                thr_cache[key] = sim.calibrate_thresholds(p, mon, R=R_cal, seed=seed + 1)
            th = thr_cache[key] if mon != "none" else None
            o = sim.simulate(p.with_(**sov), sc, mon, R=R, seed=seed, thresholds=th,
                             record=(gname == "none"))
            post = o["risk"][p.onset_day:] - o["R_base"]
            excess = post.mean(0) * 1e5                        # per 100,000 studies after onset
            fa = o["first_alarm"]
            truthA, truthB = sc["ai_drift"], sc["workload"]
            dA = fa["A"] - p.onset_day
            dB = fa["B"] - p.onset_day
            row = {
                "scenario": sname, "strategy": gname,
                "excess_per_100k_mean": excess.mean(),
                "excess_per_100k_median": np.median(excess),
                "excess_per_100k_p2.5": np.percentile(excess, 2.5),
                "excess_per_100k_p97.5": np.percentile(excess, 97.5),
                "detect_A": np.mean(~np.isnan(fa["A"])), "detect_B": np.mean(~np.isnan(fa["B"])),
                "delay_A_median": _summ(dA)[0], "delay_A_q1": _summ(dA)[1], "delay_A_q3": _summ(dA)[2],
                "delay_B_median": _summ(dB)[0], "delay_B_q1": _summ(dB)[1], "delay_B_q3": _summ(dB)[2],
                "false_alarm_pre_onset": np.mean(o["false_alarms_pre_onset"] > 0),
                "attribution_ok": np.nan,
            }
            if mon.startswith("dual"):
                gotA, gotB = ~np.isnan(fa["A"]), ~np.isnan(fa["B"])
                ok = (gotA == truthA) & (gotB == truthB)
                row["attribution_ok"] = ok.mean()
            rows.append(row)
            wk = o["risk"].mean(1).reshape(-1, 7).mean(1) * 1e3 if p.horizon_days % 7 == 0 else None
            traj[(sname, gname)] = o["risk"].mean(1) * 1e3
            if gname == "none":
                rec = o["record"]
                pd.DataFrame({"day": np.arange(p.horizon_days),
                              "R": o["risk"].mean(1) * 1e3,
                              "R0": np.full(p.horizon_days, o["R_base"] * 1e3),
                              "algorithmic": rec["algorithmic"] * 1e3,
                              "behavioural": rec["behavioural"] * 1e3,
                              "interaction": rec["interaction"] * 1e3,
                              "v": rec["v"], "trust": rec["T"], "deskill": rec["D"],
                              "ai_error": rec["e_mean"], "concordance": rec["conc"]}
                             ).iloc[::7].to_csv(os.path.join(out, f"e4_state_{sname}.csv"), index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out, "e4_summary.csv"), index=False)
    days = np.arange(p0.horizon_days)
    for sname in SCEN:
        df = pd.DataFrame({"day": days})
        for gname in STRATEGIES:
            df[gname] = traj[(sname, gname)]
        df.iloc[::7].to_csv(os.path.join(out, f"e4_traj_{sname}.csv"), index=False)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in thr_cache.items()},
              open(os.path.join(out, "e4_thresholds.json"), "w"), indent=2, default=float)
    return summ


# ----------------------------------------------------------------------------
def e5_audit_design(p0: Params, out: str, R: int, R_cal: int, seed=77):
    rows = []
    for r in (0.01, 0.02, 0.03, 0.05, 0.10, 0.20):
        for s in (0.0, 1.0, 2.0, 4.0):
            p = p0.with_(audit_fraction=r, sentinels_per_day=s)
            th = sim.calibrate_thresholds(p, "dual_audit", R=R_cal, seed=seed + 1)
            oA = sim.simulate(p, SCENARIOS["S1_algorithmic"], "dual_audit", R=R, seed=seed, thresholds=th)
            oB = sim.simulate(p, SCENARIOS["S2_behavioural"], "dual_audit", R=R, seed=seed, thresholds=th)
            dA = oA["first_alarm"]["A"] - p.onset_day
            dB = oB["first_alarm"]["B"] - p.onset_day
            rows.append({"audit_fraction": r, "sentinels_per_day": s,
                         "double_reads_per_day": r * p.cases_per_day,
                         "pA_365": np.mean(dA <= 365), "pB_365": np.mean(dB <= 365),
                         "delayA_median": _summ(dA)[0], "delayB_median": _summ(dB)[0]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e5_audit_design.csv"), index=False)
    return df


# ----------------------------------------------------------------------------
def e6_sensitivity(p0: Params, out: str):
    """Interaction share of the excess risk at full coupled drift (static)."""
    def share(p):
        M = sim.Model(p)
        eb, cb, ob, _, _ = M.baseline_state()
        e, c, o, _, _ = static_state(M, [1.0], [1.0])
        d = rk.decompose(M.cells, eb, cb, ob, e, c, o)
        exc = d["R1"][0] - d["R0"][0]
        return d["interaction"][0] / exc, exc * 1e3
    base_share, base_exc = share(p0)
    specs = {
        "rho_r": (0.70, 0.95), "rho_w": (0.20, 0.50), "v_min": (0.05, 0.30),
        "workload_effect": (0.40, 0.80), "trust_lambda": (2.0, 8.0), "prevalence": (0.05, 0.20),
        "severity_fp": (0.05, 0.30),
    }
    rows = [{"parameter": "baseline", "value": np.nan, "interaction_share": base_share, "excess_per_1000": base_exc}]
    for k, vals in specs.items():
        for val in vals:
            s, ex = share(p0.with_(**{k: val}))
            rows.append({"parameter": k, "value": val, "interaction_share": s, "excess_per_1000": ex})
    for scale in (0.5, 1.5):
        p = p0.with_(d_se_ai=tuple(scale * x for x in p0.d_se_ai), d_sp_ai=tuple(scale * x for x in p0.d_sp_ai))
        s, ex = share(p)
        rows.append({"parameter": "ai_drift_scale", "value": scale, "interaction_share": s, "excess_per_1000": ex})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e6_sensitivity.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few replicates, for a smoke test")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R, R_cal = (60, 200) if args.quick else (400, 1000)
    p = Params()
    t0 = time.time()
    meta = {"params": asdict(p), "replicates": R, "calibration_replicates": R_cal,
            "masked_workload_effect": MASKED_WORKLOAD, "numpy": np.__version__, "pandas": pd.__version__}
    meta["E1"] = e1_landscape(p, args.out)
    meta["E2"] = e2_identifiability(p, args.out)
    meta["E3"] = e3_incorporation_bias(p, args.out)
    e4 = e4_timecourse(p, args.out, R, R_cal)
    e5 = e5_audit_design(p, args.out, R // 2, R_cal // 2)
    e6 = e6_sensitivity(p, args.out)
    meta["runtime_s"] = time.time() - t0
    json.dump(meta, open(os.path.join(args.out, "run_metadata.json"), "w"), indent=2, default=float)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("# Simulation summary\n\n")
        for k in ("E1", "E2", "E3"):
            f.write(f"## {k}\n\n```\n{json.dumps(meta[k], indent=2, default=float)}\n```\n\n")
        f.write("## E4\n\n" + e4.round(3).to_markdown(index=False) + "\n\n")
        f.write("## E5\n\n" + e5.round(3).to_markdown(index=False) + "\n\n")
        f.write("## E6\n\n" + e6.round(3).to_markdown(index=False) + "\n")
    print(f"done in {meta['runtime_s']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
