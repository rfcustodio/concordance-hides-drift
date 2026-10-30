"""Monte Carlo time-course simulation of a deployed AI-assisted reading service.

The simulation is vectorised over replicates.  On each day it

1. sets the AI error, clinician verification and unassisted accuracy for every
   cell (stratum x truth) from the current state (algorithmic drift level,
   workload, trust, deskilling);
2. computes the *expected* severity-weighted risk analytically (no sampling
   noise in the outcome we score);
3. samples the observable data that each monitoring strategy would see;
4. updates trust and deskilling;
5. every 7 days updates the CUSUM charts of the active monitoring strategy,
   raises alarms and schedules remediation.

Monitoring strategies
---------------------
none                No post-deployment monitoring (approval-only).
report_referenced   AI false-negative rate estimated against the final report
                    (the most common practice reported by Chow et al. 2026).
concordance         Two-sided chart on the AI/report disagreement rate
                    (FAMOS / MHRA AI Airlock style).  Cause is ambiguous, so an
                    alarm triggers a full investigation that remediates both the
                    model and the reading practice after a longer delay.
delayed_truth       AI false-negative rate against delayed reference truth
                    (pathology / follow-up), positives only, after a lag.
dual_audit          Proposed: random AI-blinded expert double reads on a small
                    fraction of cases (and optionally projected sentinel cases)
                    giving separate charts for the AI error rate (A) and for
                    non-interception of AI errors (B).
dual_truth_sentinel Proposed, low cost: delayed reference truth for A plus
                    projected sentinel cases (known AI errors inserted in the
                    worklist, as in threat image projection) for B.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import Params
from .risk import make_cells, interception, p_final_wrong, decompose

STREAMS = {
    "none": (),
    "report_referenced": ("rep_fn",),
    "concordance": ("conc_up", "conc_down"),
    "delayed_truth": ("truth_fn",),
    "dual_audit": ("aud_err", "aud_miss", "sent_miss"),
    "dual_truth_sentinel": ("truth_fn", "sent_miss"),
}
# which remediation an alarm on each stream triggers
STREAM_ACTION = {
    "rep_fn": "A", "truth_fn": "A", "aud_err": "A",
    "aud_miss": "B", "sent_miss": "B",
    "conc_up": "AB", "conc_down": "AB",
}


def _ramp(t, start, length):
    return np.clip((t - start) / float(length), 0.0, 1.0)


@dataclass
class Model:
    p: Params

    def __post_init__(self):
        p = self.p
        self.cells = make_cells(p.stratum_weights, p.prevalence, p.severity_fn, p.severity_fp)
        K = len(p.stratum_weights)
        pos = self.cells.pos
        st = self.cells.stratum
        self.e0 = np.where(pos, 1 - np.asarray(p.se_ai0)[st], 1 - np.asarray(p.sp_ai0)[st])
        self.de = np.where(pos, np.asarray(p.d_se_ai)[st], np.asarray(p.d_sp_ai)[st])
        self.h0 = np.where(pos, np.asarray(p.se_h0)[st], np.asarray(p.sp_h0)[st])
        self.dh = np.where(pos, np.asarray(p.d_se_h)[st], np.asarray(p.d_sp_h)[st])
        # sentinel cases: projected AI false negatives on intermediate-difficulty positives
        self.sentinel_cell = int(np.where(pos & (st == 1))[0][0])
        # equilibrium trust at validation, and the corresponding verification
        a0 = self.agreement_if_verified(self.e0, self.h0)
        self.T_eq = a0 / (a0 + p.trust_lambda * (1 - a0))
        self.v_eq = self.verification(self.T_eq, 0.0)
        self.u_eq = 1 - self.v_eq

    # -- behavioural sub-model -------------------------------------------------
    def verification(self, T, W):
        p = self.p
        v = p.v_max - (p.v_max - p.v_min) * T
        return np.clip(v * (1 - p.workload_effect * W), 0.0, 1.0)

    def agreement_if_verified(self, e, h):
        """Expected fraction of verified cases where own read equals the AI."""
        a_cell = e * (1 - h) + (1 - e) * h
        return np.sum(self.cells.w * a_cell, axis=-1)

    def deskill_index(self, U):
        return np.clip((U - self.u_eq) / (1 - self.u_eq), 0.0, 1.0)

    def cell_state(self, a_lvl, W, T, U):
        """Return e, c, o, h, v for batch state arrays of shape (R,)."""
        e = np.clip(self.e0 + a_lvl[:, None] * self.de, 0, 1)
        h = np.clip(self.h0 - self.deskill_index(U)[:, None] * self.dh, 0, 1)
        v = self.verification(T, W)[:, None] * np.ones_like(e)
        c, o = interception(v, h, self.p.rho_r, self.p.rho_w)
        return e, c, o, h, v

    def baseline_state(self):
        one = np.ones(1)
        return self.cell_state(0 * one, 0 * one, self.T_eq * one, self.u_eq * one)


class Cusum:
    """Bank of one-sided Bernoulli log-likelihood-ratio CUSUM charts."""

    def __init__(self, R, p1_fn):
        self.S = np.zeros(R)
        self.max = np.zeros(R)
        self.p0 = None
        self.p1_fn = p1_fn
        self.base_x = np.zeros(R)
        self.base_n = np.zeros(R)

    def add_baseline(self, x, n):
        self.base_x += x
        self.base_n += n

    def freeze(self):
        # Jeffreys-type estimate keeps p0 inside (0, 1)
        self.p0 = (self.base_x + 0.5) / (self.base_n + 1.0)
        self.p1 = np.clip(self.p1_fn(self.p0), 1e-4, 1 - 1e-4)

    def update(self, x, n):
        llr = x * np.log(self.p1 / self.p0) + (n - x) * np.log((1 - self.p1) / (1 - self.p0))
        self.S = np.maximum(0.0, self.S + llr)
        self.max = np.maximum(self.max, self.S)
        return self.S


def simulate(p: Params, scenario: dict, monitor: str, R: int = 200, seed: int = 0,
             thresholds: dict | None = None, record: bool = False) -> dict:
    """Run R replicates of one scenario under one monitoring strategy.

    thresholds: dict stream -> threshold (array (R,) or scalar). If None, no
    alarms are raised (used for threshold calibration); the per-replicate
    maximum of each CUSUM over the monitoring period is returned instead.
    """
    rng = np.random.default_rng(seed)
    M = Model(p)
    cells = M.cells
    C = cells.w.size
    n = p.cases_per_day
    Tn = p.horizon_days
    streams = STREAMS[monitor]

    # ---- state -------------------------------------------------------------
    T = np.full(R, M.T_eq)
    U = np.full(R, M.u_eq)
    fixA = np.zeros(R)      # drift level absorbed by the last model remediation
    fixB = np.zeros(R)      # workload level absorbed by the last behavioural remediation
    pendA = np.full(R, np.inf)
    pendB = np.full(R, np.inf)
    first_alarm = {k: np.full(R, np.nan) for k in ("A", "B", "any")}
    n_false = np.zeros(R)

    # ---- monitors ----------------------------------------------------------
    alt = {
        "rep_fn": lambda q: q + p.shift_fn_abs,
        "truth_fn": lambda q: q + p.shift_fn_abs,
        "conc_up": lambda q: q * p.shift_conc_ratio,
        "conc_down": lambda q: q / p.shift_conc_ratio,
        "aud_err": lambda q: q + p.shift_ai_err_abs,
        "aud_miss": lambda q: q + p.shift_miss_abs,
        "sent_miss": lambda q: q + p.shift_miss_abs,
    }
    if monitor == "dual_audit" and p.sentinels_per_day <= 0:
        streams = ("aud_err", "aud_miss")
    charts = {s: Cusum(R, alt[s]) for s in streams}
    week = {s: [np.zeros(R), np.zeros(R)] for s in streams}
    truth_buffer = {}

    # ---- base cell state for decomposition ---------------------------------
    e_b, c_b, o_b, _, _ = M.baseline_state()
    R_base = float(np.sum(cells.w * cells.S * p_final_wrong(e_b[0], c_b[0], o_b[0])))

    risk_t = np.empty((Tn, R))
    rec = {k: np.empty(Tn) for k in ("v", "T", "D", "e_mean", "conc", "algorithmic",
                                      "behavioural", "interaction")} if record else None

    pos = cells.pos
    for t in range(Tn):
        # scheduled remediation
        a_raw = _ramp(t, p.onset_day, p.drift_ramp_days) * float(scenario["ai_drift"])
        W_raw = _ramp(t, p.onset_day, p.workload_ramp_days) * float(scenario["workload"])
        doA = pendA <= t
        if doA.any():
            fixA[doA] = a_raw
            pendA[doA] = np.inf
        doB = pendB <= t
        if doB.any():
            fixB[doB] = W_raw
            T[doB] = M.T_eq
            U[doB] = M.u_eq + 0.5 * (U[doB] - M.u_eq)
            pendB[doB] = np.inf
        a_lvl = np.clip(a_raw - fixA, 0, 1)
        W = np.clip(W_raw - fixB, 0, 1)

        e, c, o, h, v = M.cell_state(a_lvl, W, T, U)
        pw = p_final_wrong(e, c, o)
        risk_t[t] = np.sum(cells.w * cells.S * pw, axis=1)

        if record:
            dec = decompose(cells, e_b, c_b, o_b, e, c, o)
            rec["v"][t] = v[:, 0].mean()
            rec["T"][t] = T.mean()
            rec["D"][t] = M.deskill_index(U).mean()
            rec["e_mean"][t] = np.sum(cells.w * e, axis=1).mean()
            rec["conc"][t] = (1 - np.sum(cells.w * (e * c + (1 - e) * o), axis=1)).mean()
            for k in ("algorithmic", "behavioural", "interaction"):
                rec[k][t] = dec[k].mean()

        # ---- observable data ----------------------------------------------
        Ncell = rng.multinomial(n, cells.w, size=R)                       # (R, C)
        probs = np.stack([e * (1 - c), e * c, (1 - e) * o, (1 - e) * (1 - o)], axis=-1)
        J = rng.multinomial(Ncell, probs)                                   # (R, C, 4): AIw-Fw, AIw-Fr, AIr-Fw, AIr-Fr

        for s in streams:
            if s == "rep_fn":
                # among report-positive cases: AI negative?
                rep_pos_ai_neg = J[:, pos, 1].sum(1) + J[:, ~pos, 2].sum(1)
                rep_pos_ai_pos = J[:, pos, 3].sum(1) + J[:, ~pos, 0].sum(1)
                x, m = rep_pos_ai_neg, rep_pos_ai_neg + rep_pos_ai_pos
            elif s in ("conc_up", "conc_down"):
                x, m = J[:, :, 1].sum(1) + J[:, :, 2].sum(1), np.full(R, n)
            elif s == "truth_fn":
                ai_wrong = rng.binomial(J[:, pos, 0:2].sum(2), p.truth_completeness).sum(1)
                ai_right = rng.binomial(J[:, pos, 2:4].sum(2), p.truth_completeness).sum(1)
                truth_buffer[t + p.truth_lag_days] = (ai_wrong, ai_wrong + ai_right)
                x, m = truth_buffer.pop(t, (np.zeros(R), np.zeros(R)))
            elif s == "aud_err":
                A = rng.binomial(J, p.audit_fraction)
                x, m = A[:, :, 0:2].sum((1, 2)), A.sum((1, 2))
                aud_cache = A
            elif s == "aud_miss":
                A = aud_cache
                x, m = A[:, :, 0].sum(1), A[:, :, 0:2].sum((1, 2))
            elif s == "sent_miss":
                ns = rng.poisson(p.sentinels_per_day, size=R)
                k = M.sentinel_cell
                x, m = rng.binomial(ns, 1 - c[:, k]), ns
            week[s][0] += x
            week[s][1] += m

        # ---- behaviour updates ---------------------------------------------
        Nver = rng.binomial(n, v[:, 0])
        a_bar = M.agreement_if_verified(e, h)
        agree = rng.binomial(Nver, a_bar)
        a_obs = np.where(Nver > 0, agree / np.maximum(Nver, 1), a_bar)
        T = np.clip(T + p.trust_eta * ((1 - T) * a_obs - p.trust_lambda * T * (1 - a_obs)), 0, 1)
        U = U + (1.0 / p.deskill_tau_days) * ((1 - v[:, 0]) - U)

        # ---- weekly monitoring ---------------------------------------------
        if (t + 1) % 7 == 0:
            for s in streams:
                x, m = week[s]
                ch = charts[s]
                if t + 1 <= p.baseline_days:
                    ch.add_baseline(x, m)
                    if t + 1 + 7 > p.baseline_days:
                        ch.freeze()
                else:
                    S = ch.update(x, m)
                    if thresholds is not None:
                        hit = S > thresholds[s]
                        if hit.any():
                            act = STREAM_ACTION[s]
                            delay = p.concordance_investigation_days if act == "AB" else p.adjudication_days
                            for kind in ("A", "B"):
                                if kind in act:
                                    tgt = pendA if kind == "A" else pendB
                                    tgt[hit] = np.minimum(tgt[hit], t + delay)
                                    newly = hit & np.isnan(first_alarm[kind]) & (t >= p.onset_day)
                                    first_alarm[kind][newly] = t
                            newly_any = hit & np.isnan(first_alarm["any"]) & (t >= p.onset_day)
                            first_alarm["any"][newly_any] = t
                            n_false[hit & (t < p.onset_day)] += 1
                            ch.S[hit] = 0.0
                week[s] = [np.zeros(R), np.zeros(R)]

    out = {
        "risk": risk_t,
        "R_base": R_base,
        "first_alarm": first_alarm,
        "false_alarms_pre_onset": n_false,
        "cusum_max": {s: charts[s].max for s in streams},
        "T_eq": M.T_eq, "v_eq": M.v_eq,
    }
    if record:
        out["record"] = rec
    return out


def calibrate_thresholds(p: Params, monitor: str, R: int = 600, seed: int = 12345) -> dict:
    """Thresholds giving P(any false alarm over the monitoring period) = target in S0.

    A common per-stream quantile q is chosen by bisection so that the family of
    CUSUM charts of the strategy jointly meets the target false-alarm rate.
    """
    streams = STREAMS[monitor]
    if monitor == "dual_audit" and p.sentinels_per_day <= 0:
        streams = ("aud_err", "aud_miss")
    if not streams:
        return {}
    out = simulate(p, dict(ai_drift=False, workload=False), monitor, R=R, seed=seed, thresholds=None)
    mx = np.stack([out["cusum_max"][s] for s in streams], axis=1)    # (R, J)
    lo, hi = 0.5, 1.0
    for _ in range(40):
        q = 0.5 * (lo + hi)
        h = np.quantile(mx, q, axis=0)
        fa = np.mean((mx > h).any(axis=1))
        if fa > p.false_alarm_target:
            lo = q
        else:
            hi = q
    h = np.quantile(mx, hi, axis=0)
    return {s: max(float(h[j]), 1e-6) for j, s in enumerate(streams)}
