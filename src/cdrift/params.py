"""Default parameters of the reading-workflow simulation.

All values are illustrative and chosen to be plausible for a binary imaging
finding read with AI assistance (e.g. a chest-radiograph or screening finding).
They are NOT estimates from any specific deployment.  Every experiment in
``experiments/`` records the full parameter set it used next to its results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json


@dataclass(frozen=True)
class Params:
    # --- case mix ------------------------------------------------------------
    stratum_weights: tuple = (0.50, 0.35, 0.15)      # easy, intermediate, hard
    prevalence: float = 0.10
    severity_fn: float = 1.0                          # missed finding
    severity_fp: float = 0.1                          # false positive (recall)
    cases_per_day: int = 150

    # --- AI at validation (t = 0) -------------------------------------------
    se_ai0: tuple = (0.97, 0.90, 0.72)
    sp_ai0: tuple = (0.97, 0.93, 0.85)

    # --- algorithmic drift (e.g. new acquisition protocol) -------------------
    d_se_ai: tuple = (0.05, 0.20, 0.25)               # loss of sensitivity at full drift
    d_sp_ai: tuple = (0.01, 0.03, 0.05)
    drift_ramp_days: int = 180

    # --- clinician (unassisted) ---------------------------------------------
    se_h0: tuple = (0.95, 0.86, 0.66)
    sp_h0: tuple = (0.97, 0.93, 0.86)
    rho_r: float = 0.85                               # keeps own (correct) read when disagreeing
    rho_w: float = 0.35                               # keeps own (wrong) read when disagreeing
    v_max: float = 0.95                               # verification with zero trust
    v_min: float = 0.15                               # verification with full trust

    # --- trust dynamics (asymmetric, Lee & See style) ------------------------
    trust_eta: float = 0.02                           # per day
    trust_lambda: float = 4.0                         # weight of an observed disagreement

    # --- workload-driven behavioural drift ----------------------------------
    workload_effect: float = 0.6                      # fractional loss of verification at full workload
    workload_ramp_days: int = 180

    # --- deskilling ----------------------------------------------------------
    d_se_h: tuple = (0.02, 0.08, 0.12)                # loss of unassisted sensitivity at full deskilling
    d_sp_h: tuple = (0.01, 0.03, 0.05)
    deskill_tau_days: float = 180.0

    # --- timeline ------------------------------------------------------------
    horizon_days: int = 1095
    onset_day: int = 365
    baseline_days: int = 182                          # monitoring baseline is estimated on [0, baseline_days)

    # --- monitoring ----------------------------------------------------------
    audit_fraction: float = 0.03                      # AI-blinded expert double read
    sentinels_per_day: float = 0.0                    # projected AI-error cases (TIP-style)
    truth_lag_days: int = 90                          # delayed reference (pathology / follow-up)
    truth_completeness: float = 0.8
    adjudication_days: int = 30
    concordance_investigation_days: int = 60
    false_alarm_target: float = 0.05                  # P(any false alarm over the monitoring period)

    # CUSUM design shifts (out-of-control alternatives)
    shift_fn_abs: float = 0.05                        # report- or truth-referenced AI FN rate
    shift_conc_ratio: float = 1.5                     # disagreement rate up or down
    shift_ai_err_abs: float = 0.03                    # audited AI error rate
    shift_miss_abs: float = 0.15                      # non-interception among AI errors

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


SCENARIOS = {
    "S0_none": dict(ai_drift=False, workload=False),
    "S1_algorithmic": dict(ai_drift=True, workload=False),
    "S2_behavioural": dict(ai_drift=False, workload=True),
    "S3_coupled": dict(ai_drift=True, workload=True),
}

MONITORS = ("none", "report_referenced", "concordance", "delayed_truth", "dual_audit")
