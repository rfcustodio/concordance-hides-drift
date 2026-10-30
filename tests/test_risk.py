import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cdrift import Params, sim, SCENARIOS  # noqa: E402
from cdrift import risk as rk  # noqa: E402

rng = np.random.default_rng(0)
cells = rk.make_cells((0.5, 0.35, 0.15), 0.1)


def rand_state(n=1000):
    C = cells.w.size
    e = rng.uniform(0, 0.5, (n, C))
    v = rng.uniform(0, 1, (n, C))
    h = rng.uniform(0.5, 1, (n, C))
    c, o = rk.interception(v, h, 0.85, 0.35)
    return e, c, o


def test_decomposition_is_exact():
    e0, c0, o0 = rand_state()
    e1, c1, o1 = rand_state()
    d = rk.decompose(cells, e0, c0, o0, e1, c1, o1)
    assert np.abs(d["residual"]).max() < 1e-12


def test_interaction_is_supermodular_in_matching_cells():
    # increasing AI error and non-interception on the same cells gives I >= 0
    e0, c0, o0 = rand_state()
    e1 = np.clip(e0 + 0.1, 0, 1)
    c1, o1 = 0.5 * c0, 0.5 * o0
    d = rk.decompose(cells, e0, c0, o0, e1, c1, o1)
    assert (d["interaction"] >= -1e-15).all()


def test_report_referenced_error_equals_override_rate():
    e, c, o = rand_state()
    assert np.allclose(rk.report_referenced_error(cells, e, c, o), rk.disagreement(cells, e, c, o))


def test_missed_errors_identity_without_harmful_override():
    # with o = 0:  E[e (1 - c)] = E[e] - override rate   (Proposition 3.3)
    e, c, _ = rand_state()
    o = np.zeros_like(e)
    missed = np.sum(cells.w * e * (1 - c), axis=-1)
    assert np.allclose(missed, rk.true_ai_error(cells, e) - rk.disagreement(cells, e, c, o))


def test_report_sensitivity_inflated_when_verification_falls():
    p = Params()
    M = sim.Model(p)
    e = np.clip(M.e0 + M.de, 0, 1)[None, :]
    vals = []
    for v in (0.9, 0.5, 0.1):
        c, o = rk.interception(np.full(e.shape, v), np.broadcast_to(M.h0, e.shape), p.rho_r, p.rho_w)
        vals.append(rk.report_referenced_sensitivity(cells_for(p), e, c, o)[0])
    assert vals[0] < vals[1] < vals[2]
    assert vals[2] > rk.true_sensitivity(cells_for(p), e)[0]


def cells_for(p):
    return rk.make_cells(p.stratum_weights, p.prevalence, p.severity_fn, p.severity_fp)


def test_null_scenario_is_stationary():
    p = Params()
    out = sim.simulate(p, SCENARIOS["S0_none"], "none", R=20, seed=1)
    r = out["risk"].mean(1)
    assert abs(r[-1] - out["R_base"]) / out["R_base"] < 0.02


def test_calibration_meets_false_alarm_target():
    p = Params()
    th = sim.calibrate_thresholds(p, "concordance", R=300, seed=5)
    out = sim.simulate(p, SCENARIOS["S0_none"], "concordance", R=300, seed=99, thresholds=th)
    any_alarm = ~np.isnan(out["first_alarm"]["any"]) | (out["false_alarms_pre_onset"] > 0)
    assert any_alarm.mean() < 0.12
