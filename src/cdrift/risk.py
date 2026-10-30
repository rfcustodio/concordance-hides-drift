"""Analytic error-interception model of a human-AI reading workflow.

Every quantity is defined per *cell* (case stratum k x truth class y).  For a
cell we use

    e : probability that the AI output is wrong
    h : probability that the clinician's independent (unassisted) read is right
    v : probability that the clinician performs an independent verification
    rho_r : probability that, after verifying and disagreeing, the clinician
            keeps her own read when her read is right
    rho_w : the same probability when her read is wrong (rho_w < rho_r encodes
            informative arbitration; 1 - rho is anchoring on the AI)

Under conditional independence of the AI and clinician errors given the cell,

    c = v * h * rho_r          interception: P(final right | AI wrong)
    o = v * (1 - h) * rho_w    harmful override: P(final wrong | AI right)
    m = 1 - c                non-interception (miss) probability

    P(final wrong) = e * m + (1 - e) * o

Risk is the severity-weighted expected final error,

    R = sum_cells  w_cell * S_cell * [ e m + (1 - e) o ].

The module also implements the exact decomposition of R(t) - R(0) into an
algorithmic main effect, a behavioural main effect and an interaction term
(Proposition 3.1 of the manuscript), the concordance (agreement) between AI and
final report, and the report-referenced estimators that are biased by
deference (Propositions 3.3 and 3.4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cells:
    """Cell weights and severities, flattened over (stratum, truth).

    Attributes
    ----------
    w : array (K*2,) population weight of each cell (sums to 1)
    S : array (K*2,) severity of a final error in the cell
    pos : boolean array (K*2,) True where the truth is positive
    stratum : int array (K*2,) stratum index of each cell
    """

    w: np.ndarray
    S: np.ndarray
    pos: np.ndarray
    stratum: np.ndarray


def make_cells(stratum_weights, prevalence, severity_fn=1.0, severity_fp=0.1) -> Cells:
    sw = np.asarray(stratum_weights, dtype=float)
    sw = sw / sw.sum()
    K = sw.size
    w = np.empty(2 * K)
    S = np.empty(2 * K)
    pos = np.zeros(2 * K, dtype=bool)
    stratum = np.empty(2 * K, dtype=int)
    for k in range(K):
        w[2 * k] = sw[k] * prevalence
        w[2 * k + 1] = sw[k] * (1.0 - prevalence)
        S[2 * k], S[2 * k + 1] = severity_fn, severity_fp
        pos[2 * k] = True
        stratum[2 * k] = stratum[2 * k + 1] = k
    return Cells(w=w, S=S, pos=pos, stratum=stratum)


def interception(v, h, rho_r, rho_w=None):
    """Return (c, o): interception and harmful-override probabilities."""
    if rho_w is None:
        rho_w = rho_r
    v, h = np.broadcast_arrays(np.asarray(v, float), np.asarray(h, float))
    c = v * h * rho_r
    o = v * (1.0 - h) * rho_w
    return c, o


def p_final_wrong(e, c, o):
    e = np.asarray(e, float)
    return e * (1.0 - c) + (1.0 - e) * o


def risk(cells: Cells, e, c, o) -> np.ndarray:
    """Severity-weighted expected final error rate (per case).

    e, c, o may carry leading batch dimensions; the last axis indexes cells.
    """
    return np.sum(cells.w * cells.S * p_final_wrong(e, c, o), axis=-1)


def decompose(cells: Cells, e0, c0, o0, e1, c1, o1) -> dict:
    """Exact decomposition of R1 - R0 (Proposition 3.1).

    With m = 1 - c, de = e1 - e0, dm = m1 - m0, do = o1 - o0:

        R1 - R0 = E[S (m0 - o0) de]            algorithmic main effect
                + E[S e0 dm] + E[S (1-e0) do]   behavioural main effect
                + E[S de (dm - do)]             interaction

    The identity is exact (no approximation); see tests/test_risk.py.
    """
    m0, m1 = 1.0 - np.asarray(c0, float), 1.0 - np.asarray(c1, float)
    e0, e1, o0, o1 = (np.asarray(x, float) for x in (e0, e1, o0, o1))
    de, dm, do = e1 - e0, m1 - m0, o1 - o0
    ws = cells.w * cells.S
    out = {
        "R0": np.sum(ws * p_final_wrong(e0, c0, o0), axis=-1),
        "R1": np.sum(ws * p_final_wrong(e1, c1, o1), axis=-1),
        "algorithmic": np.sum(ws * (m0 - o0) * de, axis=-1),
        "behavioural": np.sum(ws * (e0 * dm + (1.0 - e0) * do), axis=-1),
        "interaction": np.sum(ws * de * (dm - do), axis=-1),
    }
    out["residual"] = out["R1"] - out["R0"] - out["algorithmic"] - out["behavioural"] - out["interaction"]
    return out


def disagreement(cells: Cells, e, c, o) -> np.ndarray:
    """P(final report != AI output) = E[e c + (1 - e) o]  (override rate)."""
    e = np.asarray(e, float)
    return np.sum(cells.w * (e * c + (1.0 - e) * o), axis=-1)


def concordance(cells: Cells, e, c, o) -> np.ndarray:
    return 1.0 - disagreement(cells, e, c, o)


def report_referenced_error(cells: Cells, e, c, o) -> np.ndarray:
    """AI error rate estimated with the (AI-influenced) final report as reference.

    Because the report differs from the AI only where the clinician overrode it,
    this estimator equals the override rate (Proposition 3.4).  Its bias relative to
    the true AI error rate E[e] is  -E[e (1 - c)] + E[(1 - e) o].
    """
    return disagreement(cells, e, c, o)


def true_ai_error(cells: Cells, e) -> np.ndarray:
    return np.sum(cells.w * np.asarray(e, float), axis=-1)


def report_referenced_sensitivity(cells: Cells, e, c, o) -> np.ndarray:
    """P(AI positive | final report positive): the 'AI vs report' sensitivity."""
    e = np.asarray(e, float)
    pos = cells.pos
    # final positive & AI positive
    tp_ai_right = np.where(pos, (1 - e) * (1 - o), 0.0)       # true pos, AI pos, kept
    fp_ai_wrong = np.where(~pos, e * (1 - c), 0.0)            # true neg, AI pos, not intercepted
    # final positive & AI negative
    tp_ai_wrong = np.where(pos, e * c, 0.0)                   # true pos, AI neg, intercepted
    fp_ai_right = np.where(~pos, (1 - e) * o, 0.0)            # true neg, AI neg, overridden to pos
    num = np.sum(cells.w * (tp_ai_right + fp_ai_wrong), axis=-1)
    den = num + np.sum(cells.w * (tp_ai_wrong + fp_ai_right), axis=-1)
    return num / den


def true_sensitivity(cells: Cells, e) -> np.ndarray:
    e = np.asarray(e, float)
    wp = np.where(cells.pos, cells.w, 0.0)
    return np.sum(wp * (1 - e), axis=-1) / wp.sum()
