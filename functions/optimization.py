"""
Two-stage optimization for the hybrid PV + BESS asset.

Stage 1 (07:30 D-1): joint LP finds the optimal aFRR capacity bids by trading off
    aFRR capacity revenue against the DA opportunity cost of reserving BESS headroom.

Stage 2 (12:00 D-1): DA LP re-optimizes the schedule with the won aFRR commitments
    added as hard availability constraints.
"""

import numpy as np
import pandas as pd
import pulp

from helpers.settings import (
    BESS_ENERGY, BESS_POWER, CYCLE_COST, GRID_MW, INITIAL_SOC, MAX_CYCLES
)

_DT = 0.25   # hours per 15-min period
_T  = 96     # 15-min periods per day
_H  = 24     # hours per day

# Prefer HiGHS; fall back to CBC if not installed
_highs = pulp.HiGHS(msg=False)
_SOLVER = _highs if _highs.available() else pulp.PULP_CBC_CMD(msg=0)


def optimize_afrr_bids(
    solar_forecast: pd.Series,
    spot_forecast: pd.Series,
    afrr_capacity_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stage 1: joint DA + aFRR LP to determine optimal aFRR capacity bids.

    The LP maximises DA revenue plus aFRR capacity payments. Reserving BESS or grid
    headroom for aFRR reduces available DA output, creating an implicit opportunity cost.
    The LP only commits capacity in hours where the forecast aFRR price exceeds that cost.

    Bid price = aFRR forecast clearing price (at optimality, the LP sets volume > 0 exactly
    when the forecast price equals the opportunity cost, so bidding at the forecast price is
    the theoretically correct threshold).

    Returns
    -------
    pd.DataFrame  hourly, indexed like afrr_capacity_forecast, columns:
        UpVolumeMW, UpBidPriceEUR, DownVolumeMW, DownBidPriceEUR
    """
    solar    = solar_forecast.values.astype(float)
    spot     = spot_forecast.values.astype(float)
    up_price = afrr_capacity_forecast["UpPriceForecast"].values.astype(float)
    dn_price = afrr_capacity_forecast["DownPriceForecast"].values.astype(float)

    prob = pulp.LpProblem("stage1_joint", pulp.LpMaximize)

    p_sol = [pulp.LpVariable(f"p_sol_{t}", 0, solar[t]) for t in range(_T)]
    p_dis = [pulp.LpVariable(f"p_dis_{t}", 0, BESS_POWER) for t in range(_T)]
    p_chg = [pulp.LpVariable(f"p_chg_{t}", 0, BESS_POWER) for t in range(_T)]
    soc   = [pulp.LpVariable(f"soc_{t}",   0, BESS_ENERGY) for t in range(_T + 1)]
    up_mw = [pulp.LpVariable(f"up_{h}",    0, BESS_POWER) for h in range(_H)]
    dn_mw = [pulp.LpVariable(f"dn_{h}",    0, BESS_POWER) for h in range(_H)]

    # ── Objective ────────────────────────────────────────────────────────────
    prob += (
        pulp.lpSum(
            (p_sol[t] + p_dis[t] - p_chg[t]) * spot[t] * _DT
            - p_dis[t] * _DT * CYCLE_COST
            for t in range(_T)
        )
        + pulp.lpSum(
            up_mw[h] * (up_price[h] - CYCLE_COST) + dn_mw[h] * (dn_price[h] - CYCLE_COST)
            for h in range(_H)
        )
    )

    # ── SoC dynamics (DA moves only — aFRR activations not modelled) ─────────
    prob += soc[0] == INITIAL_SOC
    for t in range(_T):
        prob += soc[t + 1] == soc[t] + (p_chg[t] - p_dis[t]) * _DT

    # ── Cycle limit ───────────────────────────────────────────────────────────
    # Up-regulation: assumed activation discharges up_mw[h] MWh → direct cycle usage.
    # Down-regulation: assumed activation charges dn_mw[h] MWh; that energy must be
    # discharged later (same day or next), which also consumes cycle budget.
    prob += (
        pulp.lpSum(p_dis[t] * _DT for t in range(_T))
        + pulp.lpSum(up_mw[h] + dn_mw[h] for h in range(_H))
    ) <= MAX_CYCLES * BESS_ENERGY

    # ── Grid limits (base) ────────────────────────────────────────────────────
    for t in range(_T):
        prob += p_sol[t] + p_dis[t] - p_chg[t] <=  GRID_MW   # max export
        prob += p_chg[t] - p_dis[t] - p_sol[t] <=  GRID_MW   # max import

    # ── aFRR availability constraints ─────────────────────────────────────────
    # For each hour h, the BESS must hold back enough power headroom (both BESS
    # rating and grid connection) so it CAN respond if activated.  Solar occupies
    # the shared 95 MW grid connection, so high solar shrinks up-reg headroom.
    # Energy reserve constraints ensure there is enough stored energy / empty space
    # to sustain a full hour of activation; they are availability checks only —
    # no energy is consumed by activations in this model.
    for h in range(_H):
        t0 = h * 4
        for t in range(t0, t0 + 4):
            # Up-regulation headroom
            prob += p_dis[t] - p_chg[t] + up_mw[h] <= BESS_POWER          # BESS rating
            prob += p_sol[t] + p_dis[t] - p_chg[t] + up_mw[h] <= GRID_MW  # grid (solar-aware)
            # Down-regulation headroom
            prob += p_chg[t] - p_dis[t] + dn_mw[h] <= BESS_POWER          # BESS rating
            prob += p_sol[t] + p_dis[t] - p_chg[t] - dn_mw[h] >= -GRID_MW # grid import

        # Energy / space reserve at start of hour (availability check)
        prob += soc[t0] >= up_mw[h]                     # enough energy for 1-h up activation
        prob += soc[t0] <= BESS_ENERGY - dn_mw[h]       # enough space for 1-h down activation

    prob.solve(_SOLVER)
    if prob.status != 1:
        raise RuntimeError(f"Stage 1 LP infeasible: {pulp.LpStatus[prob.status]}")

    up_vol = np.array([pulp.value(up_mw[h]) or 0.0 for h in range(_H)])
    dn_vol = np.array([pulp.value(dn_mw[h]) or 0.0 for h in range(_H)])

    # Round to 1 MW granularity; zero out sub-0.5 MW (below minimum bid size)
    up_vol = np.floor(np.where(up_vol >= 0.5, up_vol, 0.0))
    dn_vol = np.floor(np.where(dn_vol >= 0.5, dn_vol, 0.0))

    return pd.DataFrame(
        {
            "UpVolumeMW":    up_vol,
            "UpBidPriceEUR": up_price,
            "DownVolumeMW":  dn_vol,
            "DownBidPriceEUR": dn_price,
        },
        index=afrr_capacity_forecast.index,
    )


def optimize_da_schedule(
    solar_forecast: pd.Series,
    spot_forecast: pd.Series,
    afrr_bids: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stage 2: DA schedule optimisation with aFRR commitments as hard constraints.

    For each hour where capacity was bid (volume > 0), the BESS must keep the
    corresponding power and energy headroom free throughout all four 15-min periods
    of that hour.

    Returns
    -------
    pd.DataFrame  15-min DA bid matrix (price × volume) via make_day_ahead_bids
    """
    solar  = solar_forecast.values.astype(float)
    spot   = spot_forecast.values.astype(float)
    up_res = afrr_bids["UpVolumeMW"].values.astype(float)
    dn_res = afrr_bids["DownVolumeMW"].values.astype(float)
    idx    = spot_forecast.index

    prob = pulp.LpProblem("stage2_da", pulp.LpMaximize)

    p_sol = [pulp.LpVariable(f"p_sol_{t}", 0, solar[t]) for t in range(_T)]
    p_dis = [pulp.LpVariable(f"p_dis_{t}", 0, BESS_POWER) for t in range(_T)]
    p_chg = [pulp.LpVariable(f"p_chg_{t}", 0, BESS_POWER) for t in range(_T)]
    soc   = [pulp.LpVariable(f"soc_{t}",   0, BESS_ENERGY) for t in range(_T + 1)]

    prob += pulp.lpSum(
        (p_sol[t] + p_dis[t] - p_chg[t]) * spot[t] * _DT - p_dis[t] * _DT * CYCLE_COST
        for t in range(_T)
    )

    prob += soc[0] == INITIAL_SOC
    for t in range(_T):
        prob += soc[t + 1] == soc[t] + (p_chg[t] - p_dis[t]) * _DT

    prob += pulp.lpSum(p_dis[t] * _DT for t in range(_T)) <= MAX_CYCLES * BESS_ENERGY - float(up_res.sum()) - float(dn_res.sum())

    for t in range(_T):
        prob += p_sol[t] + p_dis[t] - p_chg[t] <=  GRID_MW
        prob += p_chg[t] - p_dis[t] - p_sol[t] <=  GRID_MW

    for h in range(_H):
        up, dn = up_res[h], dn_res[h]
        t0 = h * 4
        for t in range(t0, t0 + 4):
            prob += p_dis[t] - p_chg[t]             <= BESS_POWER - up
            prob += p_sol[t] + p_dis[t] - p_chg[t] <= GRID_MW    - up
            prob += p_chg[t]                         <= solar[t] + GRID_MW - dn  # co-location: solar + grid capacity minus aFRR reserve
            prob += p_sol[t] + p_dis[t] - p_chg[t] >= -GRID_MW   + dn
        prob += soc[t0] >= up
        prob += soc[t0] <= BESS_ENERGY - dn

    prob.solve(_SOLVER)
    if prob.status != 1:
        raise RuntimeError(f"Stage 2 LP infeasible: {pulp.LpStatus[prob.status]}")

    return pd.DataFrame(
        {
            "SolarMW":     [pulp.value(v) or 0.0 for v in p_sol],
            "DischargeMW": [max(0.0, pulp.value(v) or 0.0) for v in p_dis],
            "ChargeMW":    [max(0.0, pulp.value(v) or 0.0) for v in p_chg],
            "SoCMWh":      [pulp.value(soc[t]) or 0.0 for t in range(_T)],
        },
        index=idx,
    )
