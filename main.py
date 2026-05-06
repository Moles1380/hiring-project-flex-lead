"""
Trading strategy for 100 MW PV + 50 MW / 200 MWh BESS hybrid asset in DK1.

Pipeline
────────
1. Fetch solar forecast, spot price forecast, and aFRR capacity price forecast.
2. Stage 1 (07:30 D-1): joint LP determines optimal aFRR capacity bids by trading
   off capacity revenue against the DA opportunity cost of reserving BESS headroom.
3. Stage 2 (12:00 D-1): DA LP re-optimises the schedule with the aFRR commitments
   fixed as hard availability constraints.
4. Print and return both bid matrices.
"""

import pandas as pd

from helpers.settings import BESS_ENERGY
from functions.afrr_prediction import forecast_afrr_prices
from functions.da_prediction import forecast_da_prices
from functions.solar_forecast import forecast_solar
from functions.optimization import optimize_afrr_bids, optimize_da_schedule
from helpers.bids import make_afrr_bid_matrix, make_da_bid_matrix, plot_da_schedule


def co_location_optimization() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    target_date = pd.Timestamp("now", tz="CET").normalize() + pd.DateOffset(days=1)
    print(f"Generating bids for {target_date.date()}")

    print("  Fetching solar forecast …")
    solar_forecast = forecast_solar(target_date)

    print("  Fetching spot price forecast …")
    spot_forecast = forecast_da_prices(target_date)

    print("  Fetching aFRR capacity price forecast …")
    afrr_capacity_forecast = forecast_afrr_prices(target_date)

    print("  Stage 1: optimising aFRR capacity bids …")
    afrr_bids = optimize_afrr_bids(solar_forecast, spot_forecast, afrr_capacity_forecast)

    print(
        f"    aFRR up:   {afrr_bids['UpVolumeMW'].sum():.0f} MW·h reserved, "
        f"avg price {afrr_bids.loc[afrr_bids['UpVolumeMW'] > 0, 'UpBidPriceEUR'].mean():.1f} EUR/MW"
        if (afrr_bids["UpVolumeMW"] > 0).any() else "    aFRR up:   no bids"
    )
    print(
        f"    aFRR down: {afrr_bids['DownVolumeMW'].sum():.0f} MW·h reserved, "
        f"avg price {afrr_bids.loc[afrr_bids['DownVolumeMW'] > 0, 'DownBidPriceEUR'].mean():.1f} EUR/MW"
        if (afrr_bids["DownVolumeMW"] > 0).any() else "    aFRR down: no bids"
    )

    print("  Stage 2: optimising day-ahead schedule …")
    da_schedule = optimize_da_schedule(solar_forecast, spot_forecast, afrr_bids)
    #da_bids = make_da_bid_matrix(da_schedule, spot_forecast)

    afrr_up_matrix, afrr_dn_matrix = make_afrr_bid_matrix(afrr_bids)

    print("\n=== aFRR Up-Regulation Bid Matrix (EUR/MW × hour) ===")
    print(afrr_up_matrix.to_string())
    print("\n=== aFRR Down-Regulation Bid Matrix (EUR/MW × hour) ===")
    print(afrr_dn_matrix.to_string())
    da_cycles = da_schedule["DischargeMW"].sum() * 0.25 / BESS_ENERGY
    afrr_cycles = (afrr_bids["UpVolumeMW"].sum() + afrr_bids["DownVolumeMW"].sum()) / BESS_ENERGY
    print(f"\n  Battery cycles: DA = {da_cycles:.2f}, aFRR (assumed full activation) = {afrr_cycles:.2f}, total = {da_cycles + afrr_cycles:.2f}")

    print("\n=== Day-Ahead Schedule (15-min) ===")
    print(da_schedule.to_string())

    plot_da_schedule(da_schedule, spot_forecast)

    return afrr_up_matrix, afrr_dn_matrix, da_schedule


if __name__ == "__main__":
    co_location_optimization()
