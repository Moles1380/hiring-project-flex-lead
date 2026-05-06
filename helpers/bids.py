import matplotlib.pyplot as plt
import pandas as pd

from helpers.data import get_forecasts, get_spot_price
from helpers.settings import BESS_ENERGY, CYCLE_COST


def make_afrr_bid_matrix(afrr_bids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the aFRR optimisation output into separate up- and down-regulation bid matrices.

    :param afrr_bids: DataFrame with columns UpVolumeMW, UpBidPriceEUR,
        DownVolumeMW, DownBidPriceEUR, indexed by hourly UTC timestamps.
    :return: (up_matrix, dn_matrix) — each indexed by hour, columns are EUR/MW price
        levels, values are MW offered. Positive values propagate rightward (higher prices).
    """
    up_mask = afrr_bids["UpVolumeMW"] > 0
    dn_mask = afrr_bids["DownVolumeMW"] > 0

    up_prices = sorted(afrr_bids.loc[up_mask, "UpBidPriceEUR"].round(2).unique())
    dn_prices = sorted(afrr_bids.loc[dn_mask, "DownBidPriceEUR"].round(2).unique())

    def _build(prices, vol_col, price_col):
        if not prices:
            return pd.DataFrame(0.0, index=afrr_bids.index, columns=[0.0])
        m = pd.DataFrame(0.0, index=afrr_bids.index, columns=prices)
        m.columns.name = "PriceEUR_per_MW"
        for ts, row in afrr_bids.iterrows():
            if row[vol_col] > 0:
                m.loc[ts, round(row[price_col], 2)] += row[vol_col]
        return m.where(m > 0).ffill(axis=1).fillna(m)

    up_matrix = _build(up_prices, "UpVolumeMW", "UpBidPriceEUR")
    dn_matrix = _build(dn_prices, "DownVolumeMW", "DownBidPriceEUR")

    return up_matrix, dn_matrix


def make_day_ahead_bids(
    sell: pd.DataFrame,
    buy: pd.DataFrame,
) -> pd.DataFrame:
    """Simple function to turn buy and sell timeseries into a bid matrix.

    :param sell: pandas dataframe with one column per sell order. The column name is the minimum price
        we want to sell that volume for.
    :param buy: pandas dataframe with one column per buy order. The column name is the maximum price
        we want to buy that volume for.
    :return: pandas dataframe with bids
    """

    # we sell nothing below min_sell_price
    sell[sell.columns.min() - 0.01] = 0
    # cumulate volume
    sell = sell.sort_index(axis=1, ascending=True).cumsum(axis=1)

    # we buy nothing above max_buy_price
    buy[buy.columns.max() + 0.01] = 0
    buy = buy.sort_index(axis=1, ascending=False).cumsum(axis=1)

    price_levels = sorted(sell.columns.tolist() + buy.columns.tolist())
    # we buy the same volume for any price below a given level expressed in the columns
    buy = buy.reindex(price_levels, axis=1).bfill(axis=1).fillna(0)
    # we sell the same volume for any price above a given level expressed in the columns
    sell = sell.reindex(price_levels, axis=1).bfill(axis=1).fillna(0)

    bids = sell.abs() - buy.abs()
    bids.columns = bids.columns.round(2)
    return bids


def make_da_bid_matrix(schedule: pd.DataFrame, spot_forecast: pd.Series) -> pd.DataFrame:
    """Convert a DA schedule DataFrame into a price/volume bid matrix.

    :param schedule: DataFrame with columns SolarMW, DischargeMW, ChargeMW
        (as returned by optimize_da_schedule), indexed by 15-min UTC timestamps.
    :param spot_forecast: 15-min spot price forecast used to determine the
        charging bid price.
    :return: bid matrix from make_day_ahead_bids (price levels × time).
    """
    spot = spot_forecast.values.astype(float)
    idx = schedule.index

    sell = pd.DataFrame(
        {0.0: schedule["SolarMW"].copy(), float(CYCLE_COST): schedule["DischargeMW"].copy()},
        index=idx,
    )

    charging = schedule["ChargeMW"] > 0.1
    if charging.any():
        charge_max_price = float(spot[charging.values].max())
        buy = pd.DataFrame({charge_max_price: schedule["ChargeMW"].copy()}, index=idx)
    else:
        buy = pd.DataFrame({0.0: pd.Series(0.0, index=idx)})

    return make_day_ahead_bids(sell, buy)


def plot_da_schedule(schedule: pd.DataFrame, spot_forecast: pd.Series) -> None:
    """Plot the DA schedule: solar as a line, discharge/charge as bars, spot price on secondary axis."""
    bar_width = pd.Timedelta("14min")
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(schedule.index, schedule["DischargeMW"], width=bar_width,
           label="Discharge (MW)", color="steelblue", alpha=0.8)
    ax.bar(schedule.index, -schedule["ChargeMW"], width=bar_width,
           label="Charge (MW)", color="tomato", alpha=0.8)
    ax.plot(schedule.index, schedule["SolarMW"],
            label="Solar (MW)", color="gold", linewidth=2)

    ax2 = ax.twinx()
    ax2.plot(spot_forecast.index, spot_forecast.values,
             label="DA Price (EUR/MWh)", color="green", linewidth=1.5, linestyle="--", alpha=0.7)
    ax2.set_ylabel("Price (EUR/MWh)")

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.08))
    ax3.plot(schedule.index, schedule["SoCMWh"] / BESS_ENERGY * 100,
             label="SoC (%)", color="purple", linewidth=1.5, linestyle=":")
    ax3.set_ylabel("SoC (%)")
    ax3.set_ylim(0, 100)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("MW  [+ discharge / − charge]")
    ax.set_title("Day-Ahead Schedule")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper left")


    fig.tight_layout()
    plt.show()


def example():
    """Simple example to illustrate how to create bids.
    We get the solar forecast for today, rescale it a bit to have a realistic level.
    We then decide to charge in the 8 lowest priced settlement periods and discharge in the highest.
    Note: this is clearly not a realistic strategy as we are assuming perfect foresight of prices
    and completely ignoring state of charge.
    """
    today = pd.Timestamp("now", tz="CET").normalize()
    date_range = pd.date_range(
        start=today, end=today + pd.DateOffset(days=1), freq="15min", inclusive="left"
    )

    forecast = (
        get_forecasts(
            time_from=date_range[0],
            time_to=date_range[-1] + pd.Timedelta("1h"),
            price_area="DK1",
            forecast_type="Solar",
        )
        .set_index("Minutes5UTC")["ForecastDayAhead"]
        .resample("15min")
        .mean()
        .reindex(date_range)
        .fillna(0)
        / 2000  # rescale
        * 100
    ).clip(upper=100)
    spot = get_spot_price(
        time_from=date_range[0],
        time_to=date_range[-1] + pd.Timedelta("1h"),
        price_area="DK1",
    ).set_index("TimeUTC")["DayAheadPriceEUR"]

    # we discharge in the highest 8 settlement periods
    discharge_threshold = spot.nlargest(8).min()
    charge_threshold = spot.nsmallest(8).max()
    bess_discharge = (spot >= discharge_threshold) * 50
    # we charge in the lowest 8 settlement periods
    bess_charge = (spot <= charge_threshold) * 50
    # we sell the forecasted solar production for a minimum of 1.5 EUR
    sell = pd.DataFrame({1.5: forecast, discharge_threshold: bess_discharge})
    buy = pd.DataFrame({charge_threshold: bess_charge})
    bids = make_day_ahead_bids(sell, buy)

    print(bids)


if __name__ == "__main__":
    example()
