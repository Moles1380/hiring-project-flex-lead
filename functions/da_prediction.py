import pandas as pd

from helpers.data import get_spot_price
from helpers.settings import PRICE_AREA

def forecast_da_prices(target_date: pd.Timestamp) -> pd.Series:
    """
    Naive DA price forecast for target_date.

    Fetches the last 3 days of cleared spot prices, forward-fills each day's
    hourly prices to 15-minute granularity, and returns the mean daily profile
    indexed by target_date's 15-min UTC timestamps.

    :param target_date: the day to forecast (timezone-aware)
    :return: Series of 96 forecast prices indexed by 15-min UTC timestamps
    """

    spot = get_spot_price(
        time_from=target_date - pd.DateOffset(days=2),
        time_to=target_date,
        price_area=PRICE_AREA,
    ).set_index("TimeUTC")["DayAheadPriceEUR"]

    # Forward-fill hourly prices to 15-min, then compute mean per 15-min slot across days
    spot_15min = spot.resample("15min").ffill()
    mean_profile = spot_15min.groupby(spot_15min.index.time).mean().values

    target_idx = target_date.tz_convert("UTC") + pd.timedelta_range(start="0min", periods=96, freq="15min")
    return pd.Series(mean_profile, index=target_idx, name="DayAheadPriceForecastEUR")
