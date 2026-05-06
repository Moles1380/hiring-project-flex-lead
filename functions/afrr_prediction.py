import pandas as pd

from helpers.data import get_capacity_market_data
from helpers.settings import PRICE_AREA

def forecast_afrr_prices(target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Naive aFRR price forecast for target_date.

    Fetches the last 3 days of AFR capacity prices, and returns the mean daily profile

    :param target_date: the day to forecast (timezone-aware)
    :return: forecast aFRR capacity prices (up and down) on an hourly granularity
    """

    prices = get_capacity_market_data(
        time_from=target_date - pd.DateOffset(days=2),
        time_to=target_date,
        price_area=PRICE_AREA,
    ).set_index("TimeUTC")[["UpPriceEUR", "DownPriceEUR"]]

    mean_profile = prices.groupby(prices.index.time).mean()
    target_idx = target_date.tz_convert("UTC") + pd.timedelta_range(start="0min", periods=24, freq="60min")

    return pd.DataFrame(
        mean_profile.values,
        index=target_idx,
        columns=["UpPriceForecast", "DownPriceForecast"]
    )
