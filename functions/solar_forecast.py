from helpers.data import get_forecasts
import pandas as pd


def forecast_solar(target_date: pd.Timestamp) -> pd.Series:

    date_range = pd.date_range(
        start=target_date-pd.DateOffset(days=3), end=target_date, freq="15min", inclusive="left"
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

    mean_profile = forecast.groupby(forecast.index.time).mean().values
    target_idx = target_date.tz_convert("UTC") + pd.timedelta_range(start="0min", periods=96, freq="15min")

    return pd.Series(mean_profile, index=target_idx, name="SolarForecastMW")
