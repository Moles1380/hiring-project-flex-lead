from typing import Literal

import pandas as pd
import requests


def timestamp_to_str(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("CET").strftime("%Y-%m-%dT%H:%M")


def validate_response(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(response.text) from e


def get_spot_price(
    time_from: pd.Timestamp, time_to: pd.Timestamp, price_area: Literal["DK1", "DK2"]
) -> pd.DataFrame:

    url = "https://api.energidataservice.dk/dataset/DayAheadPrices"
    params = {
        "start": timestamp_to_str(time_from),
        "end": timestamp_to_str(time_to),
        "offset": 0,
        "filter": f"""{{"PriceArea": ["{price_area}"]}}""",
        "sort": "TimeUTC",
    }
    res = requests.get(url, params=params)
    validate_response(res)
    records = res.json()["records"]
    df = pd.json_normalize(records)
    df["TimeUTC"] = pd.to_datetime(df["TimeUTC"]).dt.tz_localize("UTC")
    df["TimeDK"] = pd.to_datetime(df["TimeDK"]).dt.tz_localize("CET", ambiguous="infer")
    return df


def get_forecasts(
    time_from: pd.Timestamp,
    time_to: pd.Timestamp,
    price_area: Literal["DK1", "DK2"],
    forecast_type: Literal["Solar", "Offshore Wind", "Onshore Wind"],
) -> pd.DataFrame:

    url = "https://api.energidataservice.dk/dataset/Forecasts_5Min"
    params = {
        "start": timestamp_to_str(time_from),
        "end": timestamp_to_str(time_to),
        "offset": 0,
        "filter": f"""{{"PriceArea": ["{price_area}"], "ForecastType": ["{forecast_type}"]}}""",
        "sort": "Minutes5DK",
    }
    res = requests.get(url, params=params)
    validate_response(res)
    records = res.json()["records"]
    df = pd.json_normalize(records)
    df["Minutes5UTC"] = pd.to_datetime(df["Minutes5UTC"]).dt.tz_localize("UTC")
    df["Minutes5DK"] = pd.to_datetime(df["Minutes5DK"]).dt.tz_localize(
        "CET", ambiguous="infer"
    )
    return df


def get_capacity_market_data(
    time_from: pd.Timestamp, time_to: pd.Timestamp, price_area: Literal["DK1", "DK2"]
) -> pd.DataFrame:

    url = "https://api.energidataservice.dk/dataset/AfrrReservesNordic"
    params = {
        "start": timestamp_to_str(time_from),
        "end": timestamp_to_str(time_to),
        "offset": 0,
        "filter": f"""{{"PriceArea": ["{price_area}"]}}""",
        "sort": "TimeUTC",
    }
    res = requests.get(url, params=params)
    validate_response(res)
    records = res.json()["records"]
    df = pd.json_normalize(records)
    df["TimeUTC"] = pd.to_datetime(df["TimeUTC"]).dt.tz_localize("UTC")
    df["TimeDK"] = pd.to_datetime(df["TimeDK"]).dt.tz_localize("CET", ambiguous="infer")
    return df


