"""
Trading strategy for 100 MW PV + 50 MW / 200 MWh BESS hybrid asset in DK1.

Pipeline
────────
1. Creating a settings file in the helpers folder
2. Creating an estimated forecast for the solar production
3. Creating simplified predictions for the day ahead and aFRR capacity prices

"""

import pandas as pd

from functions.da_prediction import forecast_da_prices
from functions.afrr_prediction import forecast_afrr_prices
from functions.solar_forecast import forecast_solar




def co_location_optimization() -> tuple[pd.DataFrame, pd.DataFrame]:

    print("  Fetching data")
    target_date = pd.Timestamp("now", tz="CET").normalize() + pd.DateOffset(days=1)

    print("  Creating solar forecast")
    solar_forecast = forecast_solar(target_date)

    print("  Creating forecasts")
    spot_forecast = forecast_da_prices(target_date)
    afrr_capacity_forecast = forecast_afrr_prices(target_date)

    print("  Optimizing the bids for the aFRR capacity auction")





if __name__ == "__main__":
    co_location_optimization()