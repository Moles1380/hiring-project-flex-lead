"""
Holds the information on the asset.
"""
# ── Asset & strategy constants ───────────────────────────────────────────────
BESS_POWER  = 50        # MW
BESS_ENERGY = 200       # MWh
MAX_CYCLES  = 2
CYCLE_COST  = 25        # EUR/MWh – minimum net spread to justify cycling
PV_MW       = 100       # MW rated capacity
GRID_MW     = 95        # MW – import & export cap
PV_SCALE    = PV_MW / 2000   # scale national solar forecast → this asset
PRICE_AREA  = "DK1"
TRAIN_DAYS  = 90
INITIAL_SOC = 100       # MWh (50 % of BESS_ENERGY at day start)