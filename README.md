# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

An agentic ML system for real-time EV charging tariff optimization using
large-scale session data. Three specialized agents handle demand forecasting,
dynamic pricing, and continuous monitoring to maximize revenue and reduce
congestion.

## Results

| Agent | Metric | Value |
|---|---|---|
| Demand Prediction | R² Score | 0.9944 |
| Demand Prediction | RMSE | 0.0133 |
| Demand Prediction | MAE | 0.0052 |
| Tariff Pricing | Revenue Uplift | +17.93% (₹89Cr → ₹105Cr, 5-day window) |
| Tariff Pricing | Off-Peak Sessions Gain | +764,227 sessions (+27.1%) |
| Tariff Pricing | Charger Utilization | 29.2% → 37.1% (simulated) |
| Pricing Efficiency | Revenue per kWh | ₹61.07 → ₹72.01 |

## System Architecture

Three-agent pipeline:

- **Demand Prediction Agent** — forecasts station utilization using Gradient
  Boosting on lag, temporal, and spatial features across 247 grids at 5-minute
  intervals; `util_lag1` is the dominant predictor
- **Tariff Pricing Agent** — applies surge pricing (up to ₹30/kWh) above 80%
  utilization, discount pricing (₹13.08/kWh) below 30%, against a ₹15/kWh
  flat baseline; demand elasticity ε = −2.118 (log-log regression, ST-EVCDP)
- **Monitoring & Learning Agent** — tracks revenue, charger utilization, wait
  times, and pricing efficiency per daily episode; implements an adaptive
  feedback loop that adjusts surge and discount multipliers based on prediction
  error and revenue-gap signals

## Datasets

- **ACN-Data** — 14,848 EV charging sessions (Caltech Adaptive Charging
  Network, California, Apr–Dec 2018); used for user behaviour EDA and revenue
  baseline
- **ST-EVCDP** — 2.13M grid-time slots across 247 urban grids at 5-minute
  intervals (Shenzhen, Jun–Jul 2022); used for demand prediction and dynamic
  tariff optimisation

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter / Google Colab

## Setup

This notebook is designed to run on **Google Colab** with data files stored in
Google Drive.

1. Upload all dataset files to a folder in your Drive (e.g. `MyDrive/socbiz/`). (Dataset files are not included due to size. Download ACN-Data from ev.caltech.edu and ST-EVCDP from github.com/IntelligentSystemsLab/ST-EVCDP)
2. Open `ev_charging_optimization.ipynb` in Google Colab.
3. Run the first cell to mount Drive and set the `PATH` variable to your folder.
4. Run all cells sequentially.

Required dataset files:
```
acndata_sessions.json.csv
occupancy.csv  |  volume.csv  |  duration.csv  |  price.csv
time.csv  |  distance.csv  |  adj.csv  |  information.csv  |  stations.csv
```

## Repository Structure

```
├── ev_charging_optimization.ipynb   # Main notebook (Colab)
├── report.pdf                       # Presentation deck (6 content slides + appendix)
├── requirements.txt
├── README.md
```

## Limitations

- ST-EVCDP covers a 30-day window only; seasonality effects are not captured
- ACN is a single workplace site (Caltech); fleet/public segmentation is not
  possible
- Off-peak session uplift is simulated via empirically estimated elasticity,
  not observed in a live deployment
- No real-time grid price data integrated
