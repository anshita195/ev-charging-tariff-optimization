# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

A three-agent ML pipeline for real-time EV charging tariff optimization using
large-scale session data. Specialized agents handle demand forecasting, dynamic
pricing, and continuous performance monitoring to maximize network revenue,
redistribute demand, and reduce peak-hour congestion — with zero new infrastructure.

## Results

| Agent | Metric | Value | Note |
|---|---|---|---|
| Demand Prediction | R² Score | 0.9944 | Explains 99.4% of demand variance |
| Demand Prediction | RMSE | 0.0133 | Improvement over naive lag1 baseline (0.0139) |
| Demand Prediction | MAE | 0.0051 | Avg absolute error of 0.5% utilization |
| Tariff Pricing | Revenue Uplift | +17.94% (₹89Cr → ₹105Cr) | 5-day test window, Shenzhen network scale |
| Tariff Pricing | Off-Peak Sessions Gain | +763,853 sessions (+27.1%) | Simulated via empirical elasticity ε = −2.118 |
| Tariff Pricing | Charger Utilization | 29.2% → 37.1% | Simulated post-pricing redistribution |
| Monitoring & Learning | Pricing Efficiency | ₹61.07 → ₹72.01 / unit volume | +17.93% over 5 evaluation episodes |
| Monitoring & Learning | Customer Response Rate | +5.07% avg volume shift | Per tariff change event |

## System Architecture

Three-agent pipeline operating on 5-minute grid-level data:

- **Demand Prediction Agent** — forecasts station utilization using Gradient
  Boosting on lag, temporal, and spatial features across 247 grids;
  `util_lag1` is the dominant predictor. Outperforms naive lag1 baseline on
  RMSE (the metric that matters most for congestion-spike detection), though
  naive baseline achieves lower MAE on small errors.

- **Tariff Pricing Agent** — applies tiered dynamic pricing against a ₹15/kWh
  flat baseline: surge up to ₹30/kWh when utilization > 80%, discount at
  ₹13.08/kWh when utilization < 30%. Demand elasticity ε = −2.118 estimated
  empirically via log-log regression on ST-EVCDP. 58.3% of slots fall in the
  discount zone, confirming chronic underutilization as the dominant
  inefficiency. Surge occurs in only 1% of slots.

- **Monitoring & Learning Agent** — tracks revenue gain, charger utilization,
  pricing efficiency, and wait-time proxy per daily evaluation episode.
  Implements an adaptive feedback loop that adjusts surge/discount multipliers
  based on prediction error and revenue-gap signals. Revenue gain remained
  stable at 17.1%–18.8% across all 5 episodes.

## Pricing Zone Breakdown

| Zone | Slot Share | Avg Utilization | Avg Tariff | Revenue Gain |
|---|---|---|---|---|
| Discount | 58.3% | 17.2% | ₹13.08/kWh | −5.8% |
| Normal | 35.0% | 41.3% | ₹16.13/kWh | +10.6% |
| Moderate | 5.6% | 67.4% | ₹19.67/kWh | +36.2% |
| Surge | 1.0% | 87.5% | ₹25.25/kWh | +57.7% |

> Discount zones generate negative revenue per unit deliberately — demand
> redistribution to off-peak windows is a design goal, not a system failure.

## Datasets

- **ACN-Data** — 14,848 EV charging sessions (Caltech Adaptive Charging
  Network, California, Apr–Dec 2018). Used for user behaviour EDA and revenue
  baseline. Single workplace site; fleet/public segmentation not possible.
  [ev.caltech.edu/dataset.html](https://ev.caltech.edu/dataset.html)

- **ST-EVCDP** — 2.13M grid-time slots across 247 urban grids at 5-minute
  intervals (Shenzhen, Jun–Jul 2022). Used for all ML modelling and dynamic
  tariff optimisation. 5-minute granularity and 247-grid spatial coverage are
  prerequisites for lag-based forecasting.
  [github.com/IntelligentSystemsLab/ST-EVCDP](https://github.com/IntelligentSystemsLab/ST-EVCDP)

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter / Google Colab

## Setup

This notebook is designed to run on **Google Colab** with data files in Google Drive.

1. Download datasets from the links above and upload to a Drive folder
   (e.g. `MyDrive/socbiz/`).
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
└── README.md
```

## Limitations

- ST-EVCDP covers a 30-day window only; seasonality effects are not captured.
  This is a dataset constraint, not a modelling choice.
- ACN is a single workplace site (Caltech); fleet vs. public segmentation is
  not possible and behaviour may not generalise to public charging networks.
- Off-peak session uplift (+763,853) is simulated via empirically estimated
  elasticity, not observed in a live deployment.
- Revenue projections (₹89Cr → ₹105Cr) are at Shenzhen network scale
  (247 grids, 1,706 stations) over a 5-day window; figures are not
  transferable to smaller networks without rescaling.
- No real-time grid price data integrated; electricity procurement cost
  dynamics are not modelled.