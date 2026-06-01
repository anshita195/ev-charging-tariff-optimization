# Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

## Overview

An agentic ML system for real-time EV charging tariff optimization 
using large-scale session data. Three specialized agents handle 
demand forecasting, dynamic pricing, and continuous monitoring 
to maximize revenue and reduce congestion.

## Results

| Agent | Metric | Value |
|---|---|---|
| Demand Prediction | R² Score | 0.9943 |
| Demand Prediction | RMSE | 0.0134 |
| Demand Prediction | MAE | 0.0050 |
| Tariff Pricing | Revenue Uplift | +17.91% |
| Tariff Pricing | Off-Peak Sessions Gain | +108,125 |
| Tariff Pricing | Charger Utilization | 29.2% → 30.3% |
| Pricing Efficiency | Revenue per kWh | ₹61.07 → ₹72.01 |

## System Architecture

Three-agent pipeline:
- **Demand Prediction Agent** — forecasts station utilization 
  using temporal and spatial features across 247 stations
- **Tariff Pricing Agent** — applies surge pricing above 80% 
  utilization, discount pricing below 30%
- **Monitoring & Learning Agent** — tracks revenue, wait times, 
  and pricing efficiency per episode

## Datasets

- **ACN-Data** — 14,848 EV charging sessions 
  (Caltech Adaptive Charging Network)
- **ST-EVCDP** — 2.1M+ charging slots across 247 urban 
  stations at 5-minute intervals (Shenzhen)

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Jupyter

## Setup

```bash
git clone https://github.com/anshita195/ev-charging-tariff-optimization
pip install -r requirements.txt
jupyter notebook ev_charging_optimization.ipynb
```

## Repository Structure

```
├── ev_charging_optimization.ipynb
├── report.pdf
├── requirements.txt
└── README.md
```