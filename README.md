# Coffee Sales Forecasting App

## Overview
This project implements a sales forecasting system for coffee shops using linear regression. It processes historical sales data to predict daily revenue, helping coffee shop owners make informed business decisions.

## Features
- Historical sales data analysis
- Daily revenue prediction
- Seasonal trend analysis
- Interactive visualizations
- Performance metrics tracking

## Technology Stack
- Python 3.x
- Jupyter Notebook
- pandas
- scikit-learn
- matplotlib/seaborn

## Project Structure
```
.
├── notebook.ipynb          # Main Jupyter notebook with analysis
├── notebook.py            # Python script version of the notebook
├── model_metrics         # Model performance metrics
└── future_predictions.csv # Generated predictions
```

## Installation
1. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Run the Jupyter notebook:
```bash
jupyter notebook notebook.ipynb
```

Or run the Python script directly:
```bash
python notebook.py
```

## Model Performance
The project includes detailed model performance metrics, which can be found in the `model_metrics.txt` file. The current model achieves:
- Mean Absolute Error (MAE): 130.51
- R-squared Score: 0.26
- Root Mean Square Error (RMSE): 26320.37