# Marketing Mind — AI Marketing Budget Optimizer

Marketing Mind is a data science project that explores marketing performance,
time-series prediction, marketing mix modeling concepts, and budget allocation
using Python.

The project includes an interactive Streamlit dashboard for exploring marketing
data, evaluating a baseline prediction model, and experimenting with budget
allocation under configurable constraints.

## Features

- Exploratory data analysis and correlation analysis
- Marketing data visualization
- Baseline machine learning model using Linear Regression
- Chronological train-test evaluation
- Adstock transformation experiments
- Comparison of different adstock decay rates
- Budget allocation optimization using SciPy
- Interactive dashboard built with Streamlit

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Streamlit

## Project Structure

```text
Marketing Mind/
├── app/
│   └── app.py
├── data/
│   ├── mmm_example.csv
│   ├── mmm_predictions.csv
│   ├── adstock_tuning_results.csv
│   ├── adstock_validation_results.csv
│   └── adstock_decay_comparison.png
├── notebooks/
│   ├── 01_explore_data.py
│   ├── 02_analyze_marketing.py
│   ├── 03_baseline_model.py
│   ├── 04_time_series_evaluation.py
│   ├── 05_marketing_mix_model.py
│   ├── 06_tune_adstock.py
│   ├── 07_validate_adstock.py
│   └── 08_visualize_adstock.py
├── src/
│   └── budget_optimizer.py
├── tests/
├── requirements.txt
└── README.md
