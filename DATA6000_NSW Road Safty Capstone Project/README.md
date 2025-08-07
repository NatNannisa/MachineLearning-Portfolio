# NSW Road Fatality Analysis & Prediction Project

This project analyzes road fatalities in New South Wales (NSW), Australia using open government datasets. It includes data cleaning, feature engineering, risk classification, and machine learning models (Random Forest and XGBoost) to identify high-risk regions and predict patterns based on time, demographics, and road user behavior.

## Project Objectives

- Clean and preprocess NSW fatal crash data from the BITRE dataset.
- Cluster crash times, ages, and speed zones into meaningful categories.
- Classify Local Government Areas (LGAs) by risk levels based on incident frequency.
- Train predictive models (Random Forest & XGBoost) to:
  - Predict high-risk LGAs.
  - Forecast time-of-day crash clusters.
- Export interactive reports and visual decision trees for analysis.

## Tools and Libraries

- Python (pandas, numpy, scikit-learn, xgboost, graphviz)
- Jupyter Notebook / Python Script
- BITRE Excel Dataset via direct download
- Power BI / Excel (for result exploration)

## Project Structure

| File | Description |
|------|-------------|
| `DATA6000_Nannisa_Pattaranansit_1810122.py` | Main script for data processing, feature engineering, and model training |
| `Cleaned_NSW_Fatalities.xlsx` | Cleaned and preprocessed fatality dataset |
| `Updated_Cleaned_NSW_Fatalities.xlsx` | Data with risk levels included |
| `LGA_Prediction_Tree_*.png` | Visualizations of Random Forest decision trees for LGA risk prediction |
| `TimeCluster_Prediction_Tree_*.png` | Visualizations for Time Cluster predictions |
| `TimeCluster_Prediction_Results.xlsx` | Final prediction results with actual vs. predicted crash time cluster |

## Key Steps

### Data Acquisition and Cleaning

- Download data directly from BITRE.
- Focus analysis on NSW only.
- Replace missing values:
  - Numerical: replaced with 0
  - Categorical: replaced with 'Unknown'

### Feature Engineering

- Extract time-of-day from crash timestamp (e.g., Morning, Afternoon).
- Cluster age and speed limits into relevant categories.
- Derive LGA-level risk categories based on incident count.

### Machine Learning Models

#### Random Forest for LGA Risk Prediction

- Inputs: Hour of crash, Age Group, Time Cluster, Speed Zone
- Output: Risk Level (Low, Moderate, High)

#### Random Forest for Time Cluster Prediction

- Inputs: Day of Week, Road User Type, Age Group
- Output: Time of Day Cluster

#### XGBoost for Time Cluster Prediction

- Inputs: Combined features (e.g., Day-Road User interaction)
- Output: Multi-class prediction for time-of-day crash patterns

## Results

- Models achieved strong accuracy in both LGA risk and time cluster prediction.
- Visual tree outputs provide interpretable insights for stakeholders.
- Results exported to Excel for further reporting or visualization in BI tools.

## Dataset Source

- BITRE Fatalities Data  
  [Australian Government Bureau of Infrastructure and Transport Research Economics (BITRE)](https://www.bitre.gov.au/statistics/safety/fatal_road_crash_database)

## Author

**Nannisa Pattaranansit**  
Master of Business Analytics – Kaplan Business School  
GitHub: [github.com/nannisapns](https://github.com/yourusername)

## Future Enhancements

- Add geospatial analysis using heatmaps of high-risk areas.
- Incorporate environmental and behavioral features such as weather conditions.
- Deploy the model in an interactive dashboard or web application.
