# Solar Dataset Exploratory Data Analysis

## Overview

This repository contains Python scripts for conducting Exploratory Data Analysis (EDA) on solar datasets from three countries: Benin, Sierra Leone, and Togo. The analysis includes data profiling, cleaning, visualization, and statistical analysis.

## Requirements

To run the EDA scripts, you need to have the following packages installed:

```
numpy==1.23.5
pandas==1.5.3
matplotlib==3.7.1
seaborn==0.12.2
scipy==1.10.1
```

These packages are listed in the requirements.txt file and can be installed using pip:

```
pip install -r requirements.txt
```

## Dataset Structure

The datasets are located in the `data/` directory:

- `data/benin-malanville.csv` - Benin (Malanville) solar dataset
- `data/sierraleone-bumbuna.csv` - Sierra Leone (Bumbuna) solar dataset
- `data/togo-dapaong_qc.csv` - Togo (Dapaong) solar dataset

## Running the Analysis

Each country has its own dedicated script:

1. For Benin:

```
python benin_eda.py
```

2. For Sierra Leone:

```
python sierraleone_eda.py
```

3. For Togo:

```
python togo_eda.py
```

## Output

Each script produces:

1. Cleaned dataset saved to the `data/` directory with the format `<country>_clean.csv`
2. Visualizations saved to the `plots/` directory
3. Terminal output with key statistics and findings

## Analysis Components

Each script performs the following analyses:

1. **Summary Statistics & Missing-Value Report**

   - Basic statistics for all numeric columns
   - Missing value identification

2. **Outlier Detection & Basic Cleaning**

   - Z-score calculation for key variables
   - Outlier flagging and handling
   - Missing value imputation

3. **Time Series Analysis**

   - Temporal patterns of GHI, DNI, DHI, and Temperature
   - Hourly and monthly trends
   - Anomaly detection

4. **Cleaning Impact Analysis**

   - Effect of solar panel cleaning on module performance

5. **Correlation Analysis**

   - Heatmap of correlations between key variables
   - Scatter plots of relationships between weather variables

6. **Wind Distribution Analysis**

   - Wind rose and polar plots for wind direction and speed
   - Histograms of key variables

7. **Temperature Analysis**

   - Relationship between relative humidity, temperature, and solar radiation

8. **Bubble Chart Analysis**
   - Multivariate relationship visualization

## Notes

- Large datasets are automatically sampled for certain visualizations to improve performance.
- Extreme outliers (Z-score > 5) are optionally removed in the cleaning process.
- The `.gitignore` file is configured to exclude all CSV files in the `data/` directory from version control.

## References

1. Duffie, J.A. and Beckman, W.A. (2013). Solar Engineering of Thermal Processes. John Wiley & Sons, Inc.
2. Sengupta, M., et al. (2018). Best Practices Handbook for the Collection and Use of Solar Resource Data for Solar Energy Applications. NREL.
3. Python Data Science Handbook: https://jakevdp.github.io/PythonDataScienceHandbook/
4. Seaborn visualization library: https://seaborn.pydata.org/
