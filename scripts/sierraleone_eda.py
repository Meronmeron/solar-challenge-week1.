#!/usr/bin/env python
# coding: utf-8

"""
Exploratory Data Analysis for Sierra Leone Solar Dataset
=======================================================
This script performs comprehensive EDA on the Sierra Leone (Bumbuna) solar dataset.
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Optional imports (install if needed)
# import plotly.express as px
# from windrose import WindroseAxes  # pip install windrose

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
    print("Created 'plots' directory for saving figures")

# Set plotting style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 1. Data Loading
print("1. Loading Data")
file_path = 'data/sierraleone-bumbuna.csv'
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nFirst few rows:")
print(df.head())

print("\nColumn data types:")
print(df.info())

# 2. Summary Statistics & Missing-Value Report
print("\n2. Summary Statistics & Missing Values")
print("\nSummary statistics:")
print(df.describe())

print("\nMissing values check:")
missing_values = df.isna().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percent
})
print("\nColumns with >5% missing values:")
print(missing_df[missing_df['Percentage'] > 5].sort_values('Percentage', ascending=False))

# 3. Timestamp Conversion
print("\n3. Converting Timestamp")
# Convert timestamp to datetime if it exists
possible_timestamp_cols = ['Timestamp', 'timestamp', 'Date', 'date', 'datetime', 'time']
timestamp_col = None

for col in possible_timestamp_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
        timestamp_col = col
        print(f"Converted {col} to datetime")
        break

if timestamp_col:
    df['year'] = df[timestamp_col].dt.year
    df['month'] = df[timestamp_col].dt.month
    df['day'] = df[timestamp_col].dt.day
    df['hour'] = df[timestamp_col].dt.hour
    print(f"Added year, month, day, and hour columns based on {timestamp_col}")

# 4. Outlier Detection
print("\n4. Outlier Detection")
def detect_outliers(df, columns, z_threshold=3):
    outlier_counts = {}
    
    for col in columns:
        if col in df.columns:
            # Calculate z-scores
            df[f'{col}_zscore'] = np.abs(stats.zscore(df[col], nan_policy='omit'))
            
            # Count outliers
            outliers = df[df[f'{col}_zscore'] > z_threshold]
            outlier_counts[col] = len(outliers)
    
    return outlier_counts

# List of key columns to check for outliers
key_columns = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
available_columns = [col for col in key_columns if col in df.columns]

# Detect outliers
outlier_counts = detect_outliers(df, available_columns)
print("Number of outliers detected (|z-score| > 3):")
for col, count in outlier_counts.items():
    print(f"{col}: {count} ({count/len(df)*100:.2f}%)")

# 5. Basic Cleaning
print("\n5. Basic Cleaning")
# Create a copy of the original dataframe before cleaning
df_clean = df.copy()

# Impute missing values in key columns with median
for col in available_columns:
    if df_clean[col].isna().sum() > 0:
        median_value = df_clean[col].median()
        print(f"Imputing {df_clean[col].isna().sum()} missing values in {col} with median: {median_value:.2f}")
        df_clean[col] = df_clean[col].fillna(median_value)
    
    # Remove extreme outliers (optional)
    if f'{col}_zscore' in df_clean.columns:
        extreme_outliers = df_clean[df_clean[f'{col}_zscore'] > 5].shape[0]
        if extreme_outliers > 0:
            print(f"Removing {extreme_outliers} extreme outliers in {col} (z-score > 5)")
            df_clean = df_clean[df_clean[f'{col}_zscore'] <= 5]

# Export cleaned DataFrame
output_path = 'data/sierraleone_clean.csv'
df_clean.to_csv(output_path, index=False)
print(f"Cleaned dataset exported to {output_path}")

# 6. Time Series Analysis
print("\n6. Time Series Analysis")
if timestamp_col:
    # Sample data if too many points
    if len(df_clean) > 10000:
        sample_size = 10000
        df_sample = df_clean.sample(sample_size, random_state=42)
        df_sample = df_sample.sort_values(by=timestamp_col)
    else:
        df_sample = df_clean.sort_values(by=timestamp_col)
    
    # Plot GHI, DNI, DHI
    radiation_cols = [col for col in ['GHI', 'DNI', 'DHI'] if col in df_clean.columns]
    
    plt.figure(figsize=(16, 10))
    for col in radiation_cols:
        plt.plot(df_sample[timestamp_col], df_sample[col], label=col)
    
    plt.title('Solar Radiation Measurements Over Time')
    plt.xlabel('Time')
    plt.ylabel('W/m²')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/sierraleone_radiation_time_series.png')
    plt.close()
    
    # Plot Tamb if available
    if 'Tamb' in df_clean.columns:
        plt.figure(figsize=(16, 8))
        plt.plot(df_sample[timestamp_col], df_sample['Tamb'], color='red')
        plt.title('Ambient Temperature Over Time')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/sierraleone_temperature_time_series.png')
        plt.close()

    # Analyze daily patterns
    hourly_data = df_clean.groupby('hour').mean(numeric_only=True).reset_index()
    
    plt.figure(figsize=(14, 8))
    for col in radiation_cols:
        plt.plot(hourly_data['hour'], hourly_data[col], marker='o', label=col)
    
    plt.title('Average Solar Radiation by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Radiation (W/m²)')
    plt.xticks(range(0, 24))
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/sierraleone_hourly_radiation.png')
    plt.close()

    # Analyze monthly patterns
    monthly_data = df_clean.groupby('month').mean(numeric_only=True).reset_index()
    
    plt.figure(figsize=(14, 8))
    for col in radiation_cols:
        plt.plot(monthly_data['month'], monthly_data[col], marker='o', label=col)
    
    plt.title('Average Solar Radiation by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Radiation (W/m²)')
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/sierraleone_monthly_radiation.png')
    plt.close()

# 7. Cleaning Impact Analysis
print("\n7. Cleaning Impact Analysis")
cleaning_cols = [col for col in df_clean.columns if 'clean' in col.lower()]

if cleaning_cols and 'ModA' in df_clean.columns and 'ModB' in df_clean.columns:
    cleaning_col = cleaning_cols[0]
    print(f"Found cleaning flag column: {cleaning_col}")
    
    # Group by cleaning flag
    grouped = df_clean.groupby(cleaning_col).mean(numeric_only=True)[['ModA', 'ModB']]
    
    # Plot average ModA & ModB pre/post-clean
    ax = grouped.plot(kind='bar', figsize=(10, 6))
    plt.title('Impact of Cleaning on Module Performance')
    plt.xlabel('Cleaning Status')
    plt.ylabel('Average Module Output')
    plt.xticks(rotation=0)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f')
    
    plt.savefig('plots/sierraleone_cleaning_impact.png')
    plt.close()
else:
    print("No cleaning flag or module data found for cleaning impact analysis.")

# 8. Correlation Analysis
print("\n8. Correlation Analysis")
correlation_cols = [col for col in ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'Tamb', 'RH', 'WS'] 
                   if col in df_clean.columns]

if correlation_cols:
    corr_matrix = df_clean[correlation_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f')
    
    plt.title('Correlation Heatmap of Key Variables')
    plt.tight_layout()
    plt.savefig('plots/sierraleone_correlation_heatmap.png')
    plt.close()

# 9. Scatter plots
print("\n9. Scatter Plot Analysis")
if 'GHI' in df_clean.columns:
    # Sample data if too many points
    if len(df_clean) > 5000:
        scatter_sample = df_clean.sample(5000, random_state=42)
    else:
        scatter_sample = df_clean
    
    # WS vs GHI
    if 'WS' in df_clean.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(scatter_sample['WS'], scatter_sample['GHI'], alpha=0.5)
        plt.title('Relationship between Wind Speed and Global Horizontal Irradiance')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('GHI (W/m²)')
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/sierraleone_ws_ghi_scatter.png')
        plt.close()
    
    # RH vs GHI and RH vs Tamb
    if 'RH' in df_clean.columns:
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))
        
        # RH vs GHI
        ax[0].scatter(scatter_sample['RH'], scatter_sample['GHI'], alpha=0.5, color='blue')
        ax[0].set_title('Relative Humidity vs GHI')
        ax[0].set_xlabel('Relative Humidity (%)')
        ax[0].set_ylabel('GHI (W/m²)')
        ax[0].grid(True, alpha=0.3)
        
        # RH vs Tamb
        if 'Tamb' in df_clean.columns:
            ax[1].scatter(scatter_sample['RH'], scatter_sample['Tamb'], alpha=0.5, color='red')
            ax[1].set_title('Relative Humidity vs Ambient Temperature')
            ax[1].set_xlabel('Relative Humidity (%)')
            ax[1].set_ylabel('Temperature (°C)')
            ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/sierraleone_rh_relationships.png')
        plt.close()

# 10. Wind Analysis
print("\n10. Wind Distribution Analysis")
# Simple polar scatter plot for wind
if 'WS' in df_clean.columns and 'WD' in df_clean.columns:
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    # Convert degrees to radians
    wd_rad = np.radians(df_clean['WD'])
    
    # Create scatter plot
    sc = ax.scatter(wd_rad, df_clean['WS'], c=df_clean['WS'], cmap='viridis', alpha=0.5)
    
    plt.colorbar(sc, label='Wind Speed (m/s)')
    ax.set_theta_zero_location('N')  # 0 degrees at the top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_rlabel_position(0)        # Move radial labels away from the plotted line
    
    plt.title('Wind Speed and Direction')
    plt.tight_layout()
    plt.savefig('plots/sierraleone_wind_rose.png')
    plt.close()

# 11. Histograms
print("\n11. Distribution Analysis")
fig, ax = plt.subplots(1, 2, figsize=(18, 7))

# GHI histogram
if 'GHI' in df_clean.columns:
    sns.histplot(df_clean['GHI'], kde=True, ax=ax[0], color='blue')
    ax[0].set_title('Distribution of Global Horizontal Irradiance')
    ax[0].set_xlabel('GHI (W/m²)')
    ax[0].set_ylabel('Frequency')

# WS histogram
if 'WS' in df_clean.columns:
    sns.histplot(df_clean['WS'], kde=True, ax=ax[1], color='green')
    ax[1].set_title('Distribution of Wind Speed')
    ax[1].set_xlabel('Wind Speed (m/s)')
    ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('plots/sierraleone_distributions.png')
plt.close()

# 12. Temperature and Humidity Analysis
print("\n12. Temperature and Humidity Analysis")
if 'RH' in df_clean.columns and 'Tamb' in df_clean.columns and 'GHI' in df_clean.columns:
    # Bin RH into categories
    df_clean['RH_bins'] = pd.cut(df_clean['RH'], bins=5)
    
    # Group by RH bins and calculate mean
    rh_grouped = df_clean.groupby('RH_bins').agg({
        'Tamb': 'mean',
        'GHI': 'mean'
    }).reset_index()
    
    # Plot the relationship
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot Tamb vs RH
    color = 'tab:red'
    ax1.set_xlabel('Relative Humidity Range')
    ax1.set_ylabel('Average Temperature (°C)', color=color)
    ax1.plot(rh_grouped['RH_bins'].astype(str), rh_grouped['Tamb'], marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Average GHI (W/m²)', color=color)
    ax2.plot(rh_grouped['RH_bins'].astype(str), rh_grouped['GHI'], marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Effect of Relative Humidity on Temperature and Solar Radiation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/sierraleone_rh_effect.png')
    plt.close()

print("\nEDA completed. All plots saved to the 'plots' directory.")
print("\nKey observations to look for in the analysis:")
print("1. Diurnal patterns in solar radiation")
print("2. Seasonal variations in GHI, DNI, and DHI")
print("3. Correlations between weather variables and solar radiation")
print("4. Impact of cleaning on module performance")
print("5. Wind direction and speed patterns")
print("6. Effects of relative humidity on temperature and solar radiation") 