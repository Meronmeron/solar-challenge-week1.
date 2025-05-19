#!/usr/bin/env python
# coding: utf-8

"""
Country Comparison Analysis
==========================
This script compares solar metrics across Benin, Sierra Leone, and Togo
using the cleaned datasets from the EDA process.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

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

# 1. Load the cleaned datasets
print("Loading cleaned datasets...")

# Benin
try:
    benin_df = pd.read_csv('data/benin_clean.csv')
    print(f"Loaded Benin data: {benin_df.shape[0]} rows, {benin_df.shape[1]} columns")
    benin_df['Country'] = 'Benin'
except Exception as e:
    print(f"Error loading Benin data: {e}")
    benin_df = None

# Sierra Leone
try:
    sierraleone_df = pd.read_csv('data/sierraleone_clean.csv')
    print(f"Loaded Sierra Leone data: {sierraleone_df.shape[0]} rows, {sierraleone_df.shape[1]} columns")
    sierraleone_df['Country'] = 'Sierra Leone'
except Exception as e:
    print(f"Error loading Sierra Leone data: {e}")
    sierraleone_df = None

# Togo
try:
    togo_df = pd.read_csv('data/togo_clean.csv')
    print(f"Loaded Togo data: {togo_df.shape[0]} rows, {togo_df.shape[1]} columns")
    togo_df['Country'] = 'Togo'
except Exception as e:
    print(f"Error loading Togo data: {e}")
    togo_df = None

# 2. Combine the datasets
country_dfs = []
if benin_df is not None:
    country_dfs.append(benin_df)
if sierraleone_df is not None:
    country_dfs.append(sierraleone_df)
if togo_df is not None:
    country_dfs.append(togo_df)

if not country_dfs:
    raise Exception("No data could be loaded. Please run the EDA scripts first to generate clean data files.")

# Focus on common columns
metrics = ['GHI', 'DNI', 'DHI']
required_cols = metrics + ['Country']

# Check each dataframe has the required columns
for country_df in country_dfs:
    missing_cols = [col for col in required_cols if col not in country_df.columns]
    if missing_cols:
        print(f"Warning: {country_df['Country'].iloc[0]} data is missing columns: {missing_cols}")

# 3. Create combined dataset with only the required columns
combined_data = []
for country_df in country_dfs:
    # Select only columns present in the data
    available_cols = [col for col in required_cols if col in country_df.columns]
    combined_data.append(country_df[available_cols])

combined_df = pd.concat(combined_data, ignore_index=True)
print(f"Combined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")

# 4. Create boxplots for each metric
print("\nCreating boxplots for each metric...")
for metric in metrics:
    if metric in combined_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Country', y=metric, data=combined_df, palette='viridis')
        plt.title(f'{metric} Comparison Across Countries')
        plt.xlabel('Country')
        plt.ylabel(f'{metric} (W/m²)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'plots/comparison_{metric}_boxplot.png')
        plt.close()
    else:
        print(f"Warning: {metric} not found in all datasets, skipping boxplot")

# 5. Create summary table
print("\nGenerating summary statistics...")
summary_data = []

for country in combined_df['Country'].unique():
    country_data = combined_df[combined_df['Country'] == country]
    
    country_stats = {'Country': country}
    
    for metric in metrics:
        if metric in country_data.columns:
            country_stats[f'{metric}_Mean'] = country_data[metric].mean()
            country_stats[f'{metric}_Median'] = country_data[metric].median()
            country_stats[f'{metric}_StdDev'] = country_data[metric].std()
    
    summary_data.append(country_stats)

summary_df = pd.DataFrame(summary_data)
print("\nSummary Statistics Table:")
print(summary_df)

# Save summary table to CSV
summary_df.to_csv('country_comparison_summary.csv', index=False)
print("Summary table saved to 'country_comparison_summary.csv'")

# 6. Statistical Testing
print("\nPerforming statistical tests...")

for metric in metrics:
    if metric in combined_df.columns:
        countries_with_metric = combined_df.dropna(subset=[metric])
        
        # Check if we have data from at least 2 countries
        if len(countries_with_metric['Country'].unique()) >= 2:
            # Create groups for statistical testing
            groups = [countries_with_metric[countries_with_metric['Country'] == country][metric].values 
                     for country in countries_with_metric['Country'].unique()]
            
            # Test for normality
            normality_tests = []
            for i, country in enumerate(countries_with_metric['Country'].unique()):
                if len(groups[i]) > 8:  # Shapiro-Wilk requires at least 3 samples
                    stat, p = stats.shapiro(groups[i][:1000])  # Limit to 1000 samples for speed
                    normality_tests.append(p > 0.05)
            
            # If all data is normally distributed, use ANOVA, otherwise use non-parametric Kruskal-Wallis
            if all(normality_tests):
                stat, p = stats.f_oneway(*groups)
                test_name = "ANOVA"
            else:
                stat, p = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis"
            
            print(f"{test_name} test for {metric}: statistic={stat:.4f}, p-value={p:.6f}")
            print(f"Conclusion: {'Significant difference' if p < 0.05 else 'No significant difference'} between countries")
        else:
            print(f"Not enough countries have {metric} data for statistical testing")
    else:
        print(f"Warning: {metric} not found in all datasets, skipping statistical test")

# 7. Create bar chart for GHI ranking
if 'GHI' in combined_df.columns:
    print("\nCreating GHI ranking bar chart...")
    ghi_ranking = summary_df.sort_values(by='GHI_Mean', ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Country', y='GHI_Mean', data=ghi_ranking, palette='viridis')
    plt.title('Countries Ranked by Average GHI')
    plt.xlabel('Country')
    plt.ylabel('Average GHI (W/m²)')
    
    # Add value labels on bars
    for i, v in enumerate(ghi_ranking['GHI_Mean']):
        ax.text(i, v + 5, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/country_ghi_ranking.png')
    plt.close()

# 8. Key Observations (markdown formatted output)
print("\n## Key Observations\n")

# GHI comparison
if 'GHI_Mean' in summary_df.columns:
    highest_ghi = summary_df.loc[summary_df['GHI_Mean'].idxmax()]
    highest_ghi_country = highest_ghi['Country']
    highest_ghi_value = highest_ghi['GHI_Mean']
    
    highest_variability = summary_df.loc[summary_df['GHI_StdDev'].idxmax()]
    highest_var_country = highest_variability['Country']
    highest_var_value = highest_variability['GHI_StdDev']
    
    print(f"* **{highest_ghi_country}** shows the highest solar potential with an average GHI of {highest_ghi_value:.1f} W/m², indicating better overall conditions for solar power generation.")

# Variability observation
if 'GHI_StdDev' in summary_df.columns and 'GHI_Mean' in summary_df.columns:
    cv_values = []
    for idx, row in summary_df.iterrows():
        if 'GHI_StdDev' in row and 'GHI_Mean' in row and pd.notnull(row['GHI_StdDev']) and pd.notnull(row['GHI_Mean']) and row['GHI_Mean'] > 0:
            cv = row['GHI_StdDev'] / row['GHI_Mean']
            cv_values.append((row['Country'], cv))
    
    if cv_values:
        cv_values.sort(key=lambda x: x[1], reverse=True)
        highest_cv_country, highest_cv = cv_values[0]
        print(f"* **{highest_cv_country}** demonstrates the most variable solar radiation (coefficient of variation: {highest_cv:.2f}), suggesting higher unpredictability and potential need for robust energy storage solutions.")

# DNI vs DHI pattern
if set(['DNI_Mean', 'DHI_Mean']).issubset(summary_df.columns):
    dni_dhi_ratios = []
    for idx, row in summary_df.iterrows():
        if pd.notnull(row['DNI_Mean']) and pd.notnull(row['DHI_Mean']) and row['DHI_Mean'] > 0:
            ratio = row['DNI_Mean'] / row['DHI_Mean']
            dni_dhi_ratios.append((row['Country'], ratio))
    
    if dni_dhi_ratios:
        dni_dhi_ratios.sort(key=lambda x: x[1], reverse=True)
        highest_ratio_country, highest_ratio = dni_dhi_ratios[0]
        lowest_ratio_country, lowest_ratio = dni_dhi_ratios[-1]
        
        print(f"* **{highest_ratio_country}** has the highest ratio of direct to diffuse radiation ({highest_ratio:.2f}), suggesting clearer skies and potentially better conditions for concentrated solar power systems, while **{lowest_ratio_country}** shows more diffuse radiation, which may benefit from photovoltaic systems that can utilize scattered light.")

# Statistical significance
print("\n**Statistical Significance**:")
for metric in metrics:
    if metric in combined_df.columns:
        countries_with_metric = combined_df.dropna(subset=[metric])
        if len(countries_with_metric['Country'].unique()) >= 2:
            # Create groups for statistical testing
            groups = [countries_with_metric[countries_with_metric['Country'] == country][metric].values 
                     for country in countries_with_metric['Country'].unique()]
            
            # Run the test again just for the markdown
            stat, p = stats.kruskal(*groups)
            if p < 0.05:
                print(f"* The differences in **{metric}** between countries are statistically significant (p = {p:.6f})")
            else:
                print(f"* No statistically significant differences in **{metric}** between countries (p = {p:.6f})")

print("\nAnalysis complete. Check the 'plots' directory for visualizations.") 