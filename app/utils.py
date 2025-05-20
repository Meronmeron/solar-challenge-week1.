"""
Utility functions for the Solar Data Visualization App
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

def load_data(countries=None):
    """
    Load data for selected countries
    
    Parameters:
    ----------
    countries : list
        List of countries to load data for. If None, load all available countries.
        
    Returns:
    -------
    combined_df : pandas.DataFrame
        Combined dataset with country data
    summary_df : pandas.DataFrame
        Summary statistics
    """
    # Define the metrics we're interested in
    metrics = ['GHI', 'DNI', 'DHI']
    
    # Load the summary statistics
    summary_df = pd.read_csv('country_comparison_summary.csv')
    
    # Initialize an empty list to store dataframes
    dfs = []
    
    # Filter countries if specified
    if countries is not None:
        country_list = countries
    else:
        country_list = summary_df['Country'].unique()
    
    # Load data for each country
    for country in country_list:
        try:
            # Try to load the cleaned data file
            country_filename = country.lower().replace(" ", "_")
            # Special case for Sierra Leone
            if country_filename == "sierra_leone":
                country_filename = "sierraleone"
            
            file_path = f'data/{country_filename}_clean.csv'
            df = pd.read_csv(file_path)
            
            # Add country column if not already present
            if 'Country' not in df.columns:
                df['Country'] = country
            
            # Only keep the columns we need
            available_cols = [col for col in metrics + ['Country'] if col in df.columns]
            df = df[available_cols]
            
            # Add to list
            dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load data for {country}: {e}")
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df, summary_df.loc[summary_df['Country'].isin(country_list)]
    else:
        st.error("No data could be loaded. Please check the data directory.")
        return None, summary_df.loc[summary_df['Country'].isin(country_list)]

def create_boxplot(df, metric):
    """
    Create a boxplot for the given metric
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with the data
    metric : str
        Metric to plot (e.g., 'GHI', 'DNI', 'DHI')
        
    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if metric not in df.columns:
        st.warning(f"{metric} not found in the data")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Country', y=metric, data=df, palette='viridis', ax=ax)
    ax.set_title(f'{metric} Comparison Across Countries')
    ax.set_xlabel('Country')
    ax.set_ylabel(f'{metric} (W/m²)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def create_bar_chart(summary_df, metric='GHI_Mean'):
    """
    Create a bar chart ranking countries by a metric
    
    Parameters:
    ----------
    summary_df : pandas.DataFrame
        DataFrame with summary statistics
    metric : str
        Metric to rank by (default: 'GHI_Mean')
        
    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if metric not in summary_df.columns:
        st.warning(f"{metric} not found in the summary data")
        return None
    
    # Sort by the metric
    sorted_df = summary_df.sort_values(by=metric, ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x='Country', y=metric, data=sorted_df, palette='viridis', ax=ax)
    
    # Add value labels on bars
    for i, bar in enumerate(bars.patches):
        value = sorted_df.iloc[i][metric]
        ax.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 5, 
                f'{value:.1f}', 
                ha='center')
    
    metric_name = metric.split('_')[0]  # Extract metric name without suffix
    ax.set_title(f'Countries Ranked by {metric_name}')
    ax.set_xlabel('Country')
    
    # Set y-label based on the metric
    if 'Mean' in metric:
        ax.set_ylabel(f'Average {metric_name} (W/m²)')
    elif 'Median' in metric:
        ax.set_ylabel(f'Median {metric_name} (W/m²)')
    elif 'StdDev' in metric:
        ax.set_ylabel(f'{metric_name} Standard Deviation (W/m²)')
    
    plt.tight_layout()
    
    return fig

def format_summary_table(summary_df):
    """
    Format summary statistics table for display
    
    Parameters:
    ----------
    summary_df : pandas.DataFrame
        DataFrame with summary statistics
        
    Returns:
    -------
    styled_df : pandas.io.formats.style.Styler
        Styled DataFrame for display
    """
    # Create a copy with only the columns we want to display
    display_cols = []
    for metric in ['GHI', 'DNI', 'DHI']:
        for stat in ['Mean', 'Median', 'StdDev']:
            col = f'{metric}_{stat}'
            if col in summary_df.columns:
                display_cols.append(col)
    
    # Create a display DataFrame with Country and all available metrics
    display_df = summary_df[['Country'] + display_cols].copy()
    
    # Format numbers to 1 decimal place
    for col in display_cols:
        display_df[col] = display_df[col].round(1)
    
    # Add units to column names
    renamed_cols = {col: f"{col.split('_')[0]} {col.split('_')[1]} (W/m²)" 
                   for col in display_cols}
    renamed_cols['Country'] = 'Country'
    
    # Rename columns
    display_df = display_df.rename(columns=renamed_cols)
    
    # Convert to styled DataFrame
    styled_df = display_df.style.background_gradient(cmap='viridis', subset=display_df.columns[1:])
    
    return styled_df

def run_statistical_test(df, metric):
    """
    Run statistical test to check if differences between countries are significant
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with the data
    metric : str
        Metric to test (e.g., 'GHI', 'DNI', 'DHI')
        
    Returns:
    -------
    results : dict
        Dictionary with test results
    """
    if metric not in df.columns:
        return {"error": f"{metric} not found in the data"}
    
    countries = df['Country'].unique()
    
    if len(countries) < 2:
        return {"error": "Need at least 2 countries for statistical testing"}
    
    # Create groups for statistical testing
    groups = []
    country_names = []
    
    for country in countries:
        country_data = df[df['Country'] == country][metric].dropna().values
        if len(country_data) > 0:
            groups.append(country_data)
            country_names.append(country)
    
    if len(groups) < 2:
        return {"error": "Not enough countries with data for statistical testing"}
    
    # Run Kruskal-Wallis test (non-parametric alternative to ANOVA)
    try:
        stat, p = stats.kruskal(*groups)
        return {
            "test_name": "Kruskal-Wallis",
            "statistic": stat,
            "p_value": p,
            "significant": p < 0.05,
            "countries": country_names
        }
    except Exception as e:
        return {"error": f"Error running statistical test: {e}"}

def create_time_series_plot(df, metric, sample_size=5000):
    """
    Create a time series plot for the given metric
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with the data
    metric : str
        Metric to plot (e.g., 'GHI', 'DNI', 'DHI')
    sample_size : int
        Number of samples to use for plotting (to avoid overplotting)
        
    Returns:
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if metric not in df.columns:
        st.warning(f"{metric} not found in the data")
        return None
    
    # Check if we have timestamp columns
    timestamp_cols = ['Timestamp', 'timestamp', 'Date', 'date', 'datetime', 'time']
    timestamp_col = None
    
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        st.warning("No timestamp column found for time series plot")
        return None
    
    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sample data to avoid overplotting
    if len(df) > sample_size:
        sample_df = df.groupby('Country').apply(lambda x: x.sample(min(len(x), sample_size//len(df['Country'].unique())), random_state=42))
        sample_df = sample_df.reset_index(drop=True)
    else:
        sample_df = df
    
    # Sort by timestamp
    sample_df = sample_df.sort_values(by=timestamp_col)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for country in sample_df['Country'].unique():
        country_data = sample_df[sample_df['Country'] == country]
        ax.plot(country_data[timestamp_col], country_data[metric], label=country, alpha=0.7)
    
    ax.set_title(f'{metric} Over Time by Country')
    ax.set_xlabel('Date/Time')
    ax.set_ylabel(f'{metric} (W/m²)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig 