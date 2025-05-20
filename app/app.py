"""
Solar Data Visualization App
Main Streamlit application for visualizing and comparing solar metrics across countries
"""

import streamlit as st
from utils import load_data, create_boxplot, create_bar_chart, format_summary_table, run_statistical_test, create_time_series_plot

# Set page configuration
st.set_page_config(
    page_title="Solar Data Comparison",
    page_icon="☀️",
    layout="wide"
)

# App title and description
st.title("☀️ Solar Data Visualization")
st.markdown("""
Compare solar radiation metrics (GHI, DNI, DHI) across different countries.
Select countries and metrics to visualize the data and see statistical comparisons.
""")

# Sidebar for controls
st.sidebar.header("Controls")

# Load available countries
_, summary_df = load_data()
available_countries = summary_df['Country'].unique().tolist()

# Country selection
selected_countries = st.sidebar.multiselect(
    "Select Countries to Compare",
    options=available_countries,
    default=available_countries[:2] if len(available_countries) >= 2 else available_countries
)

# Metric selection
metrics = ["GHI", "DNI", "DHI"]
selected_metric = st.sidebar.selectbox(
    "Select Solar Metric",
    options=metrics,
    index=0
)

# Load data for selected countries
if selected_countries:
    data_df, summary_df = load_data(selected_countries)
    
    if data_df is not None:
        # Main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Boxplots", "Rankings", "Summary Statistics", "Statistical Tests"])
        
        with tab1:
            st.header(f"{selected_metric} Distribution by Country")
            boxplot = create_boxplot(data_df, selected_metric)
            if boxplot:
                st.pyplot(boxplot)
            else:
                st.warning(f"Could not create boxplot for {selected_metric}")
        
        with tab2:
            st.header("Country Rankings")
            ranking_metric = f"{selected_metric}_Mean"
            bar_chart = create_bar_chart(summary_df, ranking_metric)
            if bar_chart:
                st.pyplot(bar_chart)
            else:
                st.warning(f"Could not create bar chart for {ranking_metric}")
        
        with tab3:
            st.header("Summary Statistics")
            styled_table = format_summary_table(summary_df)
            st.dataframe(styled_table)
        
        with tab4:
            st.header("Statistical Significance")
            
            test_results = run_statistical_test(data_df, selected_metric)
            
            if "error" in test_results:
                st.warning(test_results["error"])
            else:
                st.write(f"**Test**: {test_results['test_name']}")
                st.write(f"**P-value**: {test_results['p_value']:.4f}")
                
                if test_results["significant"]:
                    st.success(f"**Result**: The difference in {selected_metric} between countries is statistically significant (p < 0.05)")
                else:
                    st.info(f"**Result**: The difference in {selected_metric} between countries is NOT statistically significant (p ≥ 0.05)")
        
        # Additional visualization
        st.header("Time Series Visualization")
        st.info("Note: This chart may not be available if timestamp data is missing")
        
        time_series = create_time_series_plot(data_df, selected_metric)
        if time_series:
            st.pyplot(time_series)
    else:
        st.error("No data available for the selected countries. Please select different countries.")
else:
    st.info("Please select at least one country from the sidebar to get started.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Solar Challenge Project - Data Visualization") 