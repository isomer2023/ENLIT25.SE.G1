import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Energy Load Profile Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #3DCD58;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3DCD58;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Energy Load Profile Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Schneider Electric - Industrial Site Analysis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=['xlsx', 'xls'],
        help="Upload your energy consumption data in Excel format"
    )
    
    skip_rows = st.number_input(
        "Skip Header Rows",
        min_value=0,
        max_value=10,
        value=1,
        help="Number of rows to skip after the header row (e.g., set to 1 to skip units/metadata row)"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This dashboard visualizes quarterly-hour energy consumption data "
        "for industrial site load analysis."
    )

# Main content
if uploaded_file is not None:
    try:
        # Load data
        with st.spinner('Loading data...'):
            # Try to read Excel file, potentially skipping header rows
            # If skip_rows > 0, skip those rows AFTER the header row (row 0)
            if skip_rows > 0:
                # Skip specific rows: row 1, row 2, ..., row skip_rows
                # This uses row 0 as header and skips the next skip_rows rows
                rows_to_skip = list(range(1, int(skip_rows) + 1))
                df = pd.read_excel(uploaded_file, skiprows=rows_to_skip)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Auto-detect datetime column
            datetime_col = None
            
            # First, check if first column is already datetime
            if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                datetime_col = df.columns[0]
            else:
                # Try to convert first column to datetime
                for col in df.columns[:3]:  # Check first 3 columns
                    try:
                        test_conversion = pd.to_datetime(df[col], errors='coerce')
                        # If more than 50% of values are valid dates, use this column
                        if test_conversion.notna().sum() / len(df) > 0.5:
                            datetime_col = col
                            break
                    except:
                        continue
            
            if datetime_col is None:
                st.error("Could not identify a datetime column. Please ensure your file has a column with date/time values.")
                st.info("The first column should contain timestamps in a recognizable format (e.g., '2024-01-01 00:00:00').")
                st.stop()
            
            # Convert to datetime
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
            
            # Remove rows where datetime conversion failed
            initial_len = len(df)
            df = df.dropna(subset=[datetime_col])
            
            if len(df) == 0:
                st.error("No valid datetime values found in the detected datetime column.")
                st.stop()
            
            if initial_len > len(df):
                st.warning(f"Removed {initial_len - len(df)} rows with invalid datetime values.")
            
            # Display raw data info
            st.success(f"Data loaded successfully! {len(df)} records found.")
            
            # Identify load columns (all numeric columns except datetime)
            load_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(load_columns) == 0:
                st.error("No numeric columns found for load data.")
                st.stop()
            
            # Sort by datetime
            df = df.sort_values(datetime_col)
            
            # Extract time features
            df['Date'] = df[datetime_col].dt.date
            df['Hour'] = df[datetime_col].dt.hour
            df['Day_of_Week'] = df[datetime_col].dt.day_name()
            df['Month'] = df[datetime_col].dt.month_name()
            
        # Sidebar filters
        with st.sidebar:
            st.markdown("---")
            st.header("Filters")
            
            # Date range filter
            min_date = df[datetime_col].min().date()
            max_date = df[datetime_col].max().date()
            
            date_range = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # Load selection
            selected_loads = st.multiselect(
                "Select Loads to Display",
                options=load_columns,
                default=load_columns[:5] if len(load_columns) > 5 else load_columns
            )
            
        # Filter data based on date range
        if len(date_range) == 2:
            mask = (df[datetime_col].dt.date >= date_range[0]) & (df[datetime_col].dt.date <= date_range[1])
            filtered_df = df[mask]
        else:
            filtered_df = df
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Time Series", 
            "Heatmaps", 
            "Statistics", 
            "Load Breakdown",
            "Raw Data"
        ])
        
        # TAB 1: Time Series Analysis
        with tab1:
            st.subheader("Time Series Analysis")
            
            if selected_loads:
                # Total consumption over time
                fig_total = go.Figure()
                
                for load in selected_loads:
                    fig_total.add_trace(go.Scatter(
                        x=filtered_df[datetime_col],
                        y=filtered_df[load],
                        mode='lines',
                        name=load,
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Energy: %{y:.2f} kWh<extra></extra>'
                    ))
                
                fig_total.update_layout(
                    title="Energy Consumption Over Time",
                    xaxis_title="Date & Time",
                    yaxis_title="Energy Consumption (kWh)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_total, use_container_width=True)
                
                # Stacked area chart for total consumption
                st.subheader("Stacked Load Distribution")
                
                fig_stacked = go.Figure()
                
                for load in selected_loads:
                    fig_stacked.add_trace(go.Scatter(
                        x=filtered_df[datetime_col],
                        y=filtered_df[load],
                        mode='lines',
                        name=load,
                        stackgroup='one',
                        hovertemplate='<b>%{fullData.name}</b><br>Energy: %{y:.2f} kWh<extra></extra>'
                    ))
                
                fig_stacked.update_layout(
                    title="Stacked Energy Consumption",
                    xaxis_title="Date & Time",
                    yaxis_title="Total Energy Consumption (kWh)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_stacked, use_container_width=True)
            else:
                st.warning("Please select at least one load to display.")
        
        # TAB 2: Heatmaps
        with tab2:
            st.subheader("Consumption Heatmaps")
            
            if selected_loads:
                load_for_heatmap = st.selectbox(
                    "Select Load for Heatmap",
                    options=selected_loads
                )
                
                # Create pivot table for heatmap (Hour vs Day)
                heatmap_data = filtered_df.pivot_table(
                    values=load_for_heatmap,
                    index='Hour',
                    columns='Date',
                    aggfunc='mean'
                )
                
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Date", y="Hour of Day", color="Energy (kWh)"),
                    title=f"Hourly Consumption Pattern - {load_for_heatmap}",
                    aspect="auto",
                    color_continuous_scale="YlOrRd"
                )
                
                fig_heatmap.update_layout(height=600)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Day of week heatmap
                dow_heatmap_data = filtered_df.pivot_table(
                    values=load_for_heatmap,
                    index='Hour',
                    columns='Day_of_Week',
                    aggfunc='mean'
                )
                
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_heatmap_data = dow_heatmap_data[[col for col in day_order if col in dow_heatmap_data.columns]]
                
                fig_dow_heatmap = px.imshow(
                    dow_heatmap_data,
                    labels=dict(x="Day of Week", y="Hour of Day", color="Avg Energy (kWh)"),
                    title=f"Average Consumption by Day of Week - {load_for_heatmap}",
                    aspect="auto",
                    color_continuous_scale="Viridis"
                )
                
                fig_dow_heatmap.update_layout(height=600)
                st.plotly_chart(fig_dow_heatmap, use_container_width=True)
            else:
                st.warning("Please select at least one load to display.")
        
        # TAB 3: Statistics
        with tab3:
            st.subheader("Statistical Summary")
            
            if selected_loads:
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_consumption = filtered_df[selected_loads].sum().sum()
                avg_consumption = filtered_df[selected_loads].mean().mean()
                peak_consumption = filtered_df[selected_loads].max().max()
                
                with col1:
                    st.metric(
                        "Total Consumption",
                        f"{total_consumption:,.0f} kWh"
                    )
                
                with col2:
                    st.metric(
                        "Average Consumption",
                        f"{avg_consumption:.2f} kWh"
                    )
                
                with col3:
                    st.metric(
                        "Peak Demand",
                        f"{peak_consumption:.2f} kWh"
                    )
                
                with col4:
                    days_analyzed = (filtered_df[datetime_col].max() - filtered_df[datetime_col].min()).days + 1
                    st.metric(
                        "Days Analyzed",
                        f"{days_analyzed}"
                    )
                
                st.markdown("---")
                
                # Detailed statistics table
                st.subheader("Detailed Statistics by Load")
                
                stats_df = pd.DataFrame({
                    'Load': selected_loads,
                    'Total (kWh)': [filtered_df[load].sum() for load in selected_loads],
                    'Average (kWh)': [filtered_df[load].mean() for load in selected_loads],
                    'Max (kWh)': [filtered_df[load].max() for load in selected_loads],
                    'Min (kWh)': [filtered_df[load].min() for load in selected_loads],
                    'Std Dev': [filtered_df[load].std() for load in selected_loads]
                })
                
                stats_df = stats_df.sort_values('Total (kWh)', ascending=False)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Daily consumption trend
                st.subheader("Daily Consumption Trend")
                
                daily_consumption = filtered_df.groupby('Date')[selected_loads].sum().reset_index()
                daily_consumption['Total'] = daily_consumption[selected_loads].sum(axis=1)
                
                fig_daily = px.line(
                    daily_consumption,
                    x='Date',
                    y='Total',
                    title="Total Daily Energy Consumption",
                    markers=True
                )
                
                fig_daily.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Total Energy (kWh)",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_daily, use_container_width=True)
                
                # Hourly average consumption
                st.subheader("Average Hourly Consumption Pattern")
                
                hourly_avg = filtered_df.groupby('Hour')[selected_loads].mean().reset_index()
                hourly_avg['Total'] = hourly_avg[selected_loads].sum(axis=1)
                
                fig_hourly = px.bar(
                    hourly_avg,
                    x='Hour',
                    y='Total',
                    title="Average Consumption by Hour of Day",
                    color='Total',
                    color_continuous_scale='Blues'
                )
                
                fig_hourly.update_layout(
                    xaxis_title="Hour of Day",
                    yaxis_title="Average Energy (kWh)",
                    height=400,
                    template='plotly_white',
                    showlegend=False
                )
                
                st.plotly_chart(fig_hourly, use_container_width=True)
            else:
                st.warning("Please select at least one load to display.")
        
        # TAB 4: Load Breakdown
        with tab4:
            st.subheader("Load Distribution Analysis")
            
            if selected_loads:
                # Pie chart for total consumption
                col1, col2 = st.columns(2)
                
                with col1:
                    total_by_load = filtered_df[selected_loads].sum()
                    
                    fig_pie = px.pie(
                        values=total_by_load.values,
                        names=total_by_load.index,
                        title="Total Energy Consumption by Load",
                        hole=0.4
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label'
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart comparison
                    fig_bar = px.bar(
                        x=total_by_load.index,
                        y=total_by_load.values,
                        title="Total Consumption Comparison",
                        labels={'x': 'Load', 'y': 'Total Energy (kWh)'},
                        color=total_by_load.values,
                        color_continuous_scale='Greens'
                    )
                    
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Monthly breakdown
                st.subheader("Monthly Consumption Breakdown")
                
                monthly_data = filtered_df.groupby('Month')[selected_loads].sum()
                
                fig_monthly = go.Figure()
                
                for load in selected_loads:
                    fig_monthly.add_trace(go.Bar(
                        name=load,
                        x=monthly_data.index,
                        y=monthly_data[load]
                    ))
                
                fig_monthly.update_layout(
                    barmode='group',
                    title="Monthly Energy Consumption by Load",
                    xaxis_title="Month",
                    yaxis_title="Energy Consumption (kWh)",
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.warning("Please select at least one load to display.")
        
        # TAB 5: Raw Data
        with tab5:
            st.subheader("Raw Data View")
            
            # Display options
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(f"Showing {len(filtered_df)} records")
            
            with col2:
                if st.button("Download CSV"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download",
                        data=csv,
                        file_name="energy_data_export.csv",
                        mime="text/csv"
                    )
            
            # Display dataframe
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=600
            )
            
            # Basic data info
            st.subheader("Dataset Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Shape:**", filtered_df.shape)
            
            with col2:
                st.write("**Date Range:**")
                st.write(f"{min_date} to {max_date}")
            
            with col3:
                st.write("**Number of Loads:**", len(load_columns))
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("""
        **Troubleshooting Tips:**
        - If you see a datetime parsing error, try increasing the 'Skip Header Rows' value in the sidebar
        - Ensure the first column (after skipped rows) contains actual date/time values
        - Check that your Excel file has numeric columns for the load data
        - Look at the sample data format in the welcome screen below
        """)
        
        # Show first few rows for debugging
        try:
            if skip_rows > 0:
                rows_to_skip = list(range(1, int(skip_rows) + 1))
                debug_df = pd.read_excel(uploaded_file, skiprows=rows_to_skip, nrows=5)
            else:
                debug_df = pd.read_excel(uploaded_file, nrows=5)
            st.subheader("First 5 rows of your file (for debugging):")
            st.dataframe(debug_df)
        except:
            pass

else:
    # Welcome screen
    st.info("Please upload an Excel file to begin analysis")
    
    st.markdown("### Expected Data Format")
    st.markdown("""
    Your Excel file should contain:
    - **Row 1**: Column headers (DateTime, Load names, etc.)
    - **Row 2** (optional): Units/metadata row - will be automatically skipped
    - **Data rows**: Actual measurements
    
    Column structure:
    - **First column**: DateTime (timestamp for each measurement)
    - **Remaining columns**: Load categories (numeric values representing energy consumption)
    - **Frequency**: Quarterly-hour intervals (15-minute periods)
    
    Example columns: DateTime, ENOC, Extraction, Supply Air, Chillers, etc.
    """)
    
    # Sample data structure
    st.markdown("### Sample Data Structure")
    sample_data = pd.DataFrame({
        'DateTime': pd.date_range('2024-01-01', periods=5, freq='15min'),
        'HVAC': [45.2, 47.8, 43.1, 46.5, 48.2],
        'Lighting': [12.3, 12.5, 12.1, 12.4, 12.6],
        'Production': [78.5, 80.2, 77.9, 79.1, 81.3],
        'Polishing': [23.4, 24.1, 22.8, 23.7, 24.5]
    })
    
    st.dataframe(sample_data, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Energy Load Profile Dashboard | Built for Schneider Electric</p>",
    unsafe_allow_html=True
)
