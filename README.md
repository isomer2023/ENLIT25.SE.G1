# Energy Load Profile Dashboard

Interactive web dashboard for visualizing and analyzing quarterly-hour energy consumption data for industrial sites.

## Features

- **Time Series Analysis**: View consumption trends over time with interactive line charts and stacked area plots
- **Heatmaps**: Identify consumption patterns by hour, day, and week
- **Statistical Analysis**: Comprehensive statistics including totals, averages, peak demands, and trends
- **Load Breakdown**: Pie charts and comparative analysis of different load categories
- **Data Export**: Download filtered data as CSV for further analysis
- **Interactive Filtering**: Date range selection and load category filtering

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or navigate to this directory**:
   ```bash
   cd "c:\Users\franc\Desktop\ENLIT\Model"
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**:
   - The dashboard will automatically open in your default browser
   - If not, navigate to: `http://localhost:8501`

3. **Upload your data**:
   - Click on "Browse files" in the sidebar
   - Select your Excel file containing the energy consumption data

## Data Format

Your Excel file should have the following structure:

| DateTime | HVAC | Lighting | Production | Polishing | ... |
|----------|------|----------|------------|-----------|-----|
| 2024-01-01 00:00:00 | 45.2 | 12.3 | 78.5 | 23.4 | ... |
| 2024-01-01 00:15:00 | 47.8 | 12.5 | 80.2 | 24.1 | ... |
| 2024-01-01 00:30:00 | 43.1 | 12.1 | 77.9 | 22.8 | ... |

**Requirements**:
- First column must be a DateTime field
- Remaining columns should be numeric load categories
- Data should be in quarterly-hour (15-minute) intervals

## Dashboard Tabs

### 1. Time Series
- **Line charts**: Individual load consumption over time
- **Stacked area charts**: Total consumption breakdown

### 2. Heatmaps
- **Hourly patterns**: Consumption by hour and date
- **Day of week analysis**: Average consumption patterns by weekday

### 3. Statistics
- **Key metrics**: Total consumption, averages, peak demand
- **Detailed statistics table**: Per-load analysis
- **Daily trends**: Day-by-day consumption patterns
- **Hourly averages**: Typical consumption by hour

### 4. Load Breakdown
- **Pie charts**: Proportional consumption by load
- **Bar charts**: Comparative analysis
- **Monthly breakdown**: Month-over-month comparison

### 5. Raw Data
- **Data table**: View and explore filtered data
- **Export functionality**: Download data as CSV

## Customization

The dashboard uses Schneider Electric's brand colors and can be further customized by editing `app.py`:

- Modify color schemes in the Plotly chart configurations
- Adjust the layout in the Streamlit page configuration
- Add custom metrics or calculations as needed

## Troubleshooting

**Dashboard won't start:**
- Ensure all packages are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

**Can't upload file:**
- Verify the file is in Excel format (.xlsx or .xls)
- Check that the first column contains datetime values
- Ensure numeric columns don't contain text or errors

**Charts not displaying:**
- Verify at least one load is selected in the sidebar filters
- Check date range includes data from your file

## Built For

**Schneider Electric** - Industrial Site Energy Analysis

## Support

For issues or questions, please contact your project administrator.

---

*Built with Streamlit, Plotly, and Pandas*
