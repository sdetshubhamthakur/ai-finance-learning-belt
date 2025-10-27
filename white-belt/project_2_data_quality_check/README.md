# Financial Data Quality Checker

A comprehensive Python toolkit for analyzing and ensuring the quality of financial time series data. This project provides multiple statistical methods for detecting outliers, missing values, and data anomalies in financial datasets.

## 📊 Overview

This toolkit helps financial analysts, data scientists, and researchers identify data quality issues in financial time series data using robust statistical methods. It's particularly useful for:

- **Data validation** before financial modeling
- **Outlier detection** in stock price data
- **Missing value analysis** in trading datasets
- **Comparative analysis** of different outlier detection methods

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**
2. **Navigate to the project directory**
   ```bash
   cd data-quality-checker
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

5. **Install required packages**
   ```bash
   pip install pandas matplotlib seaborn numpy
   ```

### Running the Code

1. **Generate synthetic test data** (optional):
   ```bash
   python test-data-generator.py
   ```
   This creates `synthetic_financial_data.csv` with realistic financial data including intentional outliers and missing values.

2. **Run the quality checker**:
   ```bash
   python quality-checker.py
   ```

## 📁 Project Structure

```
data-quality-checker/
│
├── README.md                              # This file
├── test-data-generator.py                 # Generates synthetic financial data
├── quality-checker.py                     # Main analysis script
├── synthetic_financial_data.csv           # Sample financial dataset
├── venv/                                  # Virtual environment folder
│
└── Generated Outputs:
    ├── outliers_external_window.png       # Outlier visualization
    ├── outlier_detection_comparison.png   # Method comparison charts
    ├── row_131_focused_analysis.png       # Detailed analysis plots
    └── Various other analysis charts...
```

## 🔍 Features & Methods

### 1. Missing Value Analysis
- **Detection**: Identifies missing values across all columns
- **Visualization**: Heatmaps showing missing data patterns
- **Impact Assessment**: Quantifies the extent of missing data

### 2. Outlier Detection Methods

#### **External Window Z-Score Method** (Recommended)
- **How it works**: Uses only previous N days (default: 50) to calculate statistics
- **Advantages**: Avoids self-inclusion bias, adapts to market regime changes
- **Best for**: Real-time trading applications, regime change detection

#### **IQR (Interquartile Range) Method**
- **How it works**: Uses global quartiles to identify extreme values
- **Advantages**: Robust to non-normal distributions, simple interpretation
- **Best for**: Long-term historical analysis, non-parametric datasets

#### **Rolling Z-Score Method**
- **How it works**: Uses rolling window including current observation
- **Advantages**: Smooth detection, good for trend analysis
- **Limitations**: Can miss outliers due to self-inclusion

### 3. Visualization & Reporting
- **Time series plots** with highlighted outliers
- **Comparative analysis** charts
- **Detailed statistical summaries**
- **Actionable recommendations**

## 📈 Understanding Financial & Statistical Concepts

### Financial Terms

#### **OHLCV Data**
- **Open**: First traded price of the day
- **High**: Highest traded price of the day
- **Low**: Lowest traded price of the day
- **Close**: Last traded price of the day (most commonly analyzed)
- **Volume**: Number of shares traded

#### **Business Days**
Financial markets typically operate Monday-Friday, excluding holidays. The code uses `pd.bdate_range()` to generate realistic trading day sequences.

#### **Price Movements**
- **Normal fluctuation**: Daily price changes typically follow patterns
- **Outliers**: Unusual price movements that may indicate:
  - News events (earnings, mergers, regulatory changes)
  - Market manipulation
  - Data errors
  - Technical glitches

### Statistical Concepts

#### **Z-Score**
```
Z-Score = (Value - Mean) / Standard Deviation
```
- **Interpretation**: How many standard deviations away from the mean
- **Typical thresholds**:
  - `|Z| > 2`: Moderate outlier (~5% of data)
  - `|Z| > 3`: Strong outlier (~0.3% of data)
  - `|Z| > 4`: Extreme outlier (~0.01% of data)

#### **Interquartile Range (IQR)**
```
IQR = Q3 - Q1
Outlier if: Value < (Q1 - 1.5×IQR) OR Value > (Q3 + 1.5×IQR)
```
- **Q1**: 25th percentile
- **Q3**: 75th percentile
- **IQR**: Spread of middle 50% of data
- **1.5×IQR rule**: Standard statistical practice

#### **Rolling Windows**
- **Purpose**: Analyze data using moving time windows
- **Self-inclusion problem**: Including the current point in its own statistics can mask outliers
- **External window**: Using only previous points avoids this bias

## 🛠 Customization

### Adjusting Detection Sensitivity

#### **Z-Score Threshold**
```python
# More sensitive (detects more outliers)
external_outliers = detect_outliers_external_window(df['Close'], threshold=2.5)

# Less sensitive (detects fewer outliers)
external_outliers = detect_outliers_external_window(df['Close'], threshold=5.0)
```

#### **Window Size**
```python
# Shorter window (more responsive to recent changes)
external_outliers = detect_outliers_external_window(df['Close'], window=20)

# Longer window (more stable, less sensitive)
external_outliers = detect_outliers_external_window(df['Close'], window=100)
```

#### **IQR Multiplier**
```python
# More sensitive
iqr_outliers = detect_outliers_iqr(df['Close'], multiplier=1.0)

# Less sensitive
iqr_outliers = detect_outliers_iqr(df['Close'], multiplier=2.0)
```

### Adding Your Own Data

Replace `synthetic_financial_data.csv` with your own CSV file containing:
- **Date column**: Properly formatted dates
- **Price columns**: Open, High, Low, Close
- **Volume column**: Trading volume data

Required CSV format:
```csv
Date,Open,High,Low,Close,Volume
2019-01-01,100.0,102.0,99.0,101.0,50000
2019-01-02,101.0,103.0,100.0,102.5,55000
...
```

## 📊 Interpreting Results

### Expected Outlier Rates
- **Healthy dataset**: 1-5% outliers
- **High volatility periods**: 5-10% outliers
- **Data quality issues**: >10% outliers

### Common Patterns

#### **Market Events**
- **Earnings announcements**: Sudden price jumps
- **Market crashes**: Extreme negative movements
- **News events**: Sharp directional moves

#### **Data Quality Issues**
- **Systematic errors**: Regular patterns in outliers
- **Missing data**: Gaps in time series
- **Price adjustments**: Stock splits, dividends

### Actionable Recommendations

#### **For Detected Outliers**
1. **Investigate the date**: Check for news, earnings, or market events
2. **Verify data source**: Confirm accuracy with alternative data providers
3. **Consider context**: Some outliers are legitimate (market reactions)
4. **Document decisions**: Keep records of outliers and their treatment

#### **For Missing Data**
1. **Forward fill**: Use previous value (for stable periods)
2. **Interpolation**: Linear or more sophisticated methods
3. **Exclusion**: Remove incomplete records if minimal impact
4. **Source verification**: Check if data should exist for those dates

## 🔧 Troubleshooting

### Common Issues

#### **Import Errors**
```bash
# Solution: Install missing packages
pip install pandas matplotlib seaborn numpy
```

#### **Date Parsing Issues**
```python
# Ensure proper date format in CSV
df = pd.read_csv('your_data.csv', parse_dates=['Date'])
```

#### **Memory Issues with Large Datasets**
```python
# Process in chunks for large files
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### Performance Tips

1. **Use appropriate window sizes**: Balance sensitivity vs. computation time
2. **Filter data**: Focus on specific date ranges if needed
3. **Vectorized operations**: Pandas operations are optimized for performance
4. **Save intermediate results**: Cache processed data for repeated analysis

## 📚 Further Reading

### Statistical Methods
- [Z-Score Standardization](https://en.wikipedia.org/wiki/Standard_score)
- [Interquartile Range](https://en.wikipedia.org/wiki/Interquartile_range)
- [Outlier Detection Methods](https://en.wikipedia.org/wiki/Outlier)

### Financial Data Analysis
- [Time Series Analysis](https://en.wikipedia.org/wiki/Time_series)
- [Financial Data Quality](https://www.investopedia.com/terms/d/data-analytics.asp)
- [Market Microstructure](https://en.wikipedia.org/wiki/Market_microstructure)

### Python Libraries
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

## 🤝 Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

## 📧 Support

For questions or issues:
- Review this README thoroughly
- Check the code comments for implementation details
- Verify your data format matches the expected structure
- Ensure all dependencies are properly installed

---

**Happy analyzing! 📈**
