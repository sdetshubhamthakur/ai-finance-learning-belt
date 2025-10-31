# 📈 Stock News Sentiment Analysis Tool

A comprehensive Python tool for analyzing market sentiment through financial news headlines. This project fetches real-time news data from multiple RSS sources and performs sentiment analysis to provide actionable insights for investment decisions.

## 🚀 Overview

This tool combines financial news aggregation with natural language processing to analyze market sentiment for any publicly traded stock. By processing hundreds of news headlines from multiple sources, it provides quantitative sentiment metrics that can inform trading strategies and market analysis.

## 🔧 Features

### 📰 Multi-Source News Aggregation
- **Yahoo Finance RSS**: Stock-specific financial news
- **Google News RSS**: Comprehensive news coverage with multiple search strategies
- **Duplicate Detection**: Intelligent removal of duplicate headlines
- **Minimum Guarantee**: Ensures collection of at least 100 unique headlines per stock

### 🧠 Advanced Sentiment Analysis
- **VADER Sentiment Analysis**: Industry-standard lexicon-based sentiment analysis
- **Multi-Class Classification**: Positive, Negative, and Neutral sentiment labels
- **Compound Scoring**: Normalized sentiment scores (-1 to +1)
- **Temporal Analysis**: Sentiment trends over time

### 📊 Data Visualization & Export
- **CSV Export**: Structured data output for further analysis
- **Time-Series Visualization**: Sentiment distribution charts over time
- **Statistical Summaries**: Comprehensive sentiment distribution metrics
- **Source Attribution**: Track news sources and their sentiment patterns

## 🏗️ Technical Architecture

### Data Pipeline
```
RSS Feeds → Data Extraction → Deduplication → Sentiment Analysis → Visualization → Export
```

### Core Components

1. **News Fetcher (`get_stock_news_rss`)**
   - Multi-source RSS feed aggregation
   - Intelligent company name mapping
   - Error handling and retry logic
   - Duplicate detection algorithm

2. **Sentiment Analyzer (VADER)**
   - Lexicon-based sentiment scoring
   - Social media and financial news optimized
   - Real-time processing capability

3. **Data Processor**
   - Pandas-based data manipulation
   - Temporal grouping and analysis
   - Statistical aggregation

4. **Visualization Engine**
   - Matplotlib-based charting
   - Time-series analysis
   - Export-ready graphics

## 💼 Business Use Cases

### 1. **Algorithmic Trading**
- **Signal Generation**: Use sentiment scores as trading signals
- **Risk Management**: Identify negative sentiment spikes before market downturns
- **Entry/Exit Points**: Combine with technical analysis for better timing

### 2. **Investment Research**
- **Due Diligence**: Assess public perception before major investments
- **Sector Analysis**: Compare sentiment across industry competitors
- **Event Impact**: Measure sentiment changes around earnings, launches, or crises

### 3. **Portfolio Management**
- **Risk Assessment**: Monitor sentiment deterioration in holdings
- **Diversification**: Balance portfolio based on sentiment correlation
- **Performance Attribution**: Understand sentiment's role in returns

### 4. **Corporate Intelligence**
- **Brand Monitoring**: Track public perception of company actions
- **Competitive Analysis**: Compare sentiment against competitors
- **Crisis Management**: Early detection of negative sentiment trends

### 5. **Market Research**
- **Trend Identification**: Spot emerging market themes through sentiment
- **Consumer Behavior**: Understand market psychology and investor sentiment
- **Regulatory Impact**: Assess sentiment around policy changes

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
pip (Python package manager)
```

### Required Packages
```bash
pip install feedparser pandas requests nltk matplotlib
```

### NLTK Setup
```python
import nltk
nltk.download('vader_lexicon')
```

## 🚦 Usage

### Basic Usage
```python
# Set your target stock symbol
stock = "AAPL"  # Apple Inc.

# Fetch news and analyze sentiment
df = get_stock_news_rss(stock, min_headlines=100)
```

### Supported Stock Symbols
The tool includes intelligent mapping for major stocks:
- **AAPL** → Apple
- **MSFT** → Microsoft
- **GOOGL** → Google
- **AMZN** → Amazon
- **TSLA** → Tesla
- **META** → Meta
- **NVDA** → NVIDIA
- **NFLX** → Netflix
- **CRM** → Salesforce
- **ORCL** → Oracle

### Custom Company Names
```python
# For stocks not in the mapping
df = get_stock_news_rss("XYZ", company_name="XYZ Corporation", min_headlines=100)
```

## 📊 Output Analysis

### Sentiment Scores
- **Positive**: > 0.05 (Bullish sentiment)
- **Neutral**: -0.05 to 0.05 (Mixed/unclear sentiment)
- **Negative**: < -0.05 (Bearish sentiment)

### Key Metrics
- **Total Headlines**: Number of unique news articles processed
- **Sentiment Distribution**: Percentage breakdown by sentiment class
- **Source Diversity**: Number of different news sources
- **Time Range**: Coverage period of the news data
- **Average Sentiment**: Overall market sentiment score

## 📈 Interpreting Results

### Investment Signals

#### Bullish Indicators
- **High Positive Sentiment** (>60% positive headlines)
- **Increasing Positive Trend** over time
- **Low Negative Sentiment** (<20% negative headlines)

#### Bearish Indicators
- **High Negative Sentiment** (>50% negative headlines)
- **Declining Sentiment Trend** over time
- **Crisis-related Headlines** (earnings misses, legal issues)

#### Neutral/Hold Indicators
- **Balanced Sentiment** (roughly equal positive/negative)
- **Stable Sentiment Trend** over time
- **Low News Volume** (insufficient data for strong signals)

### Risk Considerations
- **News Quality**: Not all sources have equal credibility
- **Market Lag**: Sentiment may lag or lead actual price movements
- **Event Sensitivity**: Major events can skew short-term sentiment
- **Sample Bias**: RSS feeds may not capture all relevant news

## 🔍 Technical Details

### RSS Sources Strategy
1. **Yahoo Finance**: Authoritative financial news with stock focus
2. **Google News (Symbol)**: Broad coverage using stock ticker
3. **Google News (Company)**: Company-specific news coverage
4. **Google News (Stock-focused)**: Investment and trading news
5. **Google News (Financial)**: Earnings and financial performance news

### Sentiment Analysis Methodology
- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- **Compound Score**: Primary sentiment metric (-1 to +1)
- **Lexicon-based**: No training data required, works out-of-the-box
- **Social Media Optimized**: Handles modern language patterns

### Data Quality Measures
- **Duplicate Removal**: Exact match deduplication
- **Date Validation**: Temporal consistency checks
- **Source Tracking**: Full attribution for each headline
- **Error Handling**: Graceful failure with partial results

## 📁 Output Files

### CSV Export
- **Filename**: `{SYMBOL}_news_sentiment.csv`
- **Contents**: Headlines, timestamps, sources, sentiment scores, labels

### Visualization
- **Filename**: `{SYMBOL}_sentiment_over_months.png`
- **Type**: Stacked bar chart showing sentiment distribution over time

## 🚀 Future Enhancements

### Planned Features
- **Real-time Streaming**: Live sentiment monitoring
- **Machine Learning**: Advanced sentiment models (BERT, FinBERT)
- **Multi-language Support**: Non-English news sources
- **Social Media Integration**: Twitter, Reddit sentiment
- **Price Correlation**: Direct stock price impact analysis

### Advanced Analytics
- **Sentiment Volatility**: Measure sentiment stability
- **Source Reliability Scoring**: Weight sources by accuracy
- **Event Detection**: Automatic identification of significant events
- **Sector Comparison**: Cross-industry sentiment analysis

## 📝 License & Disclaimer

This tool is for educational and research purposes. Not financial advice.

### Important Notes
- **Rate Limiting**: Respects RSS feed rate limits with delays
- **Data Accuracy**: News sentiment doesn't guarantee market performance
- **Risk Warning**: All investments carry risk; past performance doesn't indicate future results
- **Compliance**: Ensure compliance with data source terms of service

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional RSS sources
- Enhanced sentiment models
- Better visualization options
- Performance optimizations
- Documentation improvements

## 📞 Support

For issues or questions:
1. Check existing documentation
2. Review error messages for common issues
3. Verify RSS feed availability
4. Ensure all dependencies are installed

---

**Built for the Modern Financial Analyst** 📊✨
