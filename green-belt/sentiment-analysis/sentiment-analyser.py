# import nltk  
# nltk.download('vader_lexicon')


# import at least 100 news headlines for a given stock using RSS feeds
import feedparser
import pandas as pd
from datetime import datetime
import time
import requests
from urllib.parse import quote

def get_stock_news_rss(symbol, company_name=None, min_headlines=100):
    """
    Fetch news headlines for a given stock symbol using multiple RSS feeds
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
        company_name (str): Company name for better search results (e.g., 'Apple', 'Microsoft')
        min_headlines (int): Minimum number of headlines to fetch (default: 100)
    
    Returns:
        pandas.DataFrame: DataFrame with headlines and metadata
    """
    
    # Company name mapping for better search results
    company_names = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft', 
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla',
        'META': 'Meta',
        'NVDA': 'NVIDIA',
        'NFLX': 'Netflix',
        'CRM': 'Salesforce',
        'ORCL': 'Oracle'
    }
    
    if not company_name:
        company_name = company_names.get(symbol, symbol)
    
    all_headlines = []
    all_dates = []
    all_sources = []
    all_links = []
    
    print(f"Searching for news about {symbol} ({company_name})...")
    
    # RSS Feed sources
    rss_sources = [
        {
            'name': 'Yahoo Finance - Symbol',
            'url': f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US'
        },
        {
            'name': 'Google News - Symbol', 
            'url': f'https://news.google.com/rss/search?q={quote(symbol)}&hl=en-US&gl=US&ceid=US:en'
        },
        {
            'name': 'Google News - Company',
            'url': f'https://news.google.com/rss/search?q={quote(company_name)}&hl=en-US&gl=US&ceid=US:en'
        },
        {
            'name': 'Google News - Stock',
            'url': f'https://news.google.com/rss/search?q={quote(company_name + " stock")}&hl=en-US&gl=US&ceid=US:en'
        },
        {
            'name': 'Google News - Financial',
            'url': f'https://news.google.com/rss/search?q={quote(company_name + " earnings financial")}&hl=en-US&gl=US&ceid=US:en'
        }
    ]
    
    for source in rss_sources:
        try:
            print(f"Fetching from {source['name']}...")
            
            # Parse RSS feed
            feed = feedparser.parse(source['url'])
            
            if feed.entries:
                for entry in feed.entries:
                    # Extract headline
                    title = entry.get('title', 'No title available')
                    
                    # Skip if this headline already exists (avoid duplicates)
                    if title not in all_headlines:
                        all_headlines.append(title)
                        
                        # Extract date
                        pub_date = entry.get('published_parsed', None)
                        if pub_date:
                            date = datetime(*pub_date[:6])
                            all_dates.append(date)
                        else:
                            all_dates.append(None)
                        
                        # Extract source and link
                        all_sources.append(source['name'])
                        all_links.append(entry.get('link', ''))
                
                print(f"  - Found {len(feed.entries)} articles")
            else:
                print(f"  - No articles found")
            
            # Add small delay to be respectful to servers
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  - Error fetching from {source['name']}: {str(e)}")
            continue
    
    # Remove duplicates while preserving order
    seen_headlines = set()
    unique_headlines = []
    unique_dates = []
    unique_sources = []
    unique_links = []
    
    for i, headline in enumerate(all_headlines):
        if headline not in seen_headlines:
            seen_headlines.add(headline)
            unique_headlines.append(headline)
            unique_dates.append(all_dates[i])
            unique_sources.append(all_sources[i])
            unique_links.append(all_links[i])
    
    # Create DataFrame
    df = pd.DataFrame({
        'headline': unique_headlines,
        'datetime': unique_dates,
        'source': unique_sources,
        'link': unique_links
    })
    
    # Sort by date (newest first, handle None values)
    df = df.sort_values('datetime', ascending=False, na_position='last').reset_index(drop=True)
    
    # Check if we have enough headlines
    if len(df) < min_headlines:
        print(f"\nWarning: Only {len(df)} unique headlines found, which is less than the minimum of {min_headlines}")
        print("Consider:")
        print("1. Using a more popular stock symbol")
        print("2. Checking if the company name mapping is correct")
        print("3. Trying again later as news availability varies")
    else:
        print(f"\n✓ Successfully found {len(df)} headlines (minimum {min_headlines} required)")
    
    return df

# Example usage
stock = "META"  # Example: Meta Platforms, Inc.
print(f"Fetching news headlines for {stock}...")

df = get_stock_news_rss(stock, min_headlines=100)

if not df.empty:
    print(f"\nSuccessfully fetched {len(df)} news headlines for {stock}")
    print("\nFirst 5 headlines:")
    print(df[['headline', 'datetime', 'source']].head())
    
    # Display some statistics
    print(f"\nNews sources found: {df['source'].nunique()}")
    if not df['datetime'].isna().all():
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Show source distribution
    print(f"\nHeadlines by source:")
    print(df['source'].value_counts())
else:
    print("No news headlines found.")


# Sentiment analysis for each of the headlines using VADER
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis
df['sentiment'] = df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Define labels based on sentiment scores
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral')

# Display sentiment scores
print(f"\nSentiment scores:")
print(df[['headline', 'datetime', 'source', 'sentiment', 'sentiment_label']].head())

# Summary of sentiment distribution
print(f"\nSentiment distribution:")
print(df['sentiment_label'].value_counts())

# Save to CSV
output_file = f"{stock}_news_sentiment.csv"
df.to_csv(output_file, index=False)
print(f"Sentiment analysis results saved to {output_file}")

# Aggregate and plot the sentiment distribution over time
import matplotlib.pyplot as plt

# Convert datetime to pandas datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Group by month and sentiment label
sentiment_over_time = df.groupby([df['datetime'].dt.to_period('M'), 'sentiment_label']).size().unstack(fill_value=0)

# Plot
sentiment_over_time.plot(kind='bar', stacked=True, figsize=(12, 7), color=['green', 'red', 'gray'], alpha=0.7)
plt.title(f"Sentiment Distribution Over Time for {stock}")
plt.xlabel("Month")
plt.ylabel("Number of Headlines")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig(f"{stock}_sentiment_over_months.png")
plt.show()