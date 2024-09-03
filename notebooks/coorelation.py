import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob

# Define the ticker and the time period
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'

# Fetch the stock data
aapl_data = yf.download(ticker, start=start_date, end=end_date)

# Set the random seed for reproducibility
np.random.seed(0)

# Create a DataFrame with dates from the AAPL data
dates = aapl_data.index
text_data = pd.DataFrame(dates, columns=['Date'])

# Generate random headlines
headline_samples = [
    "AAPL hits record high",
    "Concerns over AAPL's future growth",
    "AAPL to unveil new product next month",
    "AAPL reports earnings that exceed forecasts",
    "Market downturn affects AAPL",
    "AAPL invests in renewable energy",
    "New AAPL CEO announced",
    "AAPL faces regulatory scrutiny",
    "AAPL rumored to acquire a tech startup",
    "AAPL's market share grows"
]

# Assign a random headline to each date
text_data['Headline'] = np.random.choice(headline_samples, size=len(text_data))

# Calculate sentiment
def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity

text_data['Sentiment'] = text_data['Headline'].apply(calculate_sentiment)

# Aligning sentiment scores with the stock closing prices on the same date
combined_data = text_data.set_index('Date').join(aapl_data[['Close', 'Volume']])

# Calculate additional stock metrics (SMA, RSI, etc.)
combined_data['SMA_50'] = aapl_data['Close'].rolling(window=50).mean()
combined_data['SMA_200'] = aapl_data['Close'].rolling(window=200).mean()
combined_data['Daily Returns'] = aapl_data['Close'].pct_change()

# Calculating correlations
correlation_sentiment_close = combined_data['Sentiment'].corr(combined_data['Close'])
correlation_sentiment_returns = combined_data['Sentiment'].corr(combined_data['Daily Returns'])
correlation_sentiment_sma_50 = combined_data['Sentiment'].corr(combined_data['SMA_50'])
correlation_sentiment_sma_200 = combined_data['Sentiment'].corr(combined_data['SMA_200'])

# Display correlations
print("Correlation between sentiment and closing price:", correlation_sentiment_close)
print("Correlation between sentiment and daily returns:", correlation_sentiment_returns)
print("Correlation between sentiment and 50-day SMA:", correlation_sentiment_sma_50)
print("Correlation between sentiment and 200-day SMA:", correlation_sentiment_sma_200)

# Visualize correlations (optional)
import seaborn as sns
import matplotlib.pyplot as plt

# Creating a heatmap to visualize correlations
correlation_matrix = combined_data[['Sentiment', 'Close', 'Daily Returns', 'SMA_50', 'SMA_200']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
