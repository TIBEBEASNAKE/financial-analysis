import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data from CSV
file_path = 'data/your_financial_news_data.csv'  # Update with your file path
data = pd.read_csv(file_path, parse_dates=['Date'])

# Exploratory Data Analysis (EDA)

# 1. Descriptive Statistics: Analyze headline lengths
data['Headline_Length'] = data['headline'].apply(len)
print(data['Headline_Length'].describe())

# 2. Count Articles per Publisher
publisher_counts = data['publisher'].value_counts()
print("Top Publishers:\n", publisher_counts.head())

# 3. Sentiment Analysis: This will require an additional NLP step, but we'll start by printing headlines for manual inspection
print("Sample Headlines:\n", data['headline'].head())

# 4. Time Series Analysis: Count headlines per day
daily_headlines = data.groupby('Date').size()

# Create a DataFrame for time series analysis
time_series_data = pd.DataFrame({'Date': daily_headlines.index, 'Headlines_Count': daily_headlines.values})
time_series_data.set_index('Date', inplace=True)

# Decompose the time series data to observe trends, seasonality, and residuals
decomposition = seasonal_decompose(time_series_data['Headlines_Count'], model='additive', period=30)

# Plotting the components
plt.figure(figsize=(14, 10))

plt.subplot(411)
plt.plot(time_series_data['Headlines_Count'], label='Original', color='blue')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(decomposition.trend, label='Trend', color='orange')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal', color='green')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(decomposition.resid, label='Residual/Irregular', color='red')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()



#####################################
#########################
###########
# eda_module.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

class FinancialNewsEDA:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.time_series_data = None

    def load_data(self):
        """Load the financial news data from a CSV file."""
        self.data = pd.read_csv(self.file_path, parse_dates=['Date'])
        print("Data loaded successfully.")

    def descriptive_statistics(self):
        """Perform descriptive statistics on the dataset."""
        self.data['Headline_Length'] = self.data['headline'].apply(len)
        print("Descriptive statistics for headline lengths:\n", self.data['Headline_Length'].describe())

    def count_articles_per_publisher(self):
        """Count the number of articles per publisher."""
        publisher_counts = self.data['publisher'].value_counts()
        print("Top Publishers:\n", publisher_counts.head())

    def sample_headlines(self):
        """Display a sample of headlines."""
        print("Sample Headlines:\n", self.data['headline'].head())

    def time_series_analysis(self):
        """Perform time series analysis on the count of headlines per day."""
        daily_headlines = self.data.groupby('Date').size()
        self.time_series_data = pd.DataFrame({'Date': daily_headlines.index, 'Headlines_Count': daily_headlines.values})
        self.time_series_data.set_index('Date', inplace=True)

        decomposition = seasonal_decompose(self.time_series_data['Headlines_Count'], model='additive', period=30)

        # Plotting the components
        plt.figure(figsize=(14, 10))

        plt.subplot(411)
        plt.plot(self.time_series_data['Headlines_Count'], label='Original', color='blue')
        plt.legend(loc='upper left')

        plt.subplot(412)
        plt.plot(decomposition.trend, label='Trend', color='orange')
        plt.legend(loc='upper left')

        plt.subplot(413)
        plt.plot(decomposition.seasonal, label='Seasonal', color='green')
        plt.legend(loc='upper left')

        plt.subplot(414)
        plt.plot(decomposition.resid, label='Residual/Irregular', color='red')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

    def run_eda(self):
        """Run all EDA functions."""
        self.load_data()
        self.descriptive_statistics()
        self.count_articles_per_publisher()
        self.sample_headlines()
        self.time_series_analysis()
######################
###############
###############
# main_analysis.py

from eda_module import FinancialNewsEDA

# File path to your financial news data CSV file
file_path = 'data/your_financial_news_data.csv'  # Update with your actual file path

# Initialize the EDA class with the file path
eda = FinancialNewsEDA(file_path)

# Run the EDA process
eda.run_eda()



