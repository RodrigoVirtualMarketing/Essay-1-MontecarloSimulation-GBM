import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import seaborn as sns
import warnings

# Configuración de seguridad: Evitar que warnings revelen rutas locales
warnings.filterwarnings('ignore', category=FutureWarning)
# Filtro general para limpiar la salida de paths del sistema
def warning_cleaner(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}")

warnings.showwarning = warning_cleaner

"""# Un Ticker"""

df = yf.download("AAPL", start='1950-01-01', group_by='ticker', auto_adjust=True)

## df = df.sort_index(ascending=True)
df.head(1)

df["Returns"]=df[('AAPL', 'Close')].pct_change()
df.head(3)

num_simulations=10000
num_days=252 #predict horizon
last_price=df[('AAPL', 'Close')].iloc[-1]

simulation_df=np.zeros((num_days,num_simulations))

mu=df["Returns"].mean()
sigma=df["Returns"].std()

for i in range(num_simulations):
  price_list=[last_price]
  for j in range(num_days):
    price=price_list[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal())
    price_list.append(price)
  simulation_df[:,i]=price_list[1:]

final_prices=simulation_df[-1,:]
median_final_prices=np.median(final_prices)

most_likely_price_index=np.argmin(np.abs(final_prices-median_final_prices))
most_likely_price_simulation=simulation_df[:,most_likely_price_index]

plt.figure(figsize=(10,7))
plt.plot(simulation_df)
plt.title("Monte Carlo Simulation")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

print(f"The most likely PLTR simulation final price is: {most_likely_price_simulation[-1]}")

print(f"Todays PLTR price is: {df.tail(1)}")

"""# Varios Tickers"""

mis_tickers = ["AAPL",
               "MSFT",
               "GOOGL",
               "AMZN",
               "META",
               "PLTR"]

mis_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "PLTR"]
df2 = yf.download(mis_tickers, start='1950-01-01', group_by='ticker', auto_adjust=True)

for ticker in mis_tickers:
    df2[(ticker, 'Returns')] = df2[(ticker, 'Close')].pct_change()

num_simulations=10000
num_days=22 #predict horizon

for ticker in mis_tickers:
  last_price=df2[(ticker, 'Close')].iloc[-1]


simulation_df=np.zeros((num_days,num_simulations))

# Correctly access all 'Returns' columns from the MultiIndex
returns_data = df2.xs('Returns', level=1, axis=1)
mu = returns_data.mean().mean() # Mean of all ticker returns
sigma = returns_data.std().mean() # Average std deviation across tickers

# Dictionaries to store results for each ticker
all_ticker_simulation_dfs = {}
all_ticker_most_likely_prices = {}
all_ticker_median_final_prices = {}

for current_ticker in mis_tickers:
    print(f"--- Simulating for {current_ticker} ---")

    # Get ticker-specific parameters
    last_price = df2[(current_ticker, 'Close')].iloc[-1]

    # Calculate mu and sigma for the current ticker
    ticker_returns = df2[(current_ticker, 'Returns')].dropna() # Drop NaN for calculation
    mu = ticker_returns.mean()
    sigma = ticker_returns.std()

    # Initialize simulation_df for the current ticker
    simulation_df = np.zeros((num_days, num_simulations))

    # Run Monte Carlo simulation
    for i in range(num_simulations):
        price_list = [last_price]
        for j in range(num_days):
            price = price_list[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal())
            price_list.append(price)
        simulation_df[:, i] = price_list[1:]

    # Analyze results
    final_prices = simulation_df[-1, :]
    median_final_prices = np.median(final_prices)
    most_likely_price_index = np.argmin(np.abs(final_prices - median_final_prices))
    most_likely_price_simulation = simulation_df[:, most_likely_price_index]

    # Store results
    all_ticker_simulation_dfs[current_ticker] = simulation_df
    all_ticker_most_likely_prices[current_ticker] = most_likely_price_simulation[-1]
    all_ticker_median_final_prices[current_ticker] = median_final_prices

    # Plotting for the current ticker
    plt.figure(figsize=(10, 7))
    plt.plot(simulation_df)
    plt.plot(most_likely_price_simulation, color='red', linestyle='--', label=f'Most Likely Path ({current_ticker})')
    plt.title(f"Monte Carlo Simulation for {current_ticker}")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Print results for the current ticker
    print(f"The most likely {current_ticker} simulation final price is: {all_ticker_most_likely_prices[current_ticker]:.2f}")
    print(f"The median {current_ticker} simulation final price is: {all_ticker_median_final_prices[current_ticker]:.2f}")
    print("-" * 50)

# Consolidated summary of results for all tickers
print(f"\n--- Consolidated Analysis for All Tickers (Prediction Horizon: {num_days} days, Simulations: {num_simulations}) ---")
for ticker in mis_tickers:
    simulated_price = all_ticker_most_likely_prices[ticker]
    current_price = df2[(ticker, 'Close')].iloc[-1]
    print(f"{ticker}: Most Likely Simulated Final Price: {simulated_price:.3f} | Today's Current Price: {current_price:.3f}")

"""## Distribución de Precios Finales Simulados"""

for ticker in mis_tickers:
    plt.figure(figsize=(10, 6))
    final_prices = all_ticker_simulation_dfs[ticker][-1, :]
    sns.histplot(final_prices, bins=50, kde=True, color='skyblue')
    plt.title(f'Distribución de Precios Finales Simulados para {ticker}')
    plt.xlabel('Precio Final')
    plt.ylabel('Frecuencia')
    plt.axvline(all_ticker_median_final_prices[ticker], color='red', linestyle='--', label=f"Mediana: {all_ticker_median_final_prices[ticker]:.2f}")
    plt.axvline(df2[(ticker, 'Close')].iloc[-1], color='green', linestyle=':', label=f"Precio Actual: {df2[(ticker, 'Close')].iloc[-1]:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.show()

"""## Intervalo de Confianza del 95% para los Precios Finales Simulados"""

print("--- 95% Confidence Interval for Simulated Final Prices ---")
for ticker in mis_tickers:
    final_prices = all_ticker_simulation_dfs[ticker][-1, :]
    lower_bound = np.percentile(final_prices, 2.5)
    upper_bound = np.percentile(final_prices, 97.5)
    print(f"{ticker}: 95% CI: [{lower_bound:.3f}, {upper_bound:.3f}]")

plt.figure(figsize=(12, 7))

confidence_data = []
for ticker in mis_tickers:
    final_prices = all_ticker_simulation_dfs[ticker][-1, :]
    lower_bound = np.percentile(final_prices, 2.5)
    upper_bound = np.percentile(final_prices, 97.5)
    median_price = all_ticker_median_final_prices[ticker]
    current_price = df2[(ticker, 'Close')].iloc[-1]
    confidence_data.append({
        'Ticker': ticker,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Median Price': median_price,
        'Current Price': current_price
    })

confidence_df = pd.DataFrame(confidence_data)

# Updated: Using linestyle='none' instead of deprecated join=False
sns.pointplot(x='Ticker', y='Median Price', data=confidence_df, linestyle='none', color='red', markers='o', capsize=0.2, label='Median Simulated Price')
plt.errorbar(x=confidence_df['Ticker'], y=confidence_df['Median Price'], yerr=[confidence_df['Median Price'] - confidence_df['Lower Bound'], confidence_df['Upper Bound'] - confidence_df['Median Price']], fmt='none', capsize=5, color='gray', linewidth=2, label='95% Confidence Interval')

# Add current prices as separate points
sns.scatterplot(x='Ticker', y='Current Price', data=confidence_df, color='blue', marker='X', s=200, label='Current Price')

plt.title('Intervalos de Confianza del 95% para Precios Finales Simulados')
plt.xlabel('Ticker')
plt.ylabel('Precio')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
