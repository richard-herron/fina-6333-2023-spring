#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 5 - Simulations

# In finance, we typically perform simulations with [Monte Carlo methods](https://en.wikipedia.org/wiki/Monte_Carlo_method).
# Monte Carlo methods are used throughout science, engineering, and mathematics, and they are especially useful in finance due to the randomness of asset prices.
# This lecture notebook provides an introduction to [Monte Carlo methods in finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance).
# 
# 
# The basic idea behind Monte Carlo methods in finance is to create many possible paths of financial variables (e.g., stock prices, interest rates, exchange rates) based on their historical behavior, statistical properties, and any other relevant factors.
# We use these paths to simulate the performance of financial instruments or portfolios, and then we aggregate these results to estimate metrics of interest.
# This approach helps us model and analyze complex financial problems that may be difficult or impossible to solve analytically.
# 
# 
# We could spend a semester (or more) on simulations and Monte Carlo methods.
# In this lecture notebook, we will limit our focus to:
# 
# 1. Option pricing: Monte Carlo methods can be used to estimate the fair value of financial derivatives, such as options and warrants. By simulating the potential future paths of the underlying asset and calculating the payoffs of the derivative at each path, the expected payoff can be computed and discounted to present value to determine the option's price.
# 1. Value at Risk (VaR): Monte Carlo simulations can be used to compute VaR, a widely used risk management metric that estimates the potential loss in the value of a portfolio over a specific time horizon, given a certain level of confidence (e.g., 95% or 99%). By simulating numerous scenarios and observing the distribution of potential losses, the VaR can be calculated as the threshold below which a certain percentage of losses fall.
# 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('precision', '2')
pd.options.display.float_format = '{:.2f}'.format


# In[3]:


import yfinance as yf
import pandas_datareader as pdr
import requests_cache
session = requests_cache.CachedSession()


# ## Option Pricing

# We can use Monte Carlo methods to value stock options.
# First, we simulate several hundred or thousand possible (but random) price paths for the underlying stock.
# Then, we calculate the payoff for each path.
# On some paths, the option will expire "in the money" with $S_T > K$ and pay $S_T - K$.
# On other paths, the option will expire "out of the money" with $S_T < K$ and pay 0.
# We average these payoffs and discount them to today.
# The present value of these payoffs is the option price.
# This is an illustrative example, and there is a lot of depth to [Monte Carlo methods for option pricing](https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing).

# ### Simulating Stock Prices

# We can simulate stock prices with the following stochastic differential equation (SDE) for Geometric Brownian Motion (GBM): $dS = \mu S dt + \sigma S dW_t$.
# GBM does not account for mean-reversion and time-dependent volatility.
# So GBM is often used for stocks and not for bond prices, which tend to display long-term reversion to the face value.
# In the GBM SDE:
# 
# 1. $S$ is the stock price
# 1. $\mu$ is the drift coefficient (i.e., instantaneous expected return)
# 1. $\sigma$ is the diffusion coefficient (i.e., volatility of the drift)
# 1. $W_t$ is the Wiener Process or Brownian Motion
# 
# The GBM SDE has a closed-form solution: $S(t) = S_0 \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)t + \sigma W_t\right)$.
# We can apply this closed form solution recursively: $S(t_{i+1}) = S(t_i) \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)(t_{i+1} - t_i) + \sigma \sqrt{t_{i+1} - t_i} Z_{i+1}\right)$.
# Here, $Z_i$ is a Standard Normal random variable ($dW_t$ are independent and normally distributed) and $i = 0, \ldots, T-1$ is the time index.
# 
# We can use this closed form solution to simulate stock prices for AAPL.

# In[4]:


aapl = (
    yf.download(tickers='AAPL', progress=False)
    .assign(
        Date=lambda x: x.index.tz_localize(None),
        Return=lambda x: x['Adj Close'].pct_change()
    )
    .set_index('Date')
    .rename_axis(columns=['Variable'])
)


# In[5]:


aapl.describe()


# We will use returns from 2021 to predict prices in 2022.

# In[6]:


train = aapl.loc['2021']
test = aapl.loc['2022']


# We will use the following function to simulate price paths.
# Throughout this lecture notebook, we will use one-trading-day steps (i.e., `dt=1`).

# In[7]:


def simulate_gbm(S_0, mu, sigma, n_steps, dt=1, seed=42):
    '''
    Function to simulate stock prices following Geometric Brownian Motion (GBM).
    
    Parameters
    ------------
    S_0 : float
        Initial stock price
    mu : float
        Drift coefficient
    sigma : float
        Diffusion coefficient
    n_steps : int
        Length of the forecast horizon in time increments, so T = n_steps * dt
    dt : int
        Time increment, typically one day
    seed : int
        Random seed for reproducibility

    Returns
    -----------
    S_t : np.ndarray
        Array (length: n_steps + 1) of simulated prices
    '''

    np.random.seed(seed)
    dW = np.random.normal(scale=np.sqrt(dt), size=n_steps)
    W = dW.cumsum()
    
    t = np.linspace(dt, n_steps * dt, n_steps)
    
    S_t = S_0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    S_t = np.insert(S_t, 0, S_0)
    
    return S_t


# Now we will simulate price paths.
# Here is one simulated price path:

# In[8]:


simulate_gbm(
    S_0=train['Adj Close'].iloc[-1],
    mu=train['Return'].pipe(np.log1p).mean(),
    sigma=train['Return'].pipe(np.log1p).std(),
    n_steps=test.shape[0]
)


# We will combine `simulate_gbm()` with a list comprehension and `pd.concat()` to simulate many price paths.
# To simplify this combination, we will write a helper function `simulate_gbm_series()` that:
# 
# 1. Returns a series
# 1. Helps us vary the `seed` argument
# 

# In[9]:


def simulate_gbm_series(seed, train=train, test=test):
    S_t = simulate_gbm(
        S_0=train['Adj Close'].iloc[-1],
        mu=train['Return'].pipe(np.log1p).mean(),
        sigma=train['Return'].pipe(np.log1p).std(),
        n_steps=test.shape[0],
        seed=seed
    )
    return pd.Series(data=S_t, index=test.index.insert(0, train.index[-1]))


# In[10]:


n = 100

S_t = pd.concat(
    objs=[simulate_gbm_series(seed=seed) for seed in range(n)],
    axis=1,
    keys=range(n),
    names=['Simulation']
)


# In[11]:


S_t


# Below, we prefix the simulated price path column names with `_` to hide them from the legend.
# However, this feature triggers a warning *100 times*!
# We will suppress these 100 warnings with the warnings package.
# Generally, we should avoid suppressing warnings, however it is the easiest option here.

# In[12]:


import warnings


# In[13]:


fig, ax = plt.subplots(1,1)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    S_t.add_prefix('_').plot(alpha=0.1, ax=ax)
S_t.mean(axis=1).plot(label='Mean', ax=ax)
aapl.loc[S_t.index, ['Adj Close']].plot(label='Observed', ax=ax)
plt.legend()
plt.ylabel('Price ($)')
plt.title(
    'Apple Simulated and Observed Prices' + 
    f'\nTrained from {train.index[0]:%Y-%m-%d} to {train.index[-1]:%Y-%m-%d}'
)
plt.show()


# ### Pricing Options

# We can use simulated price paths to price options!
# We will use the Black and Scholes (1973) formula as a benchmark.
# Black and Scholes (1973) provide a closed form (analytic) solution to price European options.

# In[14]:


from scipy.stats import norm


# In[15]:


def price_bs(S_0, K, T, r, sigma, type='call'):
    '''
    Function used for calculating the price of European options using the analytical form of the Black-Scholes model.
    
    Parameters
    ------------
    S_0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to expiration in days
    r : float
        Daily risk-free rate
    sigma : float
        Standard deviation of daily stock returns
    type : str
        Type of the option. Allowable: ['call', 'put']
    
    Returns
    -----------
    option_premium : float
        The premium on the option calculated using the Black-Scholes model
    '''
    
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if type == 'call':
        val = (norm.cdf(d1, 0, 1) * S_0) - (norm.cdf(d2, 0, 1) * K * np.exp(-r * T))
    elif type == 'put':
        val = (norm.cdf(-d2, 0, 1) * K * np.exp(-r * T)) - (norm.cdf(-d1, 0, 1) * S_0)
    else:
        raise ValueError('Wrong input for type!')
        
    return val


# We can use the AAPL parameters above to price a European call option on AAPL stock.
# We will calculate its price at the end of 2021 with an expiration at the end of 2022, assuming a 5% risk-free rate.

# In[16]:


S_0 = train['Adj Close'].iloc[-1]
K = 100
T = test.shape[0]
r = 0.05/252
sigma = train['Return'].pipe(np.log1p).std()


# In[17]:


S_0


# In[18]:


_ = price_bs(
    S_0=S_0,
    K=K,
    T=T,
    r=r,
    sigma=sigma
)

print(f'Black and Scholes (1973) option price: {_:0.2f}')


# To simulate the Black and Scholes (1973) option price, we must simulate AAPL prices with the same 5% risk-free rate as the drift.
# We will write a new helper function to use the same inputs as above.

# In[19]:


def simulate_gbm_series(seed, S_0=S_0, T=T, r=r, sigma=sigma, train=train, test=test):
    S_t = simulate_gbm(
        S_0=S_0,
        mu=r,
        sigma=sigma,
        n_steps=T,
        seed=seed
    )
    return pd.Series(data=S_t, index=test.index.insert(0, train.index[-1]))


# We will simulate more price paths to increase the precision of out option price.

# In[20]:


n = 10_000

S_t = pd.concat(
    objs=[simulate_gbm_series(seed=seed) for seed in range(n)],
    axis=1,
    keys=range(n),
    names=['Simulation']
)


# In[21]:


S_t


# We can compare this price to a simulated price.
# The payoff on the call option is $S_T - K$ or $0$, whichever is higher.
# The price of the option is the present value of the mean payoff, discounted at the risk-free rate.

# In[22]:


payoff = np.maximum(S_t.iloc[-1] - K, 0)
_ = payoff.mean() * np.exp(-r * T)

print(f'Simulated option price: {_:0.2f}')


# The option prices do not match exactly.
# However, we can simulate more price paths to bring our simulated option price closer to the analytic solution.

# ## Estimating Value-at-Risk using Monte Carlo
# 
# Value-at-Risk (VaR) measures the risk associated with a portfolio
# VaR reports the worst expected loss, at a given level of confidence, over a certain horizon under normal market conditions. 
# For example, say the 1-day 95% VaR of our portfolio is \$100.
# This implies that that 95% of the time (under normal market conditions), we should not lose more than \\$100 over one day.
# We typically present VaR as a positive value, so a VaR of \$100 implies a loss of less than \$100.
# 
# We can calculate VaR several ways, including:
# 
# - Parametric Approach (Variance-Covariance)
# - Historical Simulation Approach
# - Monte Carlo simulations
# 
# We only consider the last method to calculate the 1-day VaR of an portfolio of 20 shares each of META and GOOG.

# In[23]:


tickers = ['GOOG', 'META']
shares = np.array([20, 20])
T = 1
n_sims = 10_000


# However, we will download all data from Yahoo! Finance and subset our data later.

# In[24]:


df = (
    yf.download(tickers=tickers, progress=False)
    .assign(
        Date=lambda x: x.index.tz_localize(None),
    )
    .set_index('Date')
    .rename_axis(columns=['Variable', 'Ticker'])
)


# Next, we calculate daily returns during 2022.
# Choosing the window to define "normal market conditions" is part art, part science, and beyod the scope of this lecture notebook.

# In[25]:


returns = df['Adj Close'].pct_change().loc['2022']


# In[26]:


returns


# We will need the variance-covariance matrix.

# In[27]:


cov_mat = returns.cov()

cov_mat * 1_000_000


# We will use the variance-covariance matrix to calculate the Cholesky decomposition.

# In[28]:


chol_mat = np.linalg.cholesky(cov_mat)

chol_mat


# The Cholesky decomposition helps us generate random variables with the same variance and covariance as the observed data.

# In[29]:


rv = np.random.normal(size=(n_sims, len(tickers)))


# In[30]:


correlated_rv = (chol_mat @ rv.T).T


# In[31]:


correlated_rv


# These random variables have a variance-covariance matrix similar to the real data.

# In[32]:


np.cov(correlated_rv.T)  * 1_000_000


# In[33]:


np.allclose(cov_mat, np.cov(correlated_rv.T), rtol=0.05)


# In[34]:


np.mean(correlated_rv, axis=0) * 100


# In[35]:


returns.mean().values * 100


# Here are the parameters for the simulated price paths:

# In[36]:


mu = returns.mean().values
sigma = returns.std().values
S_0 = df.loc['2021', 'Adj Close'].iloc[-1].values
P_0 = S_0.dot(shares)


# Calculate terminal prices using the GBM formula above:

# In[37]:


S_T = S_0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * correlated_rv)

S_T


# Calculate terminal portfolio values and returns.
# Note that these are dollar values, since VaR is typically expressed in dollar values.

# In[38]:


P_T = S_T.dot(shares)

P_T


# In[39]:


P_diff = P_T - P_0

P_diff


# In[40]:


P_diff.mean()


# Next, we calculate VaR.

# In[41]:


percentiles = [0.01, 0.05, 0.1]
var = np.percentile(P_diff, percentiles)

for x, y in zip(percentiles, var):
    print(f'1-day VaR with {100-x}% confidence: ${-y:.2f}')


# Finally, we will plot VaR:

# In[42]:


fig, ax = plt.subplots()
ax.hist(P_diff, bins=100, density=True)
ax.set_title(f'Distribution of 1-Day Changes in Portfolio Value\n from {n_sims} Simulations')
ax.axvline(x=var[2], color='red', ls='--')
ax.text(x=var[2], y=1, s='99% 1-Day VaR', color='red', ha='right', va='top', rotation=90, transform=ax.get_xaxis_transform())
ax.set_ylabel('Density')
ax.set_xlabel('1-Day Change in Portfolio Value ($)')
plt.show()

