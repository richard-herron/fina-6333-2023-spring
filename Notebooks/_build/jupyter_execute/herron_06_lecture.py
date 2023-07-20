#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 6 - Size, Value, and Momentum Investing

# This lecture notebook covers momentum investing, and the practice notebook will apply the tools we learn in this notebook to size and value investing.
# Here, we will learn:
# 
# 1. What is momentum investing?
# 1. How to use Center for Research in Security Prices (CRSP) data, which is survivorship bias free
# 1. How to implement and evaluate a momentum strategy

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# In[3]:


import yfinance as yf
import pandas_datareader as pdr
import requests_cache
session = requests_cache.CachedSession()


# ## What is momentum investing?

# From page 13 of [chapter 12](https://book.ivo-welch.info/read/source5.mba/12-effbehav.pdf) of Ivo Welch's free corporate finance textbook (***emphasis added***):
# 
# > The second-most important factor anomaly was the momentum investing strategy.
# ***Momentum investing strategies require going long in stocks that have increased
# greatly over the last year, and going short in stocks that have decreased greatly.***
# (It requires a few more contortions, but this is a reasonable characterization.) As
# with value, behavioral finance researchers were quick to adopt momentum as a
# consequence of investor psychology. They also developed plenty of theories that
# argued about how the psychology of investors could explain momentum.
# >
# > Yet over the last 17 years, Ken French’s data suggests that the average rate of Momentum has been mostly
# return on the momentum investment strategy was — drumroll — 0.03% with a
# standard deviation of 23.8%. This rate of return is statistically and economically
# insignificant. Momentum investing also had the unpleasant aspect of sudden nasty
# risk. It lost 83 cents for every dollar invested in 2009

# ## CRSP Data

# We typically use Yahoo! Finance data in class because these data are easy to download and use.
# However, Yahoo! Finance only provides data for listed (active) public companies.
# When public companies delist, Yahoo! Finance removes their data from its website and application programming interfaces (APIs).
# Companies delist for various reasons, including failures, poor performance, buyouts, and acquisitions.
# Failures and poor performance are generally associated with large negative returns before delisting.
# Buyouts and acquisitions are generally associated with large positive returns before delisting.
# Regardless of the reason for delisting, delisted company data are unavailable from Yahoo! Finance.
# Because delistings are not randomly assigned and typically related to past performance, we cannot ignore them.
# If we ignore delistings, we create a [survivorship bias](https://en.wikipedia.org/wiki/Survivorship_bias):
# 
# 
# > Survivorship bias, survival bias or immortal time bias is the logical error of concentrating on the people or things that made it past some selection process and overlooking those that did not, typically because of their lack of visibility. This can lead to incorrect conclusions regarding that which (or those who) didn't make it.
# >
# > Survivorship bias is a form of selection bias that can lead to overly optimistic beliefs because multiple failures are overlooked, such as when companies that no longer exist are excluded from analyses of financial performance. It can also lead to the false belief that the successes in a group have some special property, rather than just coincidence as in correlation "proves" causality. For example, if 3 of the 5 students with their state's highest college GPAs went to the same high school, it might lead to the notion (which the institution may even capitalize on through advertising) that their high school offers an excellent education even though it's actually due to their school being the largest in their state. Therefore, by comparing the average GPA of all of the school's students — not just the ones who made the top-five selection process — to state averages, one could better assess the school's quality (not quantity).
# > 
# > Another kind of survivorship bias would involve thinking that an incident was not all that dangerous because the only people who were involved in the incident who can speak about it are those who survived it. Even if one knew that some people are dead, they would not have their voice to add to the conversation, leading to bias in the conversation. 

# We should always be on the lookout for survivorship bias!
# Here is my favorite survivorship bias joke: 
# 
# 
# ![XKCD 1827](https://imgs.xkcd.com/comics/survivorship_bias.png)

# To avoid a survivorship bias, we will use survivorship-bias-free data from the [Center for Research in Security Prices (CRSP)](https://www.crsp.org/).
# CRSP data include delisted stocks and are used by academics and institutional investors to research and backtest trading strategies.
# Download the CRSP data file `crsp.csv` from Canvas, and put it in the same folder as this notebook.
# Then, we can read and clean the CRSP data file as follows.

# In[4]:


crsp = (
    pd.read_csv(
        filepath_or_buffer='crsp.csv',
        parse_dates=['date'],
        na_values=['A', 'B', 'C'] # CRSP uses letter codes to provide additional information, which we can ignore
    )
    .assign(date=lambda x: x['date'].dt.to_period(freq='M')) # returns span a month, so I prefer to work with periods instead of dates
    .rename_axis(columns='Variable')
    .set_index(['PERMNO', 'date'])
)


# The CRSP data files contains a small subset of all available CRSP data:
# 
# 1. `PERMNO` indicates permanent security identifiers, which is more reliable than tickers, which may change or be re-used by a different firm
# 1. `date` indicates the last trading day of the month, which we convert to a year-month "period"
# 1. `SHRCD` indicates share classes (e.g., A, B, and preferred), which I filtered to values of 10 or 11
# 1. `PRC` indicates the closing price on the last trading day of the month, and `PRC` is negative if it is the mean of the bid and ask prices instead of a price from an observed trade
# 1. `RET` indicates the holding period return as a simple return, including dividends
# 1. `SHROUT` indicates the number of shares outstanding in thousands (e.g., a `SHROUT` value of 1,000 indicates 1,000,000 shares)

# In[5]:


crsp.head()


# These CRSP data are de-duplicated, but we should double-check for duplicate PERMNO-date pairs with the `.duplicated()` method.

# In[6]:


assert not crsp.index.duplicated().any()


# For fun, we can use these data to count the number of listed stocks each month and plot the trend of publicly listed stocks.

# In[7]:


crsp.reset_index('PERMNO').resample('M')['PRC'].count().plot()
plt.xlabel('Date')
plt.ylabel('Number of Publicly Listed Stocks')
plt.title('Time Series of the Number of Publicly Listed Stocks')
plt.show()


# ## Implement a Momentum Investing Strategy

# We will implement a momentum investing strategy with 1-month holding periods of equal-weighted portfolios formed on total returns from month -12 through month -2.
# For example, at the start of January in year $t$, we assign stocks to 10 portfolios based on their total returns from January in year $t-1$ through November in year $t-1$.
# Portfolio 1 has the lowest trailing returns, and portfolio 10 has the highest trailing returns.
# We do not use the returns in month -1 for portfolio assignment because they would contaminate our portfolio returns with [bid-ask bounce](https://www.investopedia.com/ask/answers/013015/whats-difference-between-bidask-spread-and-bidask-bounce.asp).
# For example, in our example above, we do not use returns from December in year $t-1$ to assign stocks to portfolios at the start of January in year $t$.

# ### Calculate 1-Month and 11-Month Returns

# We will assign 1-month returns in month 0 to data frame `ret_1m`.
# This calculation is easier in the wide format, but we will later use the long format.

# In[8]:


ret_1m = crsp['RET'].unstack('PERMNO')

ret_1m.tail()


# We will assign 11-month returns from month -12 to month -2 to data frame `ret_11m`.
# The `.rolling()` method does not have a `.prod()` method, so we will sum log returns instead of compounding simple returns, which speeds up our calculation.

# In[9]:


ret_11m = ret_1m.pipe(np.log1p).rolling(11).sum().pipe(np.expm1)

ret_11m.tail()


# We should always check our work!
# If we were to use this code in a production environment (i.e., run it frequently to make decisions), we would add [unit tests](https://en.wikipedia.org/wiki/Unit_testing) to check our work.
# We can use `PERMNO` 10010 to (very lightly) check our work.

# In[10]:


assert np.allclose(
    a=ret_1m[10010].rolling(11).apply(lambda x: (1 + x).prod() - 1),
    b=ret_11m[10010],
    equal_nan=True
)


# ### Assign Stocks to Portfolios Based on 11-Month Returns

# We will use `pd.qcut()` to assign stocks to portfolios based on their 11-month trailing returns.
# Here is a simple example that uses `pd.qcut()` to assign the values 0 to 24 to 10 portfolios.

# In[11]:


pd.qcut(np.arange(25), q=10, labels=False) + 1


# We will save these portfolio assignments to data frame `port_11m`.
# The `pd.qcut()` function errs if we try to cut an array of all missing values, so we will use `.dropna(how='all')` to drop rows with all missing values.

# In[12]:


port_11m = ret_11m.dropna(how='all').apply(pd.qcut, q=10, labels=False, axis=1) + 1

port_11m.tail()


# Again. we should check our output, here with the last row of data.

# In[13]:


assert np.allclose(
    a=port_11m.iloc[-1],
    b=pd.qcut(x=ret_11m.iloc[-1], q=10, labels=False) + 1,
    equal_nan=True
)


# ### Combine Returns and Portfolio Assignments

# Next, we will use `pd.concat(axis=1)` to match returns and portfolio assignments.
# We must `.shift(2)` the 11-month returns and portfolios assignments to avoid a look-ahead bias and drop month -1 returns.
# 
# 1. The first shift makes sure we do not use contemporaneous returns to assign stocks to portfolios (i.e., otherwise the portfolio ranking and portfolio returns would overlap).
# 1. The second shift avoids mechanical correlations between returns one month and the next (i.e., bid-ask bounce and market microstructure noise).

# In[14]:


mom_0 = (
    pd.concat(
        objs=[ret_1m, ret_11m.shift(2), port_11m.shift(2)],
        axis=1, 
        keys=['Return', 'Return_Trailing', 'Portfolio'],
        names=['Variable', 'PERMNO']
    )
    .stack('PERMNO')
    .dropna()
    .assign(Portfolio=lambda x: x['Portfolio'].astype(int))
)


# In[15]:


mom_0.head()


# ### Evaluate Performance

# Is there a relation between returns from month -12 to month -2 and return in month 0?
# We can use the `.corr()` to estimate this.

# In[16]:


mom_0.filter(regex='Return').corr()


# This correlation is very weak!
# But we expect we correlations based on pervious course topics.
# We can try again with log returns, which may reduce the noise from outliers.
# Recall $R_{Log} = log(1 + R_{Simple})$, which we can quickly implement with `.pipe(np.log1p)`.

# In[17]:


mom_0.filter(regex='Return').pipe(np.log1p).corr()


# The correlation of log returns is larger, but still low because single-stock returns are noisy!
# We can reduce this noise with portfolios formed on trailing returns.
# We will equally weight the returns in each portfolio using `.mean()`.

# In[18]:


mom_ew = (
    mom_0
    .groupby(by=['date', 'Portfolio'])
    ['Return']
    .mean()
    .unstack('Portfolio')
)


# In[19]:


mom_ew.head()


# Next, we will plot the mean return for each portfolio.
# We will convert these equal-weighted portfolio returns to percent, but leave them as monthly values.

# In[20]:


mom_ew.mean().mul(100).plot(kind='bar')
plt.ylabel('Mean Monthly Return (%)')
plt.xlabel('Momentum Portfolio')
plt.title(
    'Performance of Momentum Investing' +
    '\nMean Monthly Returns on Equal-Weighted Portfolios' +
    '\nFormed on Months -12 to Month -2 Returns'
)
plt.show()


# Above, we see about a 75 basis point spread between the returns on portfolios 1 and 10, suggesting that momemtum investing generates excess returns.
# We should also consider cumulative returns over holding periods longer than one month.
# Next, we will plot the values of $1 invested in each portfolio.

# In[21]:


mom_ew.add(1).cumprod().plot()
plt.semilogy()
plt.ylabel('Value of \$1 Invested (\$)')
plt.xlabel('Date')
plt.title(
    'Performance of Momentum Investing' +
    '\nValue of $1 Invested in Equal-Weighted Portfolios' +
    '\nFormed on Months -12 to Month -2 Returns'
)
plt.show()


# Finally, we can estimate capital asset pricing model (CAPM) and Fama-French four-factor model (FF4) regressions.

# In[22]:


ff_0 = pdr.DataReader(
    name='F-F_Research_Data_Factors',
    data_source='famafrench',
    session=session,
    start='1900'
)

ff_0[0].head()


# In[23]:


ff_mom = pdr.DataReader(name='F-F_Momentum_Factor', data_source='famafrench', session=session, start='1900')
ff_mom[0].columns = [c.strip() for c in ff_mom[0].columns]

ff_mom[0].head()


# In[24]:


import statsmodels.formula.api as smf


# In[25]:


def capm(c, df):
    return smf.ols(formula=f'I(Q({c}) - RF) ~ Q("Mkt-RF")', data=df)


# In[26]:


_ = mom_ew.mul(100).join(ff_0[0])
models = [capm(c=c, df=_) for c in mom_ew.columns]
fits = [m.fit() for m in models]
params = pd.concat([f.params for f in fits], axis=1, keys=mom_ew.columns).T
bses = pd.concat([f.bse for f in fits], axis=1, keys=mom_ew.columns).T

plt.bar(
    x=params.index,
    height=params['Intercept'],
    yerr=bses['Intercept']
)

plt.ylabel('Mean Equal-Weighted Monthly Alpha (%)')
plt.xlabel('Momentum Portfolio')
plt.title(
    'Performance of Momentum Investing' +
    '\nCAPM Alphas for Equal-Weighted Portfolios' +
    '\nFormed on Months -12 to Month -2 Returns'
)
plt.show()


# In[27]:


def ff4(c, df):
    return smf.ols(formula=f'I(Q({c}) - RF) ~ Q("Mkt-RF") + SMB + HML + Mom', data=df)


# In[28]:


_ = mom_ew.mul(100).join([ff_0[0], ff_mom[0]])
models = [ff4(c=c, df=_) for c in mom_ew.columns]
fits = [m.fit() for m in models]
params = pd.concat([f.params for f in fits], axis=1, keys=mom_ew.columns).T
bses = pd.concat([f.bse for f in fits], axis=1, keys=mom_ew.columns).T

plt.bar(
    x=params.index,
    height=params['Intercept'],
    yerr=bses['Intercept']
)

plt.ylabel('Mean Equal-Weighted Monthly Alpha (%)')
plt.xlabel('Momentum Portfolio')
plt.title(
    'Performance of Momentum Investing' +
    '\nFF4 Alphas for Equal-Weighted Portfolios' +
    '\nFormed on Months -12 to Month -2 Returns'
)
plt.show()


# In the FF4 model above, the alphas are abnormally large because we use equal-weighted portfolios, which overweight small stocks.
# Because these portfolios overweight small stocks, these alphas make be associated with liquidity and difficult to earn at scale.
# In the practice notebook, we will explore value-weighted portfolios and size investing strategies.
