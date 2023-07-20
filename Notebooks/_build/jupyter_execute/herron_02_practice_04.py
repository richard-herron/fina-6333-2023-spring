#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 2 - Practice (Monday 11:45 AM, Section 4)

# ## Announcements

# - I will finish grading projects this week/weekend
# - Quiz 5 due ~~Friday at 11:59 PM~~ Sunday at 11:59 PM
#     - A handful of students have submitted identical quizzes
#     - Quizzes are individual efforts
#     - Do not assume it is hard to for me to compare quiz and project submissions
# - DataCamp 20,000 XP due *next* Friday at 11:59 PM
# - Attendance and participation account for 5% of your grade

# ##  Practice

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# ### Implement the SMA(20) strategy with Bitcoin from the lecture notebook

# ~~Try to create the `btc` data frame in one code cell with one assignment (i.e., one `=`).~~
# ***Write a function `sma()` that accepts a data frame `df` and window size `n`.
# Use your `sma()` function to implement the SMA(20) strategy and assign it to data frame `btc_sma`.***

# In[3]:


import yfinance as yf


# In[4]:


btc = (
    yf.download(tickers='BTC-USD', progress=False)
    .assign(Date = lambda x: x.index.tz_localize(None))
    .set_index('Date')
    .rename_axis(columns='Variable')
)

btc.head()


# In[5]:


def sma(df, n=20):
    return (
        df
        .assign(
            Return = lambda x: x['Adj Close'].pct_change(),
            SMA = lambda x: x['Adj Close'].rolling(n).mean(),
            Position = lambda x: np.select(
                condlist=[
                    x['Adj Close'].shift() > x['SMA'].shift(), 
                    x['Adj Close'].shift() <= x['SMA'].shift()
                ],
                choicelist=[
                    1, 
                    0
                ],
                default=np.nan
            ),
            Strategy = lambda x: x['Position'] * x['Return']
        )
    )


# In[6]:


btc_sma = btc.pipe(sma, n=20)

btc_sma.tail()


# In[7]:


_ = btc_sma[['Return', 'Strategy']].dropna()

(
    _
    .add(1)
    .cumprod()
    .rename_axis(columns='Strategy')
    .rename(columns={'Return': 'Buy-And-Hold', 'Strategy': 'SMA(20)'})
    .plot()
)
plt.ylabel('Value ($)')
plt.title(f'Value of $1 Invested at Close on {_.index[0] - pd.offsets.Day(1):%B %d, %Y}')
plt.show()


# ### How does SMA(20) outperform buy-and-hold with this sample?

# Consider the following:
# 
# 1. Does SMA(20) avoid the worst performing days? How many of the worst 20 days does SMA(20) avoid? Try the `.sort_values()` or `.nlargest()` method.
# 1. Does SMA(20) preferentially avoid low-return days? Try to combine the `.groupby()` method and `pd.qcut()` function.
# 1. Does SMA(20) preferentially avoid high-volatility days? Try to combine the `.groupby()` method and `pd.qcut()` function.

# By chance, the SMA(20) strategy avoids all but three of the worst days.

# In[8]:


btc_sma.sort_values('Return')[['Position']].head(20).value_counts()


# However, SMA(20) does not avoid the best days, again by chance.

# In[9]:


btc_sma.sort_values('Return', ascending=False)[['Position']].head(20).value_counts()


# The SMA(20) strategy has a slight edge in picking high-return days, again by chance.

# In[10]:


(
    btc_sma
    .assign(q5_return = lambda x: 1 + pd.qcut(x['Return'], q=5, labels=False))
    .groupby('q5_return')
    ['Position']
    .mean()
    .plot(kind='bar')
)

plt.xticks(rotation=0)
plt.xlabel('Return Bin (1 is Lowest, 5 is Highest)')
plt.ylabel('Fraction of Days Strategy is Long Bitcoin')
plt.title('Mean Position by Return Bin')
plt.show()


# However, the SMA(20) *does* avoid the high volatility days that create [volatility drag](https://www.kitces.com/blog/volatility-drag-variance-drain-mean-arithmetic-vs-geometric-average-investment-returns/).

# In[11]:


(
    btc_sma
    .assign(
        Volatility = lambda x: x['Return'].rolling(63).std(),
        q5_volatility = lambda x: 1 + pd.qcut(x['Volatility'], q=5, labels=False)
    )
    .groupby('q5_volatility')
    ['Position']
    .mean()
    .plot(kind='bar')
)

plt.xticks(rotation=0)
plt.xlabel('63-Day Rolling Volatility Bin (1 is Lowest, 5 is Highest)')
plt.ylabel('Fraction of Days Strategy is Long Bitcoin')
plt.title('Mean Position by 63-Day Rolling Volatility Bin')
plt.show()


# Recall that $Arith\ Mean \approx Geom\ Mean + \frac{\sigma^2}{2}$, so avoiding high volatility (high variance) days, reduced the drag on the  cumulative returns that intermediate-term and long-term investors care about!

# In[12]:


(
    btc_sma
    .groupby('Position')
    ['Return']
    .agg(['std', 'mean', lambda x: (1 + x).prod()**(1 / x.count()) - 1])
    .mul(100)
    .rename(columns={'std': 'Volatility', 'mean': 'Arith Mean', '<lambda_0>': 'Geom Mean'})
)


# ### Implement the SMA(20) strategy with the market factor from French

# We need to impute a market price before we calculate SMA(20).

# In[13]:


import pandas_datareader as pdr
import requests_cache
session = requests_cache.CachedSession()


# In[14]:


ff = (
    pdr.DataReader(
        name='F-F_Research_Data_Factors_daily',
        data_source='famafrench',
        start='1900',
        session=session
    )
    [0]
    .div(100)
    .assign(
        Mkt = lambda x: x['Mkt-RF'] + x['RF'],
        Price = lambda x: x['Mkt'].add(1).cumprod()
    )
)


# In[15]:


ff_sma = (
    ff
    .rename(columns={'Price': 'Adj Close'})
    .pipe(sma, n=20)
)

ff_sma.tail()


# In[16]:


_ = ff_sma[['Return', 'Strategy']].dropna()

(
    _
    .add(1)
    .cumprod()
    .rename_axis(columns='Strategy')
    .rename(columns={'Return': 'Buy-And-Hold', 'Strategy': 'SMA(20)'})
    .plot()
)
plt.ylabel('Value ($)')
plt.title(f'Value of $1 Invested in Market at Close on {_.index[0] - pd.offsets.Day(1):%B %d, %Y}')
plt.show()


# ### How often does SMA(20) outperform buy-and-hold with 10-year rolling windows?

# In[17]:


(
    ff_sma
    [['Return', 'Strategy']]
    .rolling(10 * 252)
    .apply(lambda x: (1 + x).prod())
    .rename_axis(columns='Strategy')
    .rename(columns={'Return': 'Buy-And-Hold', 'Strategy': 'SMA(20)'})
    .plot()
)
plt.ylabel('Value ($)')
plt.title(f'Value of $1 Investments for Rolling 10-Year Holding Periods ')
plt.show()


# In the previous example, SMA(20) looks amazing!
# But over many shorter holding periods, we see the two are comparable.
# This is largely because the SMA(20) strategy *by pure chance* avoids big market draw downs!

# In[18]:


ff_sma.sort_values('Return')[['Position', 'Return', 'Strategy']].head(10)


# SMA(20) also avoids the up days.
# However, for this sample, missing the extreme down days helps more than missing the extreme updays hurts.

# In[19]:


ff_sma.sort_values('Return', ascending=False)[['Position', 'Return', 'Strategy']].head(10)


# We can also think about this problem by decade.
# If we want to get proper calendar decades (instead of 10-year periods that start in 1926), we combine `.groupby()` with an anonymous function that converts the date-time index to a proper calendar decade.
# Again, we see that SMA(20) and buy-and-hold trade wins, but SMA(20) wins bigs in the 1930s!

# In[20]:


(
    ff_sma
    [['Return', 'Strategy']]
    .groupby(lambda x: f'{(x.year // 10) * 10}s')
    .apply(lambda x: (1 + x).prod())
    .rename_axis(index='Decade', columns='Strategy')
    .rename(columns={'Return': 'Buy-And-Hold', 'Strategy': 'SMA(20)'})
    .plot(kind='bar')
)
plt.xticks(rotation=0)
plt.ylabel('Value ($)')
plt.title(f'Value of $1 Investments Add End of 10-Year Holding Periods ')
plt.show()


# In fact, buy-and-hold outperforms SMA(20) is we start in 1950.

# In[21]:


_ = ff_sma.loc['1950':, ['Return', 'Strategy']].dropna()

(
    _
    .add(1)
    .cumprod()
    .rename_axis(columns='Strategy')
    .rename(columns={'Return': 'Buy-And-Hold', 'Strategy': 'SMA(20)'})
    .plot()
)
plt.ylabel('Value ($)')
plt.title(f'Value of $1 Invested in Market at Close on {_.index[0] - pd.offsets.Day(1):%B %d, %Y}')
plt.show()


# ### Implement a long-only BB(20, 2) strategy with Bitcoin

# More on Bollinger Bands [here](https://www.bollingerbands.com/bollinger-bands) and [here](https://www.bollingerbands.com/bollinger-band-rules).
# In short, Bollinger Bands are bands around a trend, typically defined in terms of simple moving averages and volatilities.
# Here, long-only BB(20, 2) implies we have upper and lower bands at 2 standard deviations above and below SMA(20):
# 
# 1. Buy when the closing price crosses LB(20) from below, where LB(20) is SMA(20) minus 2 sigma
# 1. Sell when the closing price crosses UB(20) from above, where UB(20) is SMA(20) plus 2 sigma
# 1. No short-selling
# 
# The long-only BB(20, 2) is more difficult to implement than the long-only SMA(20) because we need to track buys and sells.
# For example, if the closing price is between LB(20) and BB(20), we need to know if our last trade was a buy or a sell.
# Further, if the closing price is below LB(20), we can still be long because we sell when the closing price crosses UB(20) from above.
# 
# ***Again, write a function `bb()` that accepts a data frame `df`, window size `n`, and number of standard deviations `m`.
# Use your `bb()` function to implement the BB(20, 2) strategy and assign it to data frame `btc_bb`.***

# In[22]:


def bb(df, n=20, m=2):
    return (
        df
        .assign(
            Return = lambda x: x['Adj Close'].pct_change(),
            SMA = lambda x: x['Adj Close'].rolling(n).mean(),
            SMV = lambda x: x['Adj Close'].rolling(n).std(),
            UB = lambda x: x['SMA'] + m*x['SMV'],
            LB = lambda x: x['SMA'] - m*x['SMV'],
            Position_with_nan = lambda x: np.select(
                condlist=[
                    (x['Adj Close'].shift(1) >= x['LB'].shift(1)) & (x['Adj Close'].shift(2) < x['LB'].shift(2)), 
                    (x['Adj Close'].shift(1) <= x['UB'].shift(1)) & (x['Adj Close'].shift(2) > x['UB'].shift(2)), 
                ],
                choicelist=[
                    1, 
                    0
                ],
                default=np.nan
            ),
            Position = lambda x: x['Position_with_nan'].fillna(method='ffill'),
            Strategy = lambda x: x['Position'] * x['Return']
        )
    )


# In[23]:


btc_bb = btc.pipe(bb)

btc_bb.tail()


# In[24]:


_ = btc_bb[['Return', 'Strategy']].dropna()

(
    _
    .add(1)
    .cumprod()
    .rename_axis(columns='Strategy')
    .rename(columns={'Return': 'Buy-And-Hold', 'Strategy': 'BB(20, 2)'})
    .plot()
)
plt.ylabel('Value ($)')
plt.title(f'Value of $1 Invested at Close on {_.index[0] - pd.offsets.Day(1):%B %d, %Y}')
plt.show()


# For an asset that we know has large positive returns over the sample, "time in the market" beats "timing the market".

# ### Implement a long-short RSI(14) strategy with Bitcoin

# From [Fidelity](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/rsi):
# 
# > The Relative Strength Index (RSI), developed by J. Welles Wilder, is a momentum oscillator that measures the speed and change of price movements. The RSI oscillates between zero and 100. Traditionally the RSI is considered overbought when above 70 and oversold when below 30. Signals can be generated by looking for divergences and failure swings. RSI can also be used to identify the general trend.
# 
# Here is the RSI formula: $RSI(n) = 100 - \frac{100}{1 + RS(n)}$, where $RS(n) = \frac{SMA(U, n)}{SMA(D, n)}$.
# For "up days", $U = \Delta Adj\ Close$ and $D = 0$, and, for "down days", $U = 0$ and $D = - \Delta Adj\ Close$.
# Therefore, $U$ and $D$ are always non-negative.
# We can learn more about RSI [here](https://en.wikipedia.org/wiki/Relative_strength_index).
# 
# We will implement a long-short RSI(14) as follows:
# 
# 1. Enter a long position when  the RSI crosses 30 from below, and exit the position when the RSI crosses 50 from below
# 1. Enter a short position when the RSI crosses 70 from above, and exit the position when the RSI crosses 50 from above
# 
# ***Again, write a function `rsi()` that accepts a data frame `df`, window size `n`, and boundary percentiles `lb`, `mb`, and `ub`.
# Use your `rsi()` function to implement the RSI strategy and assign it to data frame `btc_rsi`.***

# In[25]:


def rsi(df, n=14, lb=30, mb=50, ub=70):
    return df.assign(
        Return = lambda x: x['Adj Close'].pct_change(),
        Diff = lambda x: x['Adj Close'].diff(),
        U = lambda x: np.select(
            condlist=[x['Diff'] >= 0, x['Diff'] < 0],
            choicelist=[x['Diff'], 0],
            default=np.nan
        ),
        D = lambda x: np.select(
            condlist=[x['Diff'] <= 0, x['Diff'] > 0],
            choicelist=[-1 * x['Diff'], 0],
            default=np.nan
        ),
        SMAU = lambda x: x['U'].rolling(n).mean(),
        SMAD = lambda x: x['D'].rolling(n).mean(),
        RS = lambda x: x['SMAU'] / x['SMAD'],
        RSI = lambda x: 100 - 100 / (1 + x['RS']),
        Position_with_nan = lambda x: np.select(
            condlist=[
                (x['RSI'].shift(1) >= lb) & (x['RSI'].shift(2) < lb), 
                (x['RSI'].shift(1) >= mb) & (x['RSI'].shift(2) < mb),
                (x['RSI'].shift(1) <= ub) & (x['RSI'].shift(2) > ub), 
                (x['RSI'].shift(1) <= mb) & (x['RSI'].shift(2) > mb),
            ],
            choicelist=[
                1, 
                0,
                -1,
                0
            ],
            default=np.nan
        ),
        Position = lambda x: x['Position_with_nan'].fillna(method='ffill'),
        Strategy = lambda x: x['Position'] * x['Return']
    )


# In[26]:


btc_rsi = rsi(btc)

btc_rsi.tail()


# In[27]:


_ = btc_rsi[['Return', 'Strategy']].dropna()

(
    _
    .add(1)
    .cumprod()
    .rename_axis(columns='Strategy')
    .rename(columns={'Return': 'Buy-And-Hold', 'Strategy': 'RSI(14)'})
    .plot()
)
plt.ylabel('Value ($)')
plt.title(f'Value of $1 Invested at Close on {_.index[0] - pd.offsets.Day(1):%B %d, %Y}')
plt.show()


# We can compare all three!
# Shorting Bitcoin has been dangerous, as the poor returns on RSI(14) show!

# In[28]:


_ = (
    btc_sma[['Return', 'Strategy']]
    .join(
        btc_bb[['Strategy']].add_suffix('_BB'), 
    )
    .join(
        btc_rsi[['Strategy']].add_suffix('_RSI'), 
    )
    .dropna()
)


(
    _
    .add(1)
    .cumprod()
    .rename_axis(columns='Strategy')
    .rename(columns=
            {
                'Return': 'Buy-And-Hold', 
                'Strategy': 'SMA(20)',
                'Strategy_BB': 'BB(20, 2)',
                'Strategy_RSI': 'RSI(14)',
            }
           )
    .plot()
)
plt.semilogy()
plt.ylabel('Value ($)')
plt.title(f'Value of $1 Invested at Close on {_.index[0] - pd.offsets.Day(1):%B %d, %Y}')
plt.show()

