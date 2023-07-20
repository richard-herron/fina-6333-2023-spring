#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 5 - Practice (Wednesday 11:45 AM, Section 4)

# ## Announcements

# - Due dates
#     - Team project 2 is due by 11:59 PM on Wednesday, 4/26
#     - Teammates review 2 is due by 11:59 PM on Wednesday, 4/26
#     - 30,000 DataCamp XP are due by 11:59 PM on *Friday, 4/28*
#     - TRACE course evaluations are due by 11:59 PM on *Friday, 4/28*
# - ***TRACE course evaluations are optional (and anonymous), but I value your feedback!***
#     - I always find student feedback useful, and I read and re-read it!
#     - I tweak my courses every semester based on student feedback!
#     - Your feedback does not have to be perfect to be useful!
#     - If you had a magic wand, what would you change about my course and approach?
# - This week and next, we can discuss whatever topics you want, including simulations and team project 2

# ##  Practice

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


# ### Estimate $\pi$ by simulating darts thrown at a dart board

# *Hints:*
# Select random $x$s and $y$s such that $-r \leq x \leq +r$ and $-r \leq x \leq +r$.
# Darts are on the board if $x^2 + y^2 \leq r^2$.
# The area of the circlular board is $\pi r^2$, and the area of square around the board is $(2r)^2 = 4r^2$.
# The fraction $f$ of darts on the board is the same as the ratio of circle area to square area, so $f = \frac{\pi r^2}{4 r^2}$.

# First we throw darts at the board.
# Darts with $x^2 + y^2 \leq r^2$ are on the board.

# In[4]:


def throw_darts(n=10_000, r=1, seed=42):
    np.random.seed(seed)
    return (
        pd.DataFrame(
            data=np.random.uniform(low=-r, high=r, size=2*n).reshape(n, 2), 
            columns=['x', 'y']
        )
        .assign(board=lambda x: x['x']**2 + x['y']**2 <= r**2)
        .rename_axis(index='n', columns='Variable')
    )


# In[5]:


throw_darts()


# Next, we visualize these darts with a scatter plot.
# Seaborn's `scatterplot()` helps color darts by location (i.e., on or off board).
# The `.pipe()` method lets us send the output of the `.assign()` method to `sns.scatterplot()` without assigning a temporary data frame.

# In[6]:


import seaborn as sns


# In[7]:


(
    throw_darts()
    .assign(Location=lambda x: np.where(x['board'], 'On Board', 'Off Board'))
    .pipe(lambda x: sns.scatterplot(x='x', y='y', data=x, hue='Location'))
)
plt.title('Simulated Dart Throws')
plt.show()


# Finally, we use the hint above to estimate $\pi$.
# The hint above says $f = \frac{\pi r^2}{4 r^2}$, where $f$ is the fraction of darts on the board.
# Therefore, $\pi = \frac{4fr^2}{r^2} = 4f$.

# In[8]:


n = 10_000
pi = 4 * throw_darts(n=n)["board"].mean()
print(f'Estimate of pi based on {n:,.0f} darts: {pi:0.4f}')


# We increase the precision of our $\pi$ estimate by increasing the number of simulated darts $n$.

# In[9]:


for n in 10**np.arange(7):
    pi = 4 * throw_darts(n=n)["board"].mean()
    print(f'Estimate of pi based on {n:<9,.0f} darts: {pi:0.4f}')


# ### Simulate your wealth $W_T$ by randomly sampling market returns

# Use monthly market returns from the French Data Library.
# Only invest one cash flow $W_0$, and plot the distribution of $W_T$.

# First, we download data from the French Data Library.
# We convert these returns from percent to decimal to simplify compounding.

# In[10]:


mkt = (
    pdr.DataReader(
        name='F-F_Research_Data_Factors_daily',
        data_source='famafrench',
        start='1900',
        session=session
    )[0]
    .assign(mkt=lambda x: (x['Mkt-RF'] + x['RF']) / 100)
    ['mkt']
)


# Next, we write a couple of helper functions.
# The `get_sample()` function draws one random sample of `n` observations from returns series `x`.
# This sample provides one simulated return history.

# In[11]:


def get_sample(x, n=10_000, seed=42, start=None):
    if start is None:
        start = x.index[-1] + pd.offsets.BDay()
    return (
        x
        .sample(n=n, replace=True, random_state=seed, ignore_index=True)
        .set_axis(pd.date_range(start=start, periods=n))
        .rename_axis(index='Date')
    )    


# The `get_samples()` function calls the `get_sample()` function `m` times to simulate `m` return histories.
# The `get_samples()` function combines these `m` return histories into one data frame.

# In[12]:


def get_samples(x, m=100, n=10_000, seed=42, start=None):
    return (
        pd.concat(
            objs=[get_sample(x=x, n=n, seed=seed+i, start=start) for i in range(m)],
            axis=1,
            keys=range(m),
            names='Sample'
        )
    )


# Next, we use these helper functions to simulate 10,000 return histories of 10,000 trading days each.

# In[13]:


mkts = get_samples(x=mkt, m=10_000, n=10_000)

mkts


# We compound daily returns $R_t$ to find future wealth $W_T$, so $W_T = W_0 \times (1 + R_1) \times (1 + R_2) \times \cdots \times (1 + R_T)$.
# We use the `.cumprod()` method to compound the daily returns in data frame `mkts`.

# In[14]:


W_0 = 1_000_000
W_t = W_0 * mkts.add(1).cumprod()


# We visualize terminal wealth $W_T$ with a cumulative distribution plot.

# In[15]:


W_t.iloc[-1].plot(kind='hist', density=True, cumulative=True, bins=1_000)
plt.semilogx()
plt.xlabel(r'$W_T$ where $W_0 = $' + f'\${W_0:,.0f}')
plt.title(
    r'Cumulative Distribution of $W_T$' + 
    f'\nafter {W_t.shape[0]:,.0f} Trading Days'
)
plt.show()


# The plot above shows the cumulative distribution of $W_T$, suggesting we expect $W_T$ greater than about $17 million for about 75% of samples.

# The plot above may be difficult to read and interpret because $W_T$ has large outliers.
# The `.describe()` method provides few salient values of $W_T$.
# We convert $W_T$ from dollars to millions of dollars with `.div(1_000_000)`.

# In[16]:


W_t.iloc[-1].div(1_000_000).describe()


# ### Repeat the exercise above but add end-of-month investments $C_t$

# We can use the same data frame `mkts` of simulated market returns.
# However, need to consider end-of-month investments of $C_t$.
# The easiest approach is to aggregate the daily market returns in `mkts` to monthly market returns in data frame `mkts_m`.
# The last month in `mkts_m` is not complete (i.e., fewer than about 21 trading days), so we drop is with `.iloc[:-1]`.

# In[17]:


mkts_m = (
    mkts
    .add(1)
    .resample(rule='M', kind='period')
    .prod()
    .sub(1)
    .iloc[:-1]
)


# The wealth at time $t$ is $W_t$ and depends on:
# 
# 1. The wealth at the end of the previous month $W_{t-1}$
# 1. The return over the previous month $R_t$ (recall we label returns by their right edge)
# 1. The end-of-month investment $C_t$
# 
# Putting it all togther: $W_t = W_{t-1} \times (1 + R_t) + C_t$
# 
# We have to loop over the monthly returns in `mkts_m` because we have to combine compounded returns and cash flows.
# The `iterrows()` methods provides an easy way to iterate (loop) over the rows in `mkts_m`.

# In[18]:


C_t = 1_000
W_0 = 1_000_000
W_last = W_0
W_t = []

for d, m in mkts_m.iterrows():
    W_last = (1 + m) * W_last + C_t
    W_t.append(W_last)

W_t = pd.concat(objs=W_t, axis=1, keys=mkts_m.index).transpose()


# We repeat the cumulative distribution of wealth plot and descriptive statistics from above.

# In[19]:


W_t.iloc[-1].plot(kind='hist', density=True, cumulative=True, bins=1_000)
plt.semilogx()
plt.xlabel(
    r'$W_T$ where $W_0 = $' + f'\${W_0:,.0f}' + 
    r' and  $C_t = $' + f'\${C_t:,.0f}'
)
plt.title(
    r'Cumulative Distribution of $W_T$' + 
    f'\nafter {W_t.shape[0]:,.0f} Trading Months'
)
plt.show()


# In[20]:


W_t.iloc[-1].div(1_000_000).describe()


# The plot above shows the cumulative distribution of $W_T$, suggesting we expect $W_T$ greater than about \\$20 million for about 75% of samples.
# Note, even though we deposit $1,000 per month for almost 40 years, we do not gain much wealth over the previous example with only a lump sum!
# Start saving early!
