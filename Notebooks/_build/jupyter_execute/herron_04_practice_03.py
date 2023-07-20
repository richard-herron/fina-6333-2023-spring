#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 4 - Practice (Monday 2:45 PM, Section 3)

# ## Announcements

# - Quiz 6 this week
#     - I will post it at about 6 PM on Wednesday, 3/31
#     - It will be due by 11:59 PM on Friday, 3/31
# - Please complete the week ten survey
#     - I am considering dropping a topic to allow more in-class group work and easier access to me
#     - I am also curious why the quantitative courses are less popular this summer
#     - Please complete by 11:59 PM on Friday, 3/31
#     - ***The week ten survey is anonymous and voluntary***
# - I will post project 2 as soon as I can
# - Assessment exam
#     - 20 questions multiple on Canvas
#     - ***You must be in the class room***
#     - No specific studying, but I suggest putting core course resources on your laptop (e.g., notes and PowerPoints)

# ##  Practice

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


# In[4]:


import scipy.optimize as sco


# ### Find the maximum Sharpe Ratio portfolio of MATANA stocks over the last three years

# ***Note that `sco.minimize()` finds minimums, so we need to minimize the negative Sharpe Ratio.***
# 
# The following code downloads data for the MATANA stocks and assigns daily decimal returns from 2020 through 2022 to data frame `returns`.
# We will stop in 2022 to make it easier to compare our results, whether we use the risk-free rate or value-weighted market portfolio as our benchmark or not.
# Recall, the Fama and French benchmark factors are only available with a lag, and are only available through December 2022 as I type.

# In[5]:


tickers = 'MSFT AAPL TSLA AMZN NVDA GOOG'

matana = (
    yf.download(tickers=tickers, progress=False)
    .assign(Date=lambda x: x.index.tz_localize(None))
    .set_index('Date')
    .rename_axis(columns=['Variable', 'Ticker'])
)

returns = matana['Adj Close'].pct_change().loc['2020':'2022']
returns.describe()


# In[6]:


def port_sharpe(x, r, ppy, tgt):
    """
    x: portfolio weights
    r: data frame of returns
    ppy: periods per year for annualization
    tgt: target or benchmark
    """
    rp = r.dot(x) # portfolio return
    er = rp.sub(tgt).dropna() # portfolio excess return
    return np.sqrt(ppy) * er.mean() / er.std() # portfolio Sharpe Ratio


# In[7]:


def port_sharpe_neg(x, r, ppy, tgt):
    return -1 * port_sharpe(x, r, ppy, tgt)


# In[8]:


res_sharpe_1 = sco.minimize(
    fun=port_sharpe_neg,
    x0=np.ones(returns.shape[1]) / returns.shape[1],
    args=(returns, 252, 0),
    bounds=[(0,1) for _ in range(returns.shape[1])],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1} # want eq constraint to = 0
    )
)

res_sharpe_1


# In[9]:


port_sharpe(x=res_sharpe_1['x'], r=returns, ppy=252, tgt=0)


# ### Find the maximum Sharpe Ratio portfolio of MATANA stocks over the last three years, but allow short weights up to 10% on each stock

# In[10]:


res_sharpe_2 = sco.minimize(
    fun=port_sharpe_neg,
    x0=np.ones(returns.shape[1]) / returns.shape[1],
    args=(returns, 252, 0),
    bounds=[(-.1,1.5) for _ in range(returns.shape[1])],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1} # want eq constraint to = 0
    )
)

res_sharpe_2


# In[11]:


(
    pd.DataFrame(
        data={
            'Long Only':res_sharpe_1['x'], 
            'Up to 10% Short per Stock':res_sharpe_2['x']
        },
        index=returns.columns
    )
    .rename_axis('Portfolio Weight')
    .plot(kind='barh')
)
plt.title('Comparison Max. Sharpe Ratio Portfolio Weights')
plt.show()


# By relaxing the long-only constrain (via changes to `bounds=`), the weights on AMZN, GOOG, and MSFT go from zero to -10%.
# Also, the Sharpe Ratio increases because we relax a binding constraint.

# In[12]:


port_sharpe(res_sharpe_1['x'], r=returns, ppy=252, tgt=0)


# In[13]:


port_sharpe(res_sharpe_2['x'], r=returns, ppy=252, tgt=0)


# ### Find the maximum Sharpe Ratio portfolio of MATANA stocks over the last three years, but allow total short weights of up to 30%

# We can find the negative values in a NumPy array as follows.

# In[14]:


x = np.arange(10) - 5
x_neg = x[x < 0]
print(f'All Values: {x}\nNegative Values: {x_neg}')


# In[15]:


res_sharpe_3 = sco.minimize(
    fun=port_sharpe_neg,
    x0=np.ones(returns.shape[1]) / returns.shape[1],
    args=(returns, 252, 0),
    bounds=[(-0.3,1.3) for _ in range(returns.shape[1])],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1}, # want eq constraint to = 0
        {'type': 'ineq', 'fun': lambda x: x[x < 0].sum() + 0.3} # want ineq constraint to >= 0
    )
)

res_sharpe_3


# In[16]:


(
    pd.DataFrame(
        data={
            'Long Only':res_sharpe_1['x'], 
            'Up to 30% Short Total':res_sharpe_2['x']
        },
        index=returns.columns
    )
    .rename_axis('Portfolio Weight')
    .plot(kind='barh')
)
plt.title('Comparison Max. Sharpe Ratio Portfolios')
plt.show()


# Again, by relaxing the long-only constrain, the weights on AMZN, GOOG, and MSFT go from zero to -10%.
# Also, the Sharpe Ratio increases because we relax a binding constraints.
# The Sharpe Ratio is higher here than in the previous exercise, but this will not always be the case, since we relax different constraints here and in the previous exercise.

# In[17]:


port_sharpe(res_sharpe_1['x'], r=returns, ppy=252, tgt=0)


# In[18]:


port_sharpe(res_sharpe_3['x'], r=returns, ppy=252, tgt=0)


# ### Find the maximum Sharpe Ratio portfolio of MATANA stocks over the last three years, but do not allow any weight to exceed 30% in magnitude

# We can do this easily with `bounds=`.

# In[19]:


res_sharpe_4 = sco.minimize(
    fun=port_sharpe_neg,
    x0=np.ones(returns.shape[1]) / returns.shape[1],
    args=(returns, 252, 0),
    tol=1e-6,
    bounds=[(0,0.3) for _ in range(returns.shape[1])],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1}, # want eq constraint to = 0
    )
)

res_sharpe_4


# ***I removed the version that achieved the same result with constraints, because it did not work the same on different computers.***
# I cannot find documentation for why this solution fails, but I suspect using the `.max()` method in a constraint makes convergence slower because changes to non-max values in `x` do not change the constraint function output.

# ### Find the minimum 95% Value at Risk (Var) portfolio of MATANA stocks over the last three years

# More on VaR [here](https://en.wikipedia.org/wiki/Value_at_risk).

# In[20]:


def port_var(x, r, q):
    return r.dot(x).quantile(q)


# In[21]:


def port_var_neg(x, r, q):
    return -1 * port_var(x=x, r=r, q=q)


# In[22]:


res_var_1 = sco.minimize(
    fun=port_var_neg,
    x0=np.ones(returns.shape[1]) / returns.shape[1],
    args=(returns, 0.05),
    bounds=[(0,1) for _ in returns],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1}, # minimize drives "eq" constraints to zero
    )
)

res_var_1


# In[23]:


port_var(x=res_var_1['x'], r=returns, q=0.05)


# It might be helpful to slightly change then minimum VaR portfolio weights to show that we minimized VaR.

# In[24]:


def tweak(x, d=0.05):
    y = np.zeros(x.shape[0])
    y[0], y[1] = d, -1 * d
    return x + y


# In[25]:


port_var(x=tweak(res_var_1['x']), r=returns, q=0.05)


# ### Find the minimum maximum draw down portfolio of MATANA stocks over the last three years

# In[26]:


def port_draw_down_max(x, r):
    rp = r.dot(x)
    price = rp.add(1).cumprod()
    cum_max = price.cummax()
    draw_down = (cum_max - price) / cum_max
    return draw_down.max()


# In[27]:


res_dd_1 = sco.minimize(
    fun=port_draw_down_max,
    x0=np.ones(returns.shape[1]) / returns.shape[1],
    args=(returns,),
    bounds=[(0,1) for _ in returns],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1}, # minimize drives "eq" constraints to zero
    )
)

res_dd_1


# In[28]:


port_draw_down_max(x=res_dd_1['x'], r=returns)


# Again. it might be helpful to slightly change then minimum VaR portfolio weights to show that we minimized VaR.

# In[29]:


port_draw_down_max(x=tweak(res_dd_1['x']), r=returns)


# ### Find the minimum maximum draw down portfolio with all available data for the current Dow-Jones Industrial Average (DJIA) stocks

# You can find the [DJIA tickers on Wikipedia](https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average).

# In[30]:


wiki = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')


# In[31]:


djia = (
    yf.download(tickers=wiki[1]['Symbol'].to_list(), progress=False)
    .assign(Date=lambda x: x.index.tz_localize(None))
    .set_index('Date')
    .rename_axis(columns=['Variable', 'Ticker'])
)

returns_2 = djia['Adj Close'].pct_change().loc[:'2022'].dropna()
returns_2.describe()


# In[32]:


res_dd_2 = sco.minimize(
    fun=port_draw_down_max,
    x0=np.ones(returns_2.shape[1]) / returns_2.shape[1],
    args=(returns_2,),
    bounds=[(0,1) for _ in returns_2],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1}, # minimize drives "eq" constraints to zero
    )
)

res_dd_2


# In[33]:


port_draw_down_max(x=res_dd_2['x'], r=returns_2)


# Again. it might be helpful to slightly change then minimum VaR portfolio weights to show that we minimized VaR.

# In[34]:


port_draw_down_max(x=tweak(res_dd_2['x']), r=returns_2)


# ### Plot the (mean-variance) efficient frontier with all available data for the current the DJIA stocks

# The range of target returns in `tret` span from the minimum to the maximum mean single-stock returns.

# In[35]:


_ = returns_2.mean().mul(252)
tret = np.linspace(_.min(), _.max(), 25)


# We will loop over these target returns, finding the minimum variance portfolio for each target return.

# In[36]:


def port_vol(x, r, ppy):
    return np.sqrt(ppy) * r.dot(x).std()


# In[37]:


def port_mean(x, r, ppy):
    return ppy * r.dot(x).mean()


# In[38]:


res_ef = []

for t in tret:
    _ = sco.minimize(
        fun=port_vol, # minimize portfolio volatility
        x0=np.ones(returns_2.shape[1]) / returns_2.shape[1], # initial portfolio weights
        args=(returns_2, 252), # additional arguments to fun, in order
        bounds=[(0, 1) for c in returns_2.columns], # bounds limit the search space for each portfolio weight
        constraints=(
            {'type': 'eq', 'fun': lambda x: x.sum() - 1}, # constrain sum of weights to one
            {'type': 'eq', 'fun': lambda x: port_mean(x=x, r=returns_2, ppy=252) - t} # constrains portfolio mean return to the target return

        )
    )
    res_ef.append(_)


# List `res_ef` contains the results of all 25 minimum-variance portfolios.
# For example, `res_ef[0]` is the minimum variance portfolio for the lowest target return.

# In[39]:


res_ef[0]


# I typically check that all portfolio volatility minimization succeeds.
# If a portfolio volatility minimization fails, we should check our function, bounds, and constraints.

# In[40]:


for r in res_ef:
    assert r['success'] 


# We can combine the target returns and volatilities into a data frame `ef`.

# In[41]:


ef = pd.DataFrame(
    {
        'tret': tret,
        'tvol': np.array([r['fun'] if r['success'] else np.nan for r in res_ef])
    }
)

ef.head()


# In[42]:


ef.mul(100).plot(x='tvol', y='tret', legend=False)
plt.ylabel('Annualized Mean Return (%)')
plt.xlabel('Annualized Volatility (%)')
plt.title(
    f'Efficient Frontier for Dow-Jones Industrial Average Stocks' +
    f'\nfrom {returns_2.index[0]:%B %d, %Y} to {returns_2.index[-1]:%B %d, %Y}'
)

for t, x, y in zip(
    returns_2.columns, 
    returns_2.std().mul(100*np.sqrt(252)),
    returns_2.mean().mul(100*252)
):
    plt.annotate(text=t, xy=(x, y))
    
plt.show()


# ### Find the maximum Sharpe Ratio portfolio with all available data for the current the DJIA stocks

# In[43]:


res_sharpe_6 = sco.minimize(
    fun=port_sharpe_neg,
    x0=np.ones(returns_2.shape[1]) / returns_2.shape[1],
    args=(returns_2, 252, 0),
    bounds=[(0,1) for _ in range(returns_2.shape[1])],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1} # want eq constraint to = 0
    )
)


# In[44]:


port_sharpe(x=res_sharpe_6['x'], r=returns_2, ppy=252, tgt=0)


# ### Compare the $\frac{1}{n}$ and maximum Sharpe Ratio portfolios with all available data for the current DJIA stocks

# Use all but the last 252 trading days to estimate the maximum Sharpe Ratio portfolio weights.
# Then use the last 252 trading days of data to compare the $\frac{1}{n}$  maximum Sharpe Ratio portfolios.

# In[45]:


res_sharpe_x = sco.minimize(
    fun=port_sharpe_neg,
    x0=np.ones(returns_2.shape[1]) / returns_2.shape[1],
    args=(returns_2.iloc[:-252], 252, 0),
    bounds=[(0,1) for _ in range(returns_2.shape[1])],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1}, # want eq constraint to = 0
    )
)

assert res_sharpe_x['success']
# res_sharpe_x


# In[46]:


plt.barh(
    y=returns_2.columns,
    width=res_sharpe_x['x'],
    label='Maximum Sharpe Ratio'
)
plt.axvline(1/30, color='red', label='Equal Weight')
plt.legend()
plt.xlabel('Portfolio Weight')
plt.title(
    'Portfolio Weights for Dow-Jones Industrial Average Stocks' +
    f'\nfrom {returns_2.index[0]:%b %d, %Y} to {returns_2.index[-1]:%b %d, %Y}'
)
plt.show()


# In[47]:


port_sharpe(x=res_sharpe_x['x'], r=returns_2.iloc[:-252], ppy=252, tgt=0)


# In[48]:


port_sharpe(x=np.ones(returns_2.shape[1])/returns_2.shape[1], r=returns_2.iloc[:-252], ppy=252, tgt=0)


# Out of sample:

# In[49]:


port_sharpe(x=res_sharpe_x['x'], r=returns_2.iloc[-252:], ppy=252, tgt=0)


# In[50]:


port_sharpe(x=np.ones(returns_2.shape[1])/returns_2.shape[1], r=returns_2.iloc[-252:], ppy=252, tgt=0)


# It is hard to beat the $\frac{1}{n}$ portfolio because mean returns (and covariances) are hard to predict!

# In[ ]:




