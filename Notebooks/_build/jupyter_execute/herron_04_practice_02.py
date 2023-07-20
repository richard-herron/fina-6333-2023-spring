#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 4 - Practice (Wednesday 2:45 PM, Section 2)

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
#     - You must be in the room 
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


# ### Find the maximum Sharpe Ratio portfolio of MATANA stocks over the last three calendar years

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


def port_sharpe(x, r, tgt, ppy):
    rp = r.dot(x)
    er = rp.sub(tgt)
    return np.sqrt(ppy) * er.mean() / er.std()


# In[7]:


def port_sharpe_neg(x, r, tgt, ppy):
    return -1 * port_sharpe(x, r, tgt, ppy)


# In[8]:


def get_ew(r):
    return np.ones(r.shape[1]) / r.shape[1]


# In[9]:


get_ew(returns)


# In[10]:


[(0, 1) for i in returns]


# In[11]:


res_sharpe_1 = sco.minimize(
    fun=port_sharpe_neg,
    x0=get_ew(returns),
    args=(returns, 0, 252),
    bounds=[(0, 1) for i in returns],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1} # eq constraint met when equal to zero
    )
)

res_sharpe_1


# In[12]:


port_sharpe(x=res_sharpe_1['x'], r=returns, tgt=0, ppy=252)


# ### Find the maximum Sharpe Ratio portfolio of MATANA stocks over the last three years, but allow short weights up to 10% on each stock

# In[13]:


res_sharpe_2 = sco.minimize(
    fun=port_sharpe_neg,
    x0=get_ew(returns),
    args=(returns, 0, 252),
    bounds=[(-0.1, 1.5) for i in returns],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1} # eq constraint met when equal to zero
    )
)

res_sharpe_2


# In[14]:


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

# In[15]:


port_sharpe(res_sharpe_1['x'], r=returns, tgt=0, ppy=252)


# In[16]:


port_sharpe(res_sharpe_2['x'], r=returns, tgt=0, ppy=252)


# ### Find the maximum Sharpe Ratio portfolio of MATANA stocks over the last three years, but allow total short weights of up to 30%

# We can find the negative values in a NumPy array as follows.

# In[17]:


x = np.arange(6) - 3
x[x < 0]


# In[18]:


res_sharpe_3 = sco.minimize(
    fun=port_sharpe_neg,
    x0=get_ew(returns),
    args=(returns, 0, 252),
    bounds=[(-0.3, 1.3) for i in returns],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1}, # eq constraint met when = 0
        {'type': 'ineq', 'fun': lambda x: x[x<0].sum() + 0.3} # ineq constraint met when >= 0
    )
)

res_sharpe_3


# In[19]:


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

# In[20]:


port_sharpe(res_sharpe_1['x'], r=returns, ppy=252, tgt=0)


# In[21]:


port_sharpe(res_sharpe_3['x'], r=returns, ppy=252, tgt=0)


# ### Find the maximum Sharpe Ratio portfolio of MATANA stocks over the last three years, but do not allow any weight to exceed 30% in magnitude

# In[22]:


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


# ### Find the minimum 95% Value at Risk (Var) portfolio of MATANA stocks over the last three years

# More on VaR [here](https://en.wikipedia.org/wiki/Value_at_risk).

# In[23]:


def port_var(x, r, q):
    return r.dot(x).quantile(q)


# In[24]:


def port_var_neg(x, r, q):
    return -1 * port_var(x=x, r=r, q=q)


# In[25]:


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


# In[26]:


port_var(x=res_var_1['x'], r=returns, q=0.05)


# It might be helpful to slightly change then minimum VaR portfolio weights to show that we minimized VaR.

# In[27]:


def tweak(x, d=0.05):
    y = np.zeros(x.shape[0])
    y[0], y[1] = d, -1 * d
    return x + y


# In[28]:


port_var(x=tweak(res_var_1['x']), r=returns, q=0.05)


# ### Find the minimum maximum draw down portfolio of MATANA stocks over the last three years

# In[29]:


def port_draw_down_max(x, r):
    rp = r.dot(x)
    price = rp.add(1).cumprod()
    cum_max = price.cummax()
    draw_down = (cum_max - price) / cum_max
    return draw_down.max()


# In[30]:


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


# In[31]:


port_draw_down_max(x=res_dd_1['x'], r=returns)


# Again. it might be helpful to slightly change then minimum VaR portfolio weights to show that we minimized VaR.

# In[32]:


port_draw_down_max(x=tweak(res_dd_1['x']), r=returns)


# ### Find the minimum maximum draw down portfolio with all available data for the current Dow-Jones Industrial Average (DJIA) stocks

# You can find the [DJIA tickers on Wikipedia](https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average).

# In[33]:


wiki = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
tickers = wiki[1]['Symbol'].to_list()

djia = (
    yf.download(tickers=tickers, progress=False)
    .assign(Date=lambda x: x.index.tz_localize(None))
    .set_index('Date')
    .rename_axis(columns=['Variable', 'Ticker'])
)

returns_2 = djia['Adj Close'].pct_change().loc['2020':'2022']
returns_2.describe()


# In[34]:


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


# In[35]:


port_draw_down_max(x=res_dd_2['x'], r=returns_2)


# Again. it might be helpful to slightly change then minimum VaR portfolio weights to show that we minimized VaR.

# In[36]:


port_draw_down_max(x=tweak(res_dd_2['x']), r=returns_2)


# ### Plot the (mean-variance) efficient frontier with all available data for the current the DJIA stocks

# The range of target returns in `tret` span from the minimum to the maximum mean single-stock returns.

# In[37]:


_ = returns_2.mean().mul(252)
tret = np.linspace(_.min(), _.max(), 25)


# We will loop over these target returns, finding the minimum variance portfolio for each target return.

# In[38]:


def port_vol(x, r, ppy):
    return np.sqrt(ppy) * r.dot(x).std()


# In[39]:


def port_mean(x, r, ppy):
    return ppy * r.dot(x).mean()


# In[40]:


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

# In[41]:


res_ef[0]


# I typically check that all portfolio volatility minimization succeeds.
# If a portfolio volatility minimization fails, we should check our function, bounds, and constraints.

# In[42]:


for r in res_ef:
    assert r['success'] 


# We can combine the target returns and volatilities into a data frame `ef`.

# In[43]:


ef = pd.DataFrame(
    {
        'tret': tret,
        'tvol': np.array([r['fun'] if r['success'] else np.nan for r in res_ef])
    }
)

ef.head()


# In[44]:


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

# In[45]:


res_sharpe_6 = sco.minimize(
    fun=port_sharpe_neg,
    x0=np.ones(returns_2.shape[1]) / returns_2.shape[1],
    args=(returns_2, 252, 0),
    bounds=[(0,1) for _ in range(returns_2.shape[1])],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1} # want eq constraint to = 0
    )
)


# In[46]:


port_sharpe(x=res_sharpe_6['x'], r=returns_2, ppy=252, tgt=0)


# ### Compare the $\frac{1}{n}$ and maximum Sharpe Ratio portfolios with all available data for the current DJIA stocks

# Use all but the last 252 trading days to estimate the maximum Sharpe Ratio portfolio weights.
# Then use the last 252 trading days of data to compare the $\frac{1}{n}$  maximum Sharpe Ratio portfolios.

# In[47]:


res_sharpe_x = sco.minimize(
    fun=port_sharpe_neg,
    x0=get_ew(returns_2),
    args=(returns_2.loc['2020':'2021'], 0, 252),
    bounds=[(0, 1) for i in returns_2],
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1} # eq constraint met when equal to zero
    )
)

res_sharpe_x


# In[48]:


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


# In[49]:


port_sharpe(res_sharpe_x['x'], returns_2.loc['2022'], 0, 252)


# In[50]:


port_sharpe(get_ew(returns_2), returns_2.loc['2022'], 0, 252)


# It is hard to beat the $\frac{1}{n}$ portfolio because mean returns (and covariances) are hard to predict!

# ---

# Side discussion on the `.dot()` method.

# In[51]:


weights = get_ew(returns)


# In[52]:


np.allclose(
    weights.dot(returns.transpose()),
    returns.dot(weights)
)


# In[53]:


np.allclose(
    returns @ weights,
    returns.dot(weights)
)


# In[54]:


np.allclose(
    weights @ returns.transpose(),
    returns.dot(weights)
)

