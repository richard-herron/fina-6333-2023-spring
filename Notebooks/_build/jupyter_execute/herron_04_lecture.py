#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 4 - Portfolio Optimization

# This notebook covers portfolio optimization.
# I have not found a perfect reference that combines portfolio optimization and Python, but here are two references that I find useful:
# 
# 1. Ivo Welch discusses the mathematics and finance of portfolio optimization in [Chapter 12 of his draft textbook on investments](https://book.ivo-welch.info/bookg.pdf#chapter.12).
# 1. Eryk Lewinson provides Python code for portfolio optimization in chapter 7 of his [*Python for Finance Cookbook*](https://onesearch.library.northeastern.edu/permalink/01NEU_INST/i2gqis/alma9952082522901401), but he uses several packages that are either non-free or abandoned.
# 
# In this notebook, we will:
# 
# 1. Review the $\frac{1}{n}$ portfolio (or equal-weighted portfolio) from [Herron Topic 1](herron_01_lecture.ipynb)
# 1. Use SciPy's `minimize()` function to:
#     1. Find the minimum variance portfolio
#     1. Find the (mean-variance) efficient frontier
# 
# In the practice notebook, we will use SciPy's `minimize()` function to achieve any objective.

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


# ## The $\frac{1}{n}$ Portfolio

# We first saw the $\frac{1}{n}$ portfolio (or equal-weighted portfolio) in [Herron Topic 1](herron_01_lecture.ipynb).
# In the $\frac{1}{n}$ portfolio, each of $n$ assets receives an equal portfolio weight $w_i = \frac{1}{n}$.
# While the $\frac{1}{n}$ strategy seems too simple to be useful, DeMiguel, Garlappi, and Uppal (2007) show that it is difficult to beat $\frac{1}{n}$ strategy, even with more advanced strategies.

# In[4]:


tickers = 'MSFT AAPL TSLA AMZN NVDA GOOG'

matana = (
    yf.download(tickers=tickers, progress=False)
    .assign(Date=lambda x: x.index.tz_localize(None))
    .set_index('Date')
    .rename_axis(columns=['Variable', 'Ticker'])
)

matana.tail()


# In[5]:


returns = matana['Adj Close'].pct_change().iloc[(-3 * 252):]

returns.describe()


# Before we revisit the advanced techniques from [Herron Topic 1](herron_01_lecture), we can calculate $\frac{1}{n}$ portfolio returns manually, where $R_P = \frac{\sum_{i}^{n} R_i}{n}$
# Since our weights are constant (i.e., do not change over time), we rebalance our portfolio every return period.
# If we have daily data, rebalance daily.
# If we have monthly data, we rebalance monthly, and so on.

# In[6]:


n = returns.shape[1]
p1 = returns.sum(axis=1).div(n)

p1.describe()


# Recall from [Herron Topic 1](herron_01_lecture) we have two better options:
# 
# 1. The `.mean(axis=1)` method for the $\frac{1}{n}$ portfolio
# 1. The `.dot(weights)` method where `weights` is a pandas series or NumPy array of portfolio weights, allowing different weights for each asset

# In[7]:


p2 = returns.mean(axis=1)

p2.describe()


# In[8]:


weights = np.ones(n) / n
p3 = returns.dot(weights)

p3.describe()


# The `.describe()` method provides summary statistics for data, letting us make quick comparisons.
# However, we should use `np.allclose()` if we want to be sure that `p1`, `p2`, and `p3` are similar.

# In[9]:


np.allclose(p1, p2)


# In[10]:


np.allclose(p1, p3)


# ---

# Here is a simple example to help understand the `.dot()` method.

# In[11]:


silly_n = 3
silly_R = pd.DataFrame(np.arange(2*silly_n).reshape(2, silly_n))
silly_w = np.ones(3) / 3


# In[12]:


print(
    f'silly_n:\n{silly_n}',
    f'silly_R:\n{silly_R}',
    f'silly_w:\n{silly_w}',
    sep='\n\n'
)


# In[13]:


silly_R.dot(silly_w)


# Under the hood, Python and the `.dot()` method (effectively) do the following calculation:

# In[14]:


for i, row in silly_R.iterrows():
    print(
        f'Row {i}: ',
        ' + '.join([f'{w:0.2f} * {y}' for w, y in zip(silly_w, row)]),
        ' = ',
        f'{silly_R.dot(silly_w).iloc[i]:0.2f}'
    )


# ---

# ## SciPy's `minimize()` Function 

# ### A Crash Course in SciPy's `minimize()` Function

# The `minimize()` function from SciPy's `optimize` module finds the input array `x` that minimizes the output of the function `fun`.
# The `minimize()` function uses optimization techniques that are outside this course, but you can consider these optimization techniques to be sophisticated trial and error.
# 
# Here are the most common arguments we will pass to the `minimize()` function:
# 
# 1. We pass our first guess for input array `x` to argument `x0=`.
# 1. We pass additional arguments for function `fun` as a tuple to argument `args=`.
# 1. We pass lower and upper bounds on `x` as a tuple of tuples to argument `bounds=`.
# 1. We constrain our results with a tuple of dictionaries of functions to argument `contraints=`.
# 
# Here is a simple example that minimizes the function `quadratic()` that accepts arguments `x` and `a` and returns $y = (x - a)^2$.

# In[15]:


import scipy.optimize as sco


# In[16]:


def quadratic(x, a=5):
    return (x - a) ** 2


# In[17]:


quadratic(x=5, a=5)


# In[18]:


quadratic(x=10, a=5)


# It is helpful to plot $y = (x - a)$ first.

# In[19]:


x = np.linspace(-5, 15, 101)
y = quadratic(x=x)
plt.plot(x, y)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$y = (x - 5)^2$')
plt.show()


# The minimum output of `quadratic()` occurs at $x=5$ if we do not use bounds or constraints, even if we start far away from $x=5$.

# In[20]:


sco.minimize(
    fun=quadratic,
    x0=np.array([2001])
)


# The minimum output of `quadratic()` occurs at $x=6$ if we bound `x` between 6 and 10 (i.e., $6 \leq x \leq 10$).

# In[21]:


sco.minimize(
    fun=quadratic,
    x0=np.array([2001]),
    bounds=((6, 10),)
)


# The minimum output of `quadratic()` occurs at $x=6$, again, if we constrain `x - 6` to be non-negative.
# We use bounds to limit the search space directly, and we use constraints to limit the search space indirectly based on a formula.

# In[22]:


sco.minimize(
    fun=quadratic,
    x0=np.array([2001]),
    constraints=({'type': 'ineq', 'fun': lambda x: x - 6})
)


# We can use the `args=` argument to pass additional arguments to `fun`.
# For example, we change the `a=` argument in `quadratic()` from the default of `a=5` to `a=20` with `args=(20,)`.
# Note that `args=` expects a tuple, so we need a trailing comma `,` if we have one argument.

# In[23]:


sco.minimize(
    fun=quadratic,
    args=(20,),
    x0=np.array([2001]),
)


# ### The Minimum Variance Portfolio

# We can find the minimum variance portfolio with `minimize()` function from SciPy's `optimize` module.
# The `minimize()` function with vary an input array `x` (starting from argument `x0=`) to minimize the objective function `fun=` subject to the bounds and constraints in `bounds=` and `constraints=`. 
# We will define a function `port_vol()` to calculate portfolio volatility.
# The first argument to `port_vol()` must be the input array `x` that the `minimize()` function searches over.
# For clarity, we will call this first argument `x`, but the argument's name is not important.

# In[24]:


def port_vol(x, r, ppy):
    return np.sqrt(ppy) * r.dot(x).std()


# We will eventually need a mean portfolio return function, too.

# In[25]:


def port_mean(x, r, ppy):
    return ppy * r.dot(x).mean()


# In[26]:


res_mv = sco.minimize(
    fun=port_vol, # objective function that we minimize
    x0=np.ones(returns.shape[1]) / returns.shape[1], # initial portfolio weights
    args=(returns, 252), # additional arguments to our objective function
    bounds=[(0,1) for _ in returns], # bounds limit the search space for each portfolio weights
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1} # minimize drives "eq" constraints to zero
    )
)

print(res_mv)


# What are the attributes of this minimum variance portfolio?

# In[27]:


def print_port_res(w, r, title, ppy=252, tgt=None):
    width = len(title)
    rp = r.dot(w)
    mu = ppy * rp.mean()
    sigma = np.sqrt(ppy) * rp.std()
    if tgt is not None:
        er = rp.sub(tgt)
        sr = np.sqrt(ppy) * er.mean() / er.std()
    else:
        sr = None
    
    return print(
        title,
        '=' * width,
        '',
        'Performance',
        '-' * width,
        'Return:'.ljust(width - 6) + f'{mu:0.4f}',
        'Volatility:'.ljust(width - 6) + f'{sigma:0.4f}',
        'Sharpe Ratio:'.ljust(width - 6) + f'{sr:0.4f}\n' if sr is not None else '',
        'Weights', 
        '-' * width, 
        '\n'.join([f'{_r}:'.ljust(width - 6) + f'{_w:0.4f}' for _r, _w in zip(r.columns, w)]),
        sep='\n',
    )


# In[28]:


print_port_res(w=res_mv['x'], r=returns, title='Minimum Variance Portfolio')


# ### The (Mean-Variance) Efficient Frontier

# We will use the `minimize()` function to map the efficient frontier.
# Here is a basic outline:
# 
# 1. Create a NumPy array `tret` of target returns
# 1. Create an empty list `res_ef` of `minimize()` results
# 1. Loop over `tret`, passing each as a constraint to the `minimize()` function
# 1. Append each `minimize()` result to `res_ef`

# In[29]:


tret = 252 * np.linspace(returns.mean().min(), returns.mean().max(), 25)

tret


# We will loop over these target returns, finding the minimum variance portfolio for each target return.

# In[30]:


res_ef = []

for t in tret:
    _ = sco.minimize(
        fun=port_vol, # minimize portfolio volatility
        x0=np.ones(returns.shape[1]) / returns.shape[1], # initial portfolio weights
        args=(returns, 252), # additional arguments to fun, in order
        bounds=[(0, 1) for c in returns.columns], # bounds limit the search space for each portfolio weight
        constraints=(
            {'type': 'eq', 'fun': lambda x: x.sum() - 1}, # constrain sum of weights to one
            {'type': 'eq', 'fun': lambda x: port_mean(x=x, r=returns, ppy=252) - t} # constrains portfolio mean return to the target return

        )
    )
    res_ef.append(_)


# List `res_ef` contains the results of all 25 minimum-variance portfolios.
# For example, `res_ef[0]` is the minimum variance portfolio for the lowest target return.

# In[31]:


res_ef[0]


# I typically check that all portfolio volatility minimization succeeds.
# If a portfolio volatility minimization fails, we should check our function, bounds, and constraints.

# In[32]:


for r in res_ef:
    assert r['success'] 


# We can combine the target returns and volatilities into a data frame `ef`.

# In[33]:


ef = pd.DataFrame(
    {
        'tret': tret,
        'tvol': np.array([r['fun'] if r['success'] else np.nan for r in res_ef])
    }
)

ef.head()


# In[34]:


ef.mul(100).plot(x='tvol', y='tret', legend=False)
plt.ylabel('Annualized Mean Return (%)')
plt.xlabel('Annualized Volatility (%)')
plt.title(
    f'Efficient Frontier' +
    f'\nfor {", ".join(returns.columns)}' +
    f'\nfrom {returns.index[0]:%B %d, %Y} to {returns.index[-1]:%B %d, %Y}'
)

for t, x, y in zip(
    returns.columns, 
    returns.std().mul(100*np.sqrt(252)),
    returns.mean().mul(100*252)
):
    plt.annotate(text=t, xy=(x, y))
    
plt.show()


# In[ ]:




