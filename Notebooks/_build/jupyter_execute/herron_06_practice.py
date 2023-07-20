#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 6 - Practice (Blank)

# ## Announcements

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


# ### Reimplement the equal-weighted momentum investing strategy from the lecture notebook

# Try to use a few cells and temporary variables as you can (i.e., perform calculations inside `pd.concat()`).

# ### Add a long-short portfolio that is long portfolio 10 and short portfolio 1

# Call this long-short portfolio UMD.
# What are the best and worst months for portfolios 1, 10, and UMD?

# ### What are the Sharpe Ratios on these 11 portfolios?

# ### Implement a value-weighted momentum investing strategy

# Assign this strategy to data frame `mom_vw`, and include long-short portfolio UMD

# ### What are the CAPM and FF4 alphas for these value-weighted portfolios?

# ### What are the Sharpe Ratios for these value-weighted portfolios?

# ### Implement an equal-weighted size investing strategy based on market capitalization at the start of each month

# ### Implement a value-weighted size investing strategy based on market capitalization at the start of each month
