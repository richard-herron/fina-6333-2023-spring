#!/usr/bin/env python
# coding: utf-8

# # Herron Topic 5 - Practice (Blank)

# ## Announcements

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

# ### Simulate your wealth $W_T$ by randomly sampling market returns

# Use monthly market returns from the French Data Library.
# Only invest one cash flow $W_0$, and plot the distribution of $W_T$.

# ### Repeat the exercise above but add end-of-month investments $C_t$
