#!/usr/bin/env python
# coding: utf-8

# # Project 2

# # Purpose

# This [November 2021 CNBC article](https://www.cnbc.com/2021/11/09/bitcoin-vs-gold-leading-gold-authorities-on-inflation-hedge-battle.html) on Bitcoin and gold as inflation and market risk hedges motivated this project.
# I have two goals for this project:
# 
# 1. To help you master data analysis
# 1. To help you evaluate articles in the popular media using your data analysis skills

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


pd.set_option('display.float_format', '{:.2f}'.format)
get_ipython().run_line_magic('precision', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[3]:


import yfinance as yf
import pandas_datareader as pdr
import requests_cache
session = requests_cache.CachedSession()


# In[4]:


import scipy.optimize as sco
import seaborn as sns
import statsmodels.formula.api as smf


# # Tasks

# ## Task 1: Do Bitcoin and gold hedge inflation risk?

# Use the typical finance definition of [hedge](https://www.investopedia.com/terms/h/hedge.asp):
# 
# > To hedge, in finance, is to take an offsetting position in an asset or investment that reduces the price risk of an existing position. A hedge is therefore a trade that is made with the purpose of reducing the risk of adverse price movements in another asset. Normally, a hedge consists of taking the opposite position in a related security or in a derivative security based on the asset to be hedged. 
# 
# Here are a few suggestions:
# 
# 1. Measure Bitcoin's price with [BTC-USD](https://finance.yahoo.com/quote/BTC-USD?p=BTC-USD&.tsrc=fin-srch) and gold's price with [GLD](https://finance.yahoo.com/quote/GLD?p=GLD&.tsrc=fin-srch)
# 1. Throughout the project, assume Bitcoin and U.S. public equity markets have the same closing time
# 1. Measure the price level with [PCEPI](https://fred.stlouisfed.org/series/PCEPI/) from the Federal Reserve Database (FRED), which is downloadable with `pdr.DataReader()`
# 1. Measure inflation (i.e., the rate of change in the price level) as the percent change in PCEPI

# ## Task 2: Do Bitcoin and gold hedge market risk?

# Here are a few suggestions:
# 
# 1. Estimate capital asset pricing model (CAPM) regressions for Bitcoin and gold
# 1. Use the daily factor data from Ken French

# ## Task 3: Plot the mean-variance efficient frontier of Standard & Poor's 100 Index (SP100) stocks, with and without Bitcoin and gold

# Here are a few suggestions:
# 
# 1. You can learn about the SP100 stocks [here](https://en.wikipedia.org/wiki/S%26P_100)
# 1. Only consider days with complete data for Bitcoin and gold
# 1. Drop any stocks with shorter return histories than Bitcoin and gold
# 1. Assume long-only portfolios

# ## Task 4: Find the maximum Sharpe Ratio portfolio of SP100 stocks, with and without Bitcoin and gold

# Follow the data requirements of task 3.

# ## Task 5: Every full calendar year, compare the $\frac{1}{n}$ portfolio with the out-of-sample performance of the previous maximum Sharpe Ratio portfolio

# Follow the data requirements of task 3.
# Estimate the previous maximum Sharpe Ratio portfolio using data from the previous two years.
# Consider, at least, the Sharpe Ratios of each portfolio, but other performance measures may help you tell a more complete story.

# ## Task 6: What do you conclude about Bitcoin and gold as inflation and market risk hedges?

# What are your overall conclusions and limitations of your analysis?
# What do the data suggest about the article that motivated this project?
# Please see the link at the top of this notebook.

# # Criteria

# 1. ***Discuss and explain your findings for all 6 tasks, and be specific!***
# 1. ***Your goal is to convince me of your calculations and conclusions***
# 1. All tasks are worth 16.67 points each
# 1. Your report should not exceed 25 pages
# 1. Here are more tips
#     1. Each task includes suggestions
#     1. I suggest you include plots and calculations for all but the last task
#     1. Remove unnecessary code, outputs, and print statements
#     1. Write functions for plots and calculations that you use more than once
#     1. I will not penalize code style, but I will penalize submissions that are difficult to follow or do not follow these instructions
# 1. How to submit your project
#     1. Restart your kernel, run all cells, and save your notebook
#     1. Export your notebook to PDF (`File > Save And Export Notebook As ... > PDF` in JupyterLab)
#         1. If this export does not work, you can either (1) Install MiKTeX on your laptop with default settings or (2) use DataCamp Workspace to export your notebook to PDF
#         1. You do not need to re-run your notebook to export it because notebooks store output cells
#     1. Upload your notebook and PDF to Canvas
#     1. Upload your PDF only to Gradescope and tag your tasks and teammates
#     1. Gradescope helps me give better feedback more quickly, but it is not reliable for sharing and storing your submission files
