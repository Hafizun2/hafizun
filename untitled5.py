# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17gW-RUxW4UZSqGMqYXj55NZINqb7TLAH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['figure.figsize'] = (10, 5)
plt.style.use('fivethirtyeight')

df = pd.read_excel('/content/2022 Data test2.xlsx')
df.fillna(0)

df1 = df.pivot_table(index =["Channel", "Servicing Branch"],
                       columns = "Transaction type",
                       values ="Premium(RM)", aggfunc =sum)
df1

df_mean = df.rolling(window=12).mean()

# Set datestamp column as index
df = df.set_index('Bill Transaction Date')

# Compute the 52 weeks rolling mean of the df DataFrame
ma = df.rolling(window=52).mean()

# Compute the 52 weeks rolling standart deviation of the co2_levels DataFrame
mstd = df.rolling(window=52).std()

# Add the upper bound column to the ma DataFrame
ma['upper'] = ma['co2'] + (2 * mstd['co2'])

# Add the lower bound column to the ma DataFrame
ma['lower'] = ma['co2'] - (2 * mstd['co2'])

# Plot the content of the ma DataFrame
ax = ma.plot(linewidth=0.8, fontsize=6);

# Specify labels, legend, and show the plot
ax.set_xlabel('Date', fontsize=10);
ax.set_ylabel('CO2 levels in Mauai Hawaii', fontsize=10);
ax.set_title('Rolling mean and variance of CO2 levels\nin Mauai Hawaii from 1958 to 2001',
             fontsize=10);
plt.savefig('../images/rolling_minmax.png')