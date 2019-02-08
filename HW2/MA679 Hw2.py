#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# MA679 Hw2 Jiahao Xu


# In[ ]:


#3.1, 3.2, 3.5, 3.6, ,3.11, 3.12, 3.13, 3.14  pg120


# In[1]:


#3.1
# H0 for Sales: Without TV, Radio and Newspaper ads, sales are zero.
# H0 for TV:With Radio and Newspaper ads, there is no relationship 
#between TV and sales.
# H0 for Radio:With TV and Newspaper ads, there is no relationship 
#between Radio and sales.
# H0 for Newspaper:With Radio and TV ads, there is no relationship 
#between Newspaper and sales.
# Based on the p-value, we can conclude that there is a relationship 
#between TV ads and Sales, and between Radio ads and Sales. 
# Since the p-value of TV and Radio is significant, then we reject 
#the null hypothesis.


# In[2]:


#3.2
# Both KNN classifier and KNN regression methods start by identifying 
# the K nearest neighbours. But they have the different result.
# KNN classifier will have different observations with different K values. 
#KNN regression methods will count the average value of different K values.


# In[10]:


from IPython.display import Image
Image(filename="/Users/apple/Desktop/111.jpg")


# In[10]:


#3.11
import numpy as np # package to create random distribution
import pandas as pd # package to create data frame
import statsmodels.formula.api as sfa
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(100)
x = np.random.normal(size=100)
y = 2*x+np.random.normal(size=100)
data1 = pd.DataFrame({'x': x, 'y': y})

fig, ax = plt.subplots()
sns.regplot(x='x', y='y', data=data1, scatter_kws={"s": 50, "alpha": 1}, ax=ax)
ax.axhline(color='black')
ax.axvline(color='black')


# In[11]:


#(a)
mod1= sfa.ols('y ~ x + 0', data1).fit()
mod1.summary()


# In[12]:


#(b)
mod2= sfa.ols('x ~ y + 0', data1).fit()
mod2.summary()


# In[13]:


#(c) The result of (a) and (b) have the same t value, but the coefficients are not inverse.


# In[21]:


#(f) 
mod3= sfa.ols('x ~ y ', data1).fit()
mod4= sfa.ols('y ~ x ', data1).fit()
print(mod3.tvalues)
print(mod4.tvalues)


# In[ ]:




