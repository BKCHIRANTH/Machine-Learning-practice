
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:


df = pd.DataFrame({ 'x' : [12,20,28,18,19,29,23,33,45,51,52,51,45,53,55],
                  'y':    [39,36,30,52,45,54,46,55,59,63,74,66,63,67,77]})


# In[5]:


df


# In[9]:


from sklearn.cluster import KMeans


# In[10]:


KMeans = KMeans(n_clusters =2)


# In[11]:


KMeans.fit(df)


# In[12]:


labels = KMeans.predict(df)
centroid = KMeans.cluster_centers_


# In[13]:


fig = plt.figure(figsize =(5,5))


# In[14]:


colmap = {1:'r',2:'g',3:'b',4:'y'}
colors = map(lambda x:colmap[x+1],labels)
colors1 = list(colors)


# In[15]:


plt.scatter(df['x'],df['y'], color=colors1,alpha=0.5,edgecolor ='k')

