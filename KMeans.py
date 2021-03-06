
# coding: utf-8

# In[14]:


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')

from sklearn import preprocessing
iris = datasets.load_iris()
X = pd.DataFrame(iris.data)

scaler = preprocessing.StandardScaler()
y = pd.DataFrame(iris.target)
y.columns = ['Targets']

scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)
xs.sample(5)


# In[15]:


from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)


# In[16]:


y_cluster_gmm = gmm.predict(xs)
y_cluster_gmm


# In[20]:


colormap = np.array(['red', 'lime', 'black'])
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title('GMM Classification')


# In[18]:


sm.accuracy_score(y, y_cluster_gmm)


# In[19]:


sm.confusion_matrix(y, y_cluster_gmm)

