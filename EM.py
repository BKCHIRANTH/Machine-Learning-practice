{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>-1.264185</td>\n",
       "      <td>0.800654</td>\n",
       "      <td>-1.056944</td>\n",
       "      <td>-1.312977</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>-1.264185</td>\n",
       "      <td>0.106445</td>\n",
       "      <td>-1.227541</td>\n",
       "      <td>-1.312977</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>-1.143017</td>\n",
       "      <td>0.106445</td>\n",
       "      <td>-1.284407</td>\n",
       "      <td>-1.444450</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>0.068662</td>\n",
       "      <td>-0.124958</td>\n",
       "      <td>0.762759</td>\n",
       "      <td>0.790591</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>53</th>\n",
       "      <td>-0.416010</td>\n",
       "      <td>-1.744778</td>\n",
       "      <td>0.137236</td>\n",
       "      <td>0.133226</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            0         1         2         3\n",
       "24  -1.264185  0.800654 -1.056944 -1.312977\n",
       "30  -1.264185  0.106445 -1.227541 -1.312977\n",
       "37  -1.143017  0.106445 -1.284407 -1.444450\n",
       "149  0.068662 -0.124958  0.762759  0.790591\n",
       "53  -0.416010 -1.744778  0.137236  0.133226"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from sklearn import datasets\n",
    "from sklearn.cluster import KMeans\n",
    "import sklearn.metrics as sm\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "%matplotlib inline\n",
    "\n",
    "from sklearn import preprocessing\n",
    "iris = datasets.load_iris()\n",
    "\n",
    "#print(\"\\n IRIS DATA :\",iris.data);\n",
    "#print(\"\\n IRIS FEATURES :\\n\",iris.feature_names)\n",
    "#print(\"\\n IRIS TARGET :\\n\",iris.target)\n",
    "#print(\"\\n IRIS TARGET NAMES:\\n\",iris.target_names)\n",
    "\n",
    "# Store the inputs as a Pandas Dataframe and set the column names\n",
    "X = pd.DataFrame(iris.data)\n",
    "\n",
    "scaler = preprocessing.StandardScaler()\n",
    "\n",
    "scaler.fit(X)\n",
    "xsa = scaler.transform(X)\n",
    "xs = pd.DataFrame(xsa, columns = X.columns)\n",
    "xs.sample(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "y = pd.DataFrame(iris.target)\n",
    "y.columns = ['Targets']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.figure.Figure at 0xaf8f4735c0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(14,7))\n",
    " \n",
    "# Create a colormap\n",
    "colormap = np.array(['red', 'lime', 'black'])\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,\n",
       "        means_init=None, n_components=3, n_init=1, precisions_init=None,\n",
       "        random_state=None, reg_covar=1e-06, tol=0.001, verbose=0,\n",
       "        verbose_interval=10, warm_start=False, weights_init=None)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.mixture import GaussianMixture\n",
    "gmm = GaussianMixture(n_components=3)\n",
    "gmm.fit(xs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,\n",
       "       2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int64)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_cluster_gmm = gmm.predict(xs)\n",
    "y_cluster_gmm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5,1,'GMM Classification')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nb9kAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0xaf927b6a20>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.subplot(1, 2, 1)\n",
    "plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)\n",
    "plt.title('GMM Classification')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.36666666666666664"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sm.accuracy_score(y, y_cluster_gmm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[50,  0,  0],\n",
       "       [ 0,  5, 45],\n",
       "       [ 0, 50,  0]], dtype=int64)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sm.confusion_matrix(y, y_cluster_gmm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
