
# coding: utf-8

# In[42]:


#-*- coding: utf-8 -*-
from sklearn.datasets import load_svmlight_file
import numpy as np
import pylab as pl
from sklearn.model_selection import train_test_split

import requests

#url="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#housing"
#urllib.request.urlretrieve(url,"housing_scale.txt")

X, y = load_svmlight_file("housing_scale.txt",n_features=13)
#X=X.data.reshpae(X.shape)
#print(X.shape,type(X))
#print(y.shape,type(y))
   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#print(X_train.shape,type(X_train))
#print(y_train.shape,type(y_train))
#print(X_test.shape,type(X_test))
#print(y_test.shape,type(y_test))

X_train=X_train.toarray()
X_test=X_test.toarray()

print(X_train.shape,type(X_train))
print(y_train.shape,type(y_train))
print(X_test.shape,type(X_test))
print(y_test.shape,type(y_test))


# In[43]:


alpha=0.01
iterations=200

m, n= np.shape (X_train)
W=np.ones ([n,1])

p,q = np.shape (X_test)

print(m,n)
losses_train=[]
losses_val=[]
for ecoph in range (iterations):   

    hypo=np.dot (X_train.T, ((np.dot (X_train,W))-y_train.reshape(m,1)))
    W-=alpha*hypo/(2*m)
           
    yp_train=np.dot (X_train,W)
    loss=np.average(((yp_train-y_train.reshape(m,1))**2)/2)
    #print("train loss:",loss)
    losses_train.append(loss)
    
    yp_val=np.dot (X_test,W)
    loss_val=np.average (((yp_val-y_test.reshape(p,1))**2)/2)
    losses_val.append ( loss_val)
    #print("test loss:",loss)


# In[44]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(16,10))
plt.plot(losses_train,color="blue",label="train loss")
plt.plot(losses_val,color="green",label="test loss")
plt.title("linear Regression and gradient descend")
plt.legend(loc='upper right')
plt.xlabel("literations")
plt.ylabel("loss")
plt.show()

