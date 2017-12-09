
# coding: utf-8

# In[40]:


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

[x, y] = load_svmlight_file("australian_scale.txt",n_features=14)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

X_train=X_train.toarray()
X_test=X_test.toarray()

print(X_train.shape,type(X_train))
print(y_train.shape,type(y_train))
print(X_test.shape,type(X_test))
print(y_test.shape,type(y_test))


# In[41]:


m, n = X_train.shape
def svm(X_train,y_train,W):
    grad = np.ones([n,1])
    for i in range(m):
        X=X_train[i].reshape(1,n)
        y=y_train[i].reshape(1,1)
        #print("X:",X.shape)
        #print("y:",y.shape)
        h=1-np.multiply(y,np.dot(X,W))
        if(h>=0):
            grad+=W-np.dot(X.T,y)
        else:
            grad+=W
    W=W-lr*grad/m
    return W

def svm_loss(X_train,y_train,C,W):
    m, n = X_train.shape
    loss=0
    for i in range(m):
        X=X_train[i]
        y=y_train[i]
        h=1-np.multiply(y,np.dot(W.T,X))
        if(h>=0):
            loss+=C*h
    return loss/m+np.dot(W.T,W)/2
        


# In[42]:


C=0.9
lr= 0.01    #lr learning rate
epochs=100

train_loss=[]
test_loss=[]

m, n = X_train.shape
W = np.ones([n,1])

for i in range(epochs):
    print("epoch ",i,":")
    W=svm(X_train,y_train,W)
    loss=svm_loss(X_train,y_train,C,W)
    train_loss.append(loss[0][0])
    print("train loss: ",loss[0][0])
    
    loss=svm_loss(X_test,y_test,C,W)
    test_loss.append(loss[0][0])
    print("test loss: ",loss[0][0]) 
    


# In[43]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

plt.figure(figsize=(16,10))
plt.plot(train_loss,color="blue",label="train loss")
plt.plot(test_loss,color="green",label="test loss")
plt.title("linear Classification")
plt.legend(loc='upper right')
plt.xlabel("literations")
plt.ylabel("loss")
plt.show()

