#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


X = pd.read_csv("/Users/panzichen/Columbia Term2 Courses/ML for DS/HW2/X.csv", header = None)
y = pd.read_csv("/Users/panzichen/Columbia Term2 Courses/ML for DS/HW2/y.csv", header = None)


# In[3]:


print(len(X), len(y))


# In[4]:


X.head()


# ### Question a

# In[5]:


X_array = np.array(X)
y_array = np.array(y)
X_array.shape


# In[6]:


from sklearn.model_selection import KFold

def partition(X_array, y_array):
    kf = KFold(n_splits=10, shuffle = True)
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    for train_index, test_index in kf.split(X_array):
        X_train, X_test = X_array[train_index], X_array[test_index]
        y_train, y_test = y_array[train_index], y_array[test_index]
        
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    
    assert len(X_train_list) == 10
    
    return X_train_list, y_train_list, X_test_list, y_test_list


# In[7]:


def cal_lamda(X_train, y_train):
    lamda = np.zeros(shape = (2,54))

    for yi in range(2):
        for d in range(54):
            
            if yi == 0:
                lamda[yi][d] = (1 + np.sum(np.multiply(X_train[:, d].reshape(-1, 1), 1 - y_train))) /                                (1 + np.sum(1 - y_train))
            
            else:
                lamda[yi][d] = (1 + np.sum(np.multiply(X_train[:, d].reshape(-1, 1), y_train))) /                                (1 + np.sum(y_train))
    return lamda


# In[8]:


def cal_likelihood(X_test, y_test, lamda):
    
    n = len(X_test)
    likelihood = np.zeros(shape = (n, 2))
    
    for xi in range(n):

        for yi in range(2):

            if yi == 1: likelihood[xi][yi] = np.sum(y_test) / n
            else: likelihood[xi][yi] = np.sum(1 - y_test) / n

            for d in range(54):
                likelihood[xi][yi] *= np.exp(-lamda[yi][d]) * np.power(lamda[yi][d], X_test[xi][d])

                # factorial
                k = X_test[xi][d]
                while k > 1: 
                    likelihood[xi][yi] /= k
                    k -= 1
    return likelihood
        


# In[9]:


def cal_metric(likelihood, y_test):
    y_predict = (likelihood[:,1].reshape(-1,1) > likelihood[:,0].reshape(-1,1)) * 1
    metric = pd.DataFrame(np.concatenate((y_test, y_predict), axis = 1), columns = ['y', 'y_predict'])
    
    TP = len(metric[(metric['y'] == 1) & (metric['y_predict'] == 1)])
    TN = len(metric[(metric['y'] == 0) & (metric['y_predict'] == 0)])
    FP = len(metric[(metric['y'] == 0) & (metric['y_predict'] == 1)])
    FN = len(metric[(metric['y'] == 1) & (metric['y_predict'] == 0)])
    #print("TP,TN,FP,FN:",TP,TN,FP,FN)
    return TP, TN, FP, FN


# In[10]:


# cross validation
X_train_list, y_train_list, X_test_list, y_test_list = partition(X_array, y_array)
TP_list = []
TN_list = []
FP_list = []
FN_list = []
lamda_list = []

for i in range(10):
    X_train, y_train, X_test, y_test = X_train_list[i], y_train_list[i], X_test_list[i], y_test_list[i]
    lamda = cal_lamda(X_train, y_train)
    lamda_list.append(lamda)
    likelihood = cal_likelihood(X_test, y_test, lamda)
    TP, TN, FP, FN = cal_metric(likelihood, y_test)
    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)


# In[11]:


TP = np.sum(TP_list)
TN = np.sum(TN_list)
FP = np.sum(FP_list)
FN = np.sum(FN_list)
assert TP + TN + FP + FN == 4600
accuracy = (TP + TN) / 4600
print("TP, TN, FP, FN, accuracy", TP, TN, FP, FN, accuracy)


# ### Question b

# In[12]:


plot_x = list(range(54))
lamda_array = np.mean(lamda_list, axis = 0)
y0_lamda = list(lamda_array[0,:])
y1_lamda = list(lamda_array[1,:])

fig, ax = plt.subplots(1,2, figsize=(20, 10), sharey = True)
markerline, stemlines, baseline = ax[0].stem(plot_x, y0_lamda)
_ = plt.setp(baseline, color='r', linewidth=2)
_ = ax[0].set_title('y=0')
markerline, stemlines, baseline = ax[1].stem(plot_x, y1_lamda)
_ = plt.setp(baseline, color='r', linewidth=2)
_ = ax[1].set_title('y=1')


# In[13]:


# For dimension 16 and 52, the estimated value of lambda for both when y=1 are larger that the counterpart, and y=1 means
# spam email, which means the word 'free' and character '!' appears more frequently in spam email.


# ### Question C

# In[14]:


def k_nearest_neighbors(k, X_train, y_train, X_test, y_test):
    
    distance_matrix = np.zeros(shape = (len(X_test), len(X_train)))
    y_predict = np.zeros(shape = (len(X_test), 1))
    for i in range(len(X_test)):
        
        distance_matrix[i] = np.sum(np.abs(X_test[i] - X_train), axis = 1).T
        nearest_index = np.argsort(distance_matrix[i])[:k]
        
        count0, count1 = 0, 0
        for num in range(k):
            if y_train[nearest_index[num]] == 1:
                count1 += 1
            else:
                count0 += 1
        
        if count1 > count0:
            y_predict[i][0]= 1
        else:
            y_predict[i][0]= 0
    
    return y_predict


# In[15]:


def cal_accuracy(X_array, y_array, k):
    X_train_list, y_train_list, X_test_list, y_test_list = partition(X_array, y_array)
    accuracy_list = []

    for i in range(10):
        X_train, y_train, X_test, y_test = X_train_list[i], y_train_list[i], X_test_list[i], y_test_list[i]
        y_predict = k_nearest_neighbors(k, X_train, y_train, X_test, y_test)
        
        metric = pd.DataFrame(np.concatenate((y_test, y_predict), axis = 1), columns = ['y', 'y_predict'])
        accuracy = len(metric[metric['y'] == metric['y_predict']]) / len(metric)
        accuracy_list.append(accuracy)
        print("i = {} finished".format(i))
    
    return np.mean(accuracy_list)


# In[16]:


k_plot = list(range(1,21))
accuracy_plot = []
for k in range(1,21):
    accuracy_plot.append(cal_accuracy(X_array, y_array, k))
    print("k = {} finished".format(k))

k_plot = np.array(k_plot).reshape(-1,1)
accuracy_plot = np.array(accuracy_plot).reshape(-1,1)
plot_metric = pd.DataFrame(np.concatenate((k_plot, accuracy_plot), axis = 1), columns = ['k', 'accuracy'])


# In[17]:


plt.figure(figsize=(20,10))
plt.plot('k', 'accuracy', marker='o', markerfacecolor='blue', markersize=3, data=plot_metric, color='skyblue')
_ = plt.xlabel('k')
_ = plt.ylabel('accuracy')


# ### Question d

# In[18]:


X_array_2 = X_array
added = np.array([1] * 4600).reshape(-1,1)
X_array_2 = np.concatenate((X_array_2, added), axis = 1)
X_array_2[:,54]


# In[19]:


y_array_2 = y_array
for i in range(len(y_array_2)):
    if y_array_2[i] == 0:
        y_array_2[i] = -1


# In[20]:


def derivative(X_train, y_train, w):
    a = 1 / (1 + np.exp(np.multiply(y_train, np.dot(X_train, w))))
    b = np.multiply(y_train, np.multiply(a, X_train))
    return np.sum(b, axis = 0).T.reshape(-1,1)
    


# In[21]:


X_train_list, y_train_list, X_test_list, y_test_list = partition(X_array_2, y_array_2)
plot_matrix = np.array(list(range(1,1001))).reshape(-1,1)
for i in range(10):
    X_train, y_train, X_test, y_test = X_train_list[i], y_train_list[i], X_test_list[i], y_test_list[i]
    L_list = []
    w = np.zeros(shape = (55, 1))
    
    for j in range(1000):
        w += (0.01/4600) * derivative(X_train, y_train, w)
        L = np.sum(np.log(1 / (1 + np.exp(-np.multiply(y_train, np.dot(X_train, w))))))
        L_list.append(L)
    
    plot_matrix = np.concatenate((plot_matrix, np.array(L_list).reshape(-1,1)), axis = 1)


# In[22]:


plot_df = pd.DataFrame(plot_matrix, columns = ['i', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plot_df.head()


# In[23]:


plt.figure(figsize=(20,10))
plt.plot('i', '1', data = plot_df)
plt.plot('i', '2', data = plot_df)
plt.plot('i', '3', data = plot_df)
plt.plot('i', '4', data = plot_df)
plt.plot('i', '5', data = plot_df)
plt.plot('i', '6', data = plot_df)
plt.plot('i', '7', data = plot_df)
plt.plot('i', '8', data = plot_df)
plt.plot('i', '9', data = plot_df)
plt.plot('i', '10', data = plot_df)
_ = plt.legend()
_ = plt.xlabel('Iteration')
_ = plt.ylabel('L')


# ### Question e

# In[24]:


def second_derivative(X_train, y_train, w):
    agge = np.zeros(shape = (55,55))
    for m in range(0, X_train.shape[0]):
        a =  1 / (1 + np.exp(-np.dot(X_train[m], w)))
        b = -1 / (1 + np.exp(np.dot(X_train[m], w)))
        agge += a * b * np.outer(X_train[m], X_train[m])
    return agge


# In[25]:


X_train_list, y_train_list, X_test_list, y_test_list = partition(X_array_2, y_array_2)
plot_matrix = np.array(list(range(1,101))).reshape(-1,1)
w_list = []
for i in range(10):
    X_train, y_train, X_test, y_test = X_train_list[i], y_train_list[i], X_test_list[i], y_test_list[i]
    L_list = []
    w = np.zeros(shape = (55, 1))
    
    for j in range(100):
        w -= np.dot(derivative(X_train, y_train, w).T, np.linalg.inv(second_derivative(X_train, y_train, w))).T
        L = np.sum(np.log(1 / (1 + np.exp(-np.multiply(y_train, np.dot(X_train, w))))))
        L_list.append(L)
    
    w_list.append(w)
    
    plot_matrix = np.concatenate((plot_matrix, np.array(L_list).reshape(-1,1)), axis = 1)


# In[26]:


plot_df = pd.DataFrame(plot_matrix, columns = ['i', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plot_df.head()


# In[27]:


plt.figure(figsize=(20,10))
plt.plot('i', '1', data = plot_df)
plt.plot('i', '2', data = plot_df)
plt.plot('i', '3', data = plot_df)
plt.plot('i', '4', data = plot_df)
plt.plot('i', '5', data = plot_df)
plt.plot('i', '6', data = plot_df)
plt.plot('i', '7', data = plot_df)
plt.plot('i', '8', data = plot_df)
plt.plot('i', '9', data = plot_df)
plt.plot('i', '10', data = plot_df)
_ = plt.legend()
_ = plt.xlabel('Iteration')
_ = plt.ylabel('L')


# ### Question f

# In[28]:


TP_list = []
TN_list = []
FP_list = []
FN_list = []
lamda_list = []

for i in range(10):
    X_train, y_train, X_test, y_test = X_train_list[i], y_train_list[i], X_test_list[i], y_test_list[i]
    w = w_list[i]
    prob_1 = 1 / (1 + np.exp(-np.dot(X_test, w))).reshape(-1,1)
    y_predict = np.zeros(shape = (460,1))
    
    # prob > 0.5 :  -> 1
    for j in range(prob_1.shape[0]):
        if prob_1[j][0] > 0.5: y_predict[j][0] = 1
        else: y_predict[j][0] = -1
    
    metric = pd.DataFrame(np.concatenate((y_test, y_predict), axis = 1), columns = ['y', 'y_predict'])
    
    # calculate TP, TN, FP, FN
    TP = len(metric[(metric['y'] == 1) & (metric['y_predict'] == 1)])
    TN = len(metric[(metric['y'] == -1) & (metric['y_predict'] == -1)])
    FP = len(metric[(metric['y'] == -1) & (metric['y_predict'] == 1)])
    FN = len(metric[(metric['y'] == 1) & (metric['y_predict'] == -1)])
    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)
    
TP = np.sum(TP_list)
TN = np.sum(TN_list)
FP = np.sum(FP_list)
FN = np.sum(FN_list)
assert TP + TN + FP + FN == 4600
accuracy = (TP + TN) / 4600
print("TP, TN, FP, FN, accuracy", TP, TN, FP, FN, accuracy)


# In[ ]:




