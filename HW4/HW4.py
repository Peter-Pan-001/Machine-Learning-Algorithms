#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# ## Problem 1 (a)

# In[2]:


file = open('./hw4_data/TeamNames.txt', 'r') 
mapping = np.array(file.read().split('\n'))
score = pd.read_csv('./hw4_data/CFB2018_scores.csv', header = None)


# In[3]:


score.loc[:,2].max()


# In[4]:


score = np.array(score)
M = np.zeros(shape = (767,767))

for i in range(len(score)):
    point_frac_A = score[i][1] / (score[i][1] + score[i][3])
    point_frac_B = score[i][3] / (score[i][1] + score[i][3])
    M[score[i][0] - 1][score[i][0] - 1] += point_frac_A
    M[score[i][2] - 1][score[i][2] - 1] += point_frac_B
    M[score[i][0] - 1][score[i][2] - 1] += point_frac_B
    M[score[i][2] - 1][score[i][0] - 1] += point_frac_A
    if score[i][1] > score[i][3]:
        M[score[i][0] - 1][score[i][0] - 1] += 1
        M[score[i][2] - 1][score[i][0] - 1] += 1
    elif score[i][1] < score[i][3]:
        M[score[i][0] - 1][score[i][2] - 1] += 1
        M[score[i][2] - 1][score[i][2] - 1] += 1

row_sum = np.sum(M, axis = 1)
for i in range(767):
    M[i,:] /= row_sum[i]
M


# In[5]:


w = np.array([1/767] * 767).reshape(1,-1)
w_store = []

for i in range(10000):
    w = np.dot(w, M)
    w_store.append(w)

top = []
for i in [9,99,999,9999]:
    top_list = mapping[np.argsort(w_store[i]).reshape(-1)[::-1][:25]]
    w_list = np.sort(w_store[i]).reshape(-1)[::-1][:25]
    top.extend([top_list, w_list])


# In[6]:


top_df = pd.DataFrame(np.array(top).T)
top_df.columns = ['t=10_name', 't=10_w', 't=100_name', 't=100_w','t=1000_name', 't=1000_w','t=10000_name', 't=10000_w']
top_df


# ## Problem 1 (b)

# In[7]:


eigenvalue, eigenvector = np.linalg.eig(M.T)
vector = eigenvector[:, np.argmax(eigenvalue)]
w_stationary = (vector / np.sum(vector)).reshape(-1)


# In[8]:


w = np.array([1/767] * 767).reshape(1,-1)
l1_store = []

for i in range(10000):
    w = np.dot(w, M)
    l1_store.append(np.sum(np.abs(w - w_stationary)))


# In[9]:


num = (np.array(range(10000)) + 1).reshape(-1,1)
distance = np.array(l1_store).reshape(-1,1)
plot = np.append(num, distance, axis = 1)
plot_df = pd.DataFrame(plot, columns = ['t', 'Norm1-distance'])
plot_df['t'] = plot_df['t'].astype(int)


# In[10]:


fig, ax = plt.subplots(1, 1, figsize = (20,10))
_ = ax.plot(plot_df.loc[:,'t'], plot_df.loc[:,'Norm1-distance'])
_ = ax.legend()
_ = ax.set_xlabel('Iteration')
_ = ax.set_ylabel('Norm1-distance')


# ## Problem 2 (a)

# In[28]:


file = open('./hw4_data/nyt_data.txt', 'r') 
documents = np.array(file.read().split('\n'))


# In[29]:


len(documents)


# In[30]:


documents[-1]


# In[38]:


X = np.zeros(shape = (3012, 8847))
W = np.zeros(shape = (3012, 25))
H = np.zeros(shape = (25, 8847))

# initialization
for i in range(len(documents) - 1):
    document = documents[i]
    word_count = document.split(',')
    for j in range(len(word_count)):
        temp = word_count[j].split(':')
        word = int(temp[0].strip())
        count = int(temp[1].strip())
        X[word-1][i] += count

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        W[i][j] = np.random.uniform(1,2)

for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        H[i][j] = np.random.uniform(1,2)


# In[39]:


loss_list = []
for i in range(100):
    H = H * np.dot(W.T,(X/(np.dot(W,H) + 10**(-16))))/(np.sum(W, axis = 0).reshape(-1,1) + 10**(-16))
    W = W * np.dot((X/(np.dot(W,H) + 10**(-16))),H.T)/(np.sum(H, axis = 1).reshape(1,-1) + 10**(-16))
    loss = np.sum(X * np.log(1/(np.dot(W,H) + 10**(-16))) + np.dot(W,H))
    loss_list.append(loss)


# In[40]:


num = (np.array(range(100)) + 1).reshape(-1,1)
loss_array = np.array(loss_list).reshape(-1,1)
plot = np.append(num, loss_array, axis = 1)
plot_df = pd.DataFrame(plot, columns = ['t', 'Objective'])
plot_df['t'] = plot_df['t'].astype(int)


# In[41]:


fig, ax = plt.subplots(1, 1, figsize = (20,10))
_ = ax.plot(plot_df.loc[:,'t'], plot_df.loc[:,'Objective'])
_ = ax.legend()
_ = ax.set_xlabel('Iteration')
_ = ax.set_ylabel('Objective')


# ## Problem 2 (b)

# In[48]:


file = open('./hw4_data/nyt_vocab.dat', 'r') 
names = np.array(file.read().split('\n'))[:-1]


# In[49]:


len(names)


# In[51]:


W_norm = W / np.sum(W, axis = 0)


# In[52]:


name_list = []
weight_list = []
for c in range(W.shape[1]):
    temp = W_norm[:, c].reshape(-1)
    name = names[np.argsort(temp)[::-1][:10]]
    weight = np.sort(temp)[::-1][:10]
    name_list.append(name)
    weight_list.append(weight)


# In[56]:


for i in range(W.shape[1]):
    print('topic {}\n'.format(i+1))
    df = pd.DataFrame(np.append(name_list[i].reshape(-1,1), weight_list[i].reshape(-1,1), axis = 1),                     columns = ['name', 'weight'])
    print(df)
    print('\n')


# In[ ]:




