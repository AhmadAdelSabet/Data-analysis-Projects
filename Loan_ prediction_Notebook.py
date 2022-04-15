#!/usr/bin/env python
# coding: utf-8

# In[2]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# In[4]:


df = pd.read_csv('loan_train.csv')
df.head(10)


# In[5]:


df.shape


# In[6]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[7]:


df['loan_status'].value_counts()


# In[8]:


get_ipython().system('conda install -c anaconda seaborn -y')


# In[11]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[12]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[13]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[14]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[15]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[16]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[17]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[18]:


df[['Principal','terms','age','Gender','education']].head()


# In[19]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[20]:


X = Feature
X[0:5]


# In[40]:


df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1],inplace=True)
df.head()
y = df['loan_status'].values
y[0:5]


# In[41]:


from sklearn.preprocessing import LabelEncoder


# In[42]:


labelencoder= LabelEncoder()
df['loan_status']= labelencoder.fit_transform(df['loan_status'])
df


# In[43]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[44]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[45]:


from sklearn.neighbors import KNeighborsClassifier


# In[46]:



k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[47]:


yhat = neigh.predict(X_test)
yhat[0:5]


# In[48]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[30]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[49]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[50]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[51]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,average='macro')


# In[52]:


from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision is: ', precision_score(y_test, yhat))
print('Recall is: ', recall_score(y_test, yhat))
print('F1 is: ', f1_score(y_test, yhat))


# In[53]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat)


# In[54]:


from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters


# In[55]:


drugTree.fit(X_train,y_train)


# In[56]:


predTree = drugTree.predict(X_test)


# In[57]:


print (predTree [0:5])
print (y_test [0:5])


# In[58]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))


# In[59]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,average='macro')


# In[60]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat)


# In[61]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 


# In[62]:


yhat = clf.predict(X_test)
yhat [0:5]


# In[63]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# In[64]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat, average='macro')


# In[65]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat)


# In[66]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[67]:


yhat = LR.predict(X_test)
yhat


# In[68]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[69]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,average='macro')


# In[70]:


from sklearn.metrics import log_loss
log_loss(y_test, yhat)


# In[71]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted')


# In[72]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[73]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# In[77]:


test_df = pd.read_csv('loan_test.csv')
test_df.head(50)


# In[ ]:




