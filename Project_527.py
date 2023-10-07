#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# In[61]:


filename = "C:/Users/hgcwi/Downloads/bank-full.csv"

df = pd.read_csv(filename, delimiter=';')

print(df)

df.describe(include='all')


# In[56]:


#for c in df.columns:
    #print(c, ":", df[c].unique())

df.replace("unknown", pd.NA, inplace=True)

df.fillna(0, inplace=True)


# In[59]:


print(df.isnull().sum())


# In[42]:


# df.hist(figsize=(15, 10))
# plt.tight_layout()
# plt.show()


# In[63]:


corr = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.show()


# In[18]:


cols = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']


# In[23]:


df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.iloc[:, :-1]

y = df.iloc[:, -1]


# In[36]:


scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_standardized)

print(X_pca)


# In[37]:


print(pca.explained_variance_ratio_)


# In[38]:


cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print(cumulative_variance)


# In[ ]:




