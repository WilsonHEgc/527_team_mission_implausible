#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.compose import ColumnTransformer


# In[2]:


filename = "bank-full.csv"
data = pd.read_csv(filename, delimiter=';')
print(data.head())
data.describe(include='all')


# # data clean
# 

# In[3]:


data.info(), data.head()

missing_values = data.isnull().sum()

unknown_values = (data == "unknown").sum()

duplicates = data.duplicated().sum()

if duplicates > 0:
    data = data.drop_duplicates()


# In[4]:


missing_values


# In[5]:


unknown_values


# In[6]:


duplicates


# # data visualization

# In[8]:


plt.figure(figsize=(20, 15))
numer_cols_ex_days = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

for i, col in enumerate(numer_cols_ex_days , 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f'Histogram of {col}')

plt.tight_layout()
plt.show()


# In[9]:


numer_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

numer_varis = data[numer_cols]

numer_corr_mat = numer_varis.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(numer_corr_mat, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix for Numerical Variables')
plt.show()


# In[10]:


numer_corr_mat


# # label encoding

# In[22]:


def prepare_and_split_dataset(features, target, test_size=0.3, random_state=42):
    
    dataset = np.column_stack((features, target))
    
    train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=random_state, stratify=target)

    X_train = train_set[:, :-1]
    y_train = train_set[:, -1]

    X_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    
    return (X_train, y_train), (X_test, y_test)


# In[23]:


def label_encoding_without_y(df):
    df_copy_wo_y = df.copy()
    le = LabelEncoder()
    
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['age', 'balance', 'duration', 'y']:  
            df_copy_wo_y[col] = le.fit_transform(df_copy_wo_y[col].astype(str))
    
    return df_copy_wo_y


# In[24]:


def label_encoding(df):
    df_copy = df.copy()
    le = LabelEncoder()
    
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['age', 'balance', 'duration']:  
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))  
            
    return df_copy


# In[25]:


def score_calculation(pd, te):

    if len(pd) != len(te):
        
        accuracy_score = "Lengths Error"
        
    else:

        matches = sum([1 for pd,te in zip(pd, te) if pd == te])

    accuracy_score = matches / len(pd)
    
    return accuracy_score


# In[28]:


label_encoded_data_ex_y = label_encoding_without_y(data)
label_encoded_data = label_encoding(data)

print(label_encoded_data_ex_y.head())
print(label_encoded_data)


# In[29]:


X = label_encoded_data_ex_y.drop('y', axis=1) 
y = label_encoded_data_ex_y['y']

lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')

y_pred = lof.fit_predict(X)

label_encoded_data_ex_y['Outlier'] = y_pred

outliers = label_encoded_data_ex_y[label_encoded_data_ex_y['Outlier'] == -1]
inliers = label_encoded_data_ex_y[label_encoded_data_ex_y['Outlier'] == 1]

print(len(outliers))


# In[30]:


X = label_encoded_data  
y = data['y']  

pca = PCA()

X_pca = pca.fit_transform(X)

X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i}' for i in range(1, X_pca.shape[1] + 1)])


# In[31]:


print(X.head())


# In[32]:


cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)


# In[33]:


print(cumulative_explained_variance)


# In[34]:


plt.figure(figsize=(10,6))
plt.plot(range(1, len(cumulative_explained_variance)+1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Differnet Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance Level')
plt.legend(loc='best')
plt.show()


# # one-hot encoding

# In[34]:


categorical_columns = data.select_dtypes(include=['object']).columns

categorical_columns = categorical_columns.drop('y')

data_oh_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

print(data_oh_encoded.head())


# In[24]:


# print(categorical_columns)


# In[35]:


features = data_oh_encoded.drop('y', axis=1)
target = data_oh_encoded['y']

scaler = StandardScaler()
oh_feature = scaler.fit_transform(features)

oh_df = pd.DataFrame(oh_feature, columns=features.columns)


# In[37]:


oh_df.head()


# In[38]:


pca = PCA()

features_pca = pca.fit_transform(oh_feature)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(14, 7))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Different Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.75, color='r', linestyle='--', label='75% Explained Variance Level')
plt.legend(loc='best')
plt.axhline(y=0.50, color='g', linestyle='--', label='50% Explained Variance Level')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[39]:


explained_variance_ratio


# In[40]:


cumulative_explained_variance


# In[41]:


def my_pca(n, feature):
    pca = PCA(n_components=n)
    pca_df = pca.fit_transform(feature)
    return pca_df


# In[33]:


oh_pca_13 = my_pca(13, oh_feature)

oh_pca_23 = my_pca(23, oh_feature)


# In[34]:


oh_pca_13_df = pd.DataFrame(oh_pca_13, columns=[f'PC{i}' for i in range(1, 14)])
oh_pca_23_df = pd.DataFrame(oh_pca_23, columns=[f'PC{i}' for i in range(1, 24)])


# In[35]:


oh_pca_13_df.head()


# In[36]:


oh_pca_23_df.head()


# In[38]:


(original_train, original_test) = prepare_and_split_dataset(oh_feature, target)

(pca_13_train, pca_13_test) = prepare_and_split_dataset(oh_pca_13, target)

(pca_23_train, pca_23_test) = prepare_and_split_dataset(oh_pca_23, target)


# In[40]:


label_id = label_encoded_data_ex_y.drop('y', axis=1)

target_ld = label_encoded_data_ex_y['y']

(ld_train, ld_test) = prepare_and_split_dataset(label_id, target_ld)

# print(ld_train, ld_test)


# In[41]:


# (ld_train[0].shape, ld_test[0].shape)


# In[42]:


# (original_train[0].shape, original_test[0].shape), (pca_13_train[0].shape, pca_13_test[0].shape), (pca_23_train[0].shape, pca_23_test[0].shape)


# # svm

# In[43]:


def my_svm(X_train, y_train, X_test, y_test):

    svm_model = SVC(kernel='linear', probability=True) 

    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)

    y_prob = svm_model.predict_proba(X_test)[:, 1]

    return y_test, y_pred, y_prob 


# In[44]:


# svm one-hot original
y_te_oh_ori, y_pre_oh_ori, y_pro_oh_ori = my_svm(original_train[0], original_train[1], original_test[0], original_test[1])


# In[45]:


acc_svm_oh_origi = score_calculation(y_pre_oh_ori, y_te_oh_ori)
print(acc_svm_oh_origi)


# In[46]:


# svm one-hot pca13
y_te_oh_13, y_pre_oh_13, y_pro_oh_13 = my_svm(pca_13_train[0], pca_13_train[1], pca_13_test[0], pca_13_test[1])


# In[47]:


acc_svm_oh_13 = score_calculation(y_pre_oh_13, y_te_oh_13)
print(acc_svm_oh_13)


# In[48]:


# svm one-hot pca23
y_te_oh_23, y_pre_oh_23, y_pro_oh_23 = my_svm(pca_23_train[0], pca_23_train[1], pca_23_test[0], pca_23_test[1])


# In[49]:


acc_svm_oh_23 = score_calculation(y_pre_oh_23, y_te_oh_23)
print(acc_svm_oh_23)


# In[50]:


# print(original_train[0], original_train[1], original_test[0], original_test[1])


# In[95]:


# print(ld_train[0], ld_train[1], ld_test[0], ld_test[1])


# In[24]:


# svm lable encoding original
y_te_ld_ori, y_pre_ld_ori, y_pro_ld_ori = my_svm(ld_train[0], ld_train[1], ld_test[0], ld_test[1])


# In[25]:


acc_svm_ld_origi = score_calculation(y_pre_ld_ori, y_te_ld_ori)
print(acc_svm_ld_origi)


# # logistic regression

# In[27]:


def my_logistic_regression(X_train, y_train, X_test, y_test):

    lr_model = LogisticRegression(max_iter=1000)  

    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    y_prob = lr_model.predict_proba(X_test)[:, 1]
    
    return y_test, y_pred, y_prob 


# In[55]:


# lr one-hot original
y_te_oh_ori_lr, y_pre_oh_ori_lr, y_pro_oh_ori_lr = my_logistic_regression(original_train[0], original_train[1], original_test[0], original_test[1])


# In[60]:


acc_lr_oh_ori = score_calculation(y_pre_oh_ori_lr, y_te_oh_ori_lr)
print(acc_lr_oh_ori)


# In[61]:


# lr one-hot pca13
y_te_oh_13_lr, y_pre_oh_13_lr, y_pro_oh_13_lr = my_logistic_regression(pca_13_train[0], pca_13_train[1], pca_13_test[0], pca_13_test[1])


# In[62]:


acc_lr_oh_13 = score_calculation(y_pre_oh_13_lr, y_te_oh_13_lr)
print(acc_lr_oh_13)


# In[66]:


# lr one-hot pca23
y_te_oh_23_lr, y_pre_oh_23_lr, y_pro_oh_23_lr = my_logistic_regression(pca_23_train[0], pca_23_train[1], pca_23_test[0], pca_23_test[1])


# In[68]:


acc_lr_oh_23 = score_calculation(y_pre_oh_23_lr, y_te_oh_23_lr)
print(acc_lr_oh_23)


# In[32]:


# lr label decoding original
y_te_ld_ori_lr, y_pre_ld_ori_lr, y_pro_ld_ori_lr = my_logistic_regression(ld_train[0], ld_train[1], ld_test[0], ld_test[1])


# In[33]:


acc_lr_ld_ori = score_calculation(y_pre_ld_ori_lr, y_te_ld_ori_lr)
print(acc_lr_ld_ori)


# # Random Forest

# In[28]:


def my_random_forest(X_train, y_train, X_test, y_test):
    
    rf_model = RandomForestClassifier(random_state=42)
    
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    return y_test, y_pred, y_prob 


# In[70]:


# rf one-hot original
y_te_oh_ori_rf, y_pre_oh_ori_rf, y_pro_oh_ori_rf = my_random_forest(original_train[0], original_train[1], original_test[0], original_test[1])


# In[71]:


acc_rf_oh_ori = score_calculation(y_pre_oh_ori_rf, y_te_oh_ori_rf)
print(acc_rf_oh_ori)


# In[72]:


# rf one-hot pca13
y_te_oh_13_rf, y_pre_oh_13_rf, y_pro_oh_13_rf = my_random_forest(pca_13_train[0], pca_13_train[1], pca_13_test[0], pca_13_test[1])


# In[73]:


acc_rf_oh_13 = score_calculation(y_pre_oh_13_rf, y_te_oh_13_rf)
print(acc_rf_oh_13)


# In[74]:


# rf one-hot pca23
y_te_oh_23_rf, y_pre_oh_23_rf, y_pro_oh_23_rf = my_random_forest(pca_23_train[0], pca_23_train[1], pca_23_test[0], pca_23_test[1])


# In[75]:


acc_rf_oh_23 = score_calculation(y_pre_oh_23_rf, y_te_oh_23_rf)
print(acc_rf_oh_23)


# In[34]:


# rf label decoding original
y_te_ld_ori_rf, y_pre_ld_ori_rf, y_pro_ld_ori_rf = my_random_forest(ld_train[0], ld_train[1], ld_test[0], ld_test[1])


# In[35]:


acc_rf_ld_ori = score_calculation(y_pre_ld_ori_rf, y_te_ld_ori_rf)
print(acc_rf_ld_ori)


# In[ ]:




