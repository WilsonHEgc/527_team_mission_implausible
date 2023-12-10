#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pycaret.classification import *

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture
import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from imblearn.combine import SMOTETomek
from collections import Counter


# In[2]:


# Load the dataset
filename = '/Users/russelwilson/Desktop/bank-full.csv'
df = pd.read_csv(filename, delimiter=';')
df_original = pd.read_csv(filename, delimiter=';')
print(df.head())
df.describe(include='all')


# In[3]:


# Data Preprocessing







# In[4]:


# Data Cleaning
missing_values = df.isnull().sum()

unknown_values = (df == 'unknown').sum()

# nan_values = (df == 'NaN').sum()

duplicates = df.duplicated().sum()

# print(missing_values)
# print(unknown_values)
# print(nan_values)
# print(duplicates)


# In[5]:


categorical_columns = df.select_dtypes(include=['object']).columns


# In[6]:


unique_values_info = {}
for col in categorical_columns:
    unique_counts = df[col].value_counts()
    unique_values_info[col] = unique_counts


# In[7]:


# unique_values_info


# In[8]:


# Unique encoding









# In[9]:


df_uni_encoding = pd.DataFrame()


# In[10]:


# Label encoding for 'education'
education_mapping = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
df_uni_encoding['education_encoded'] = df['education'].map(education_mapping)


# In[11]:


# Label encoding for 'default', 'housing', 'loan', 'y' 
binary_mapping = {'no': 0, 'yes': 1}
columns_to_encode = ['default', 'housing', 'loan', 'y']

for col in columns_to_encode:
    df_uni_encoding[col + '_encoded'] = df[col].map(binary_mapping)


# In[12]:


df['poutcome'] = df['poutcome'].replace(['unknown', 'other'], 'others')

# Label encoding for 'poutcome' with the specified mapping
poutcome_mapping = {'failure': 0, 'success': 1, 'others': -1}
df_uni_encoding['poutcome_encoded'] = df['poutcome'].map(poutcome_mapping)


# In[13]:


# Frequency encoding for 'job' and 'month'
job_freq = df['job'].value_counts(normalize=True)
month_freq = df['month'].value_counts(normalize=True)
contact_freq = df['contact'].value_counts(normalize=True)
marital_freq = df['marital'].value_counts(normalize=True)

df_uni_encoding['job_encoded'] = df['job'].map(job_freq)
df_uni_encoding['month_encoded'] = df['month'].map(month_freq)
df_uni_encoding['contact_encoded'] = df['contact'].map(contact_freq)
df_uni_encoding['marital_encoded'] = df['marital'].map(marital_freq)


# In[14]:


df_uni_encoding.head()


# In[15]:


encoded_columns = [
    'education_encoded', 'default_encoded', 'housing_encoded', 'loan_encoded', 
    'poutcome_encoded', 'job_encoded', 'month_encoded', 
    'contact_encoded', 'marital_encoded', 'y_encoded'
]

numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

df_numer = df[numeric_columns]

df_numer.columns.tolist()


# In[16]:


df_uni = pd.concat([df_numer, df_uni_encoding], axis = 1)


# In[17]:


df_uni.head()


# In[18]:


df_uni.shape


# In[19]:


df_uni_ori = df_uni.copy()


# In[20]:


def score_calculation(pd, te):

    if len(pd) != len(te):
        
        accuracy_score = "Lengths Error"
        
    else:

        matches = sum([1 for pd,te in zip(pd, te) if pd == te])

    accuracy_score = matches / len(pd)
    
    return accuracy_score


# In[21]:


def calculate_mcc(y_true, y_pred):

    mcc = matthews_corrcoef(y_true, y_pred)
    
    return mcc


# In[22]:


# one-hot encoding









# In[23]:


df_without_y = df.drop("y", axis=1)
df_without_y.head()


# In[24]:


categorical_columns_without_y = categorical_columns.drop("y")


# In[25]:


categorical_columns_without_y


# In[26]:


df_oh = pd.get_dummies(df_without_y, columns=categorical_columns_without_y, drop_first=True)

df_oh.head()


# In[27]:


df_oh.shape


# In[28]:


df_oh = pd.concat([df_oh, df_uni['y_encoded']], axis = 1)


# In[29]:


df_oh.head()


# In[30]:


df_oh_ori = df_oh.copy()


# In[31]:


# Outlier Detection for Unique Encoding







# In[32]:


x_out = df_uni.copy()

x_out.head()


# In[33]:


df_uni.head()


# In[34]:


# Applying IQR
Q1 = x_out.quantile(0.25)
Q3 = x_out.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((x_out < (Q1 - 1.5 * IQR)) | (x_out > (Q3 + 1.5 * IQR))).any(axis=1)

# Applying Isolation Forest
iso_forest = IsolationForest(random_state=527)
outliers_iso_forest = iso_forest.fit_predict(x_out) == -1

# Applying Local Outlier Factor
lof = LocalOutlierFactor()
outliers_lof = lof.fit_predict(x_out) == -1

# Counting the number of outliers
outliers_count = {
    "IQR": np.sum(outliers_iqr),
    "Isolation Forest": np.sum(outliers_iso_forest),
    "Local Outlier Factor": np.sum(outliers_lof)
}


# In[35]:


outliers_count


# In[36]:


# Encoded dataset without IQR detected outliers
df_uni_without_iqr_outliers = df_uni[~outliers_iqr]

# Encoded dataset without Isolation Forest detected outliers
df_uni_without_iso_forest_outliers = df_uni[~outliers_iso_forest]

# Encoded dataset without LOF detected outliers
df_uni_without_lof_outliers = df_uni[~outliers_lof]

datasets_shapes = {
    "Original Encoded Dataset": df_uni_ori.shape,
    "Without IQR Outliers": df_uni_without_iqr_outliers.shape,
    "Without Isolation Forest Outliers": df_uni_without_iso_forest_outliers.shape,
    "Without LOF Outliers": df_uni_without_lof_outliers.shape
}


# In[37]:


datasets_shapes


# In[38]:


# Outlier Detection for One-Hot Encoding








# In[39]:


x_out = df_oh.copy()

x_out.head()


# In[40]:


df_oh.head()


# In[41]:


# Applying IQR
Q1 = x_out.quantile(0.25)
Q3 = x_out.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((x_out < (Q1 - 1.5 * IQR)) | (x_out > (Q3 + 1.5 * IQR))).any(axis=1)

# Applying Isolation Forest
iso_forest = IsolationForest(random_state=527)
outliers_iso_forest = iso_forest.fit_predict(x_out) == -1

# Applying Local Outlier Factor
lof = LocalOutlierFactor()
outliers_lof = lof.fit_predict(x_out) == -1

# Counting the number of outliers
outliers_count = {
    "IQR": np.sum(outliers_iqr),
    "Isolation Forest": np.sum(outliers_iso_forest),
    "Local Outlier Factor": np.sum(outliers_lof)
}


# In[42]:


outliers_count


# In[43]:


# Encoded dataset without IQR detected outliers
df_oh_without_iqr_outliers = df_oh[~outliers_iqr]

# Encoded dataset without Isolation Forest detected outliers
df_oh_without_iso_forest_outliers = df_oh[~outliers_iso_forest]

# Encoded dataset without LOF detected outliers
df_oh_without_lof_outliers = df_oh[~outliers_lof]

datasets_shapes = {
    "Original Encoded Dataset": df_oh_ori.shape,
    "Without IQR Outliers": df_oh_without_iqr_outliers.shape,
    "Without Isolation Forest Outliers": df_oh_without_iso_forest_outliers.shape,
    "Without LOF Outliers": df_oh_without_lof_outliers.shape
}


# In[44]:


datasets_shapes


# In[45]:


# Unique Encoding Original Dataset








# In[46]:


clf = setup(df_uni_ori, target = "y_encoded")


# In[47]:


best_model = compare_models()


# In[48]:


best_model


# In[49]:


# uni encoded data
X = df_uni_ori.drop('y_encoded', axis=1)
y = df_uni_ori['y_encoded']


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=527)


# In[51]:


sgd = SGDClassifier(random_state=527)

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_accuracy = sgd.score(X_test, y_test)

print(f"Accuracy: {sgd_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[52]:


sgd_rbf = make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd_rbf.fit(X_train, y_train)

y_pred = sgd_rbf.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_rbf_accuracy = sgd_rbf.score(X_test, y_test)

print(f"Accuracy: {sgd_rbf_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[ ]:





# In[55]:


model = LGBMClassifier()
gridParams = {
    'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.02, 0.1],
    'n_estimators': [40, 80, 100, 200, 400],
    'num_leaves': [20, 30, 40, 50, 70, 100],
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

print(f"Mean Best Score: {mean_best_score:.4f} ± {std_best_score:.4f}")
print("Best parameters per fold:", best_params)


# In[56]:


optim_lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['l2', 'auc'],
    'num_leaves': 30,  
    'learning_rate': 0.1,  
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'n_estimators': 80 
}

# lgbm_model = LGBMClassifier(learning_rate=0.1, n_estimators=80, num_leaves=30)

# lgbm_model.fit(X_train, y_train)

# y_pred = lgbm_model.predict(X_test)

# optim_accuracy = accuracy_score(y_test, y_pred)

# optim_accuracy


# In[57]:


lgbm_model = lgb.train(
    optim_lgbm_params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)


# In[58]:


y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

lgbm_accuracy = accuracy_score(y_test, y_pred_binary)
lgbm_accuracy


# In[ ]:


model = RandomForestClassifier()

gridParams = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


model = HistGradientBoostingClassifier()

# Grid parameters for HistGradientBoostingClassifier
gridParams = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [50, 100, 200],
    'max_leaf_nodes': [20, 30, 40],
    'min_samples_leaf': [10, 20, 30]
}

# Inner and outer cross-validation settings
inner_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing

best_scores = []
best_params = []

# Nested cross-validation
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


# Unique Encoding without IF Outlier Dataset








# In[53]:


clf2 = setup(df_uni_without_iso_forest_outliers, target = "y_encoded")


# In[54]:


best_model2 = compare_models()


# In[55]:


best_model2


# In[56]:


# uni data without isolation forest
X = df_uni_without_iso_forest_outliers.drop('y_encoded', axis=1)
y = df_uni_without_iso_forest_outliers['y_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=527)


# In[57]:


sgd = SGDClassifier(random_state=527)

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_accuracy = sgd.score(X_test, y_test)

print(f"Accuracy: {sgd_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[58]:


sgd_rbf = make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd_rbf.fit(X_train, y_train)

y_pred = sgd_rbf.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_rbf_accuracy = sgd_rbf.score(X_test, y_test)

print(f"Accuracy: {sgd_rbf_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[ ]:





# In[ ]:


model = LGBMClassifier()
gridParams = {
    'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.02, 0.1],
    'n_estimators': [40, 80, 100, 200, 400],
    'num_leaves': [20, 30, 40, 50, 70, 100],
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

print(f"Mean Best Score: {mean_best_score:.4f} ± {std_best_score:.4f}")
print("Best parameters per fold:", best_params)


# In[ ]:


optim_lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['l2', 'auc'],
    'num_leaves': 30,  
    'learning_rate': 0.1,  
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'n_estimators': 80 
}

# lgbm_model = LGBMClassifier(learning_rate=0.1, n_estimators=80, num_leaves=30)

# lgbm_model.fit(X_train, y_train)

# y_pred = lgbm_model.predict(X_test)

# optim_accuracy = accuracy_score(y_test, y_pred)

# optim_accuracy


# In[ ]:


lgbm_model = lgb.train(
    optim_lgbm_params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)


# In[ ]:


y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

lgbm_accuracy = accuracy_score(y_test, y_pred_binary)
lgbm_accuracy


# In[ ]:


model = RandomForestClassifier()

gridParams = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


model = HistGradientBoostingClassifier()

# Grid parameters for HistGradientBoostingClassifier
gridParams = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [50, 100, 200],
    'max_leaf_nodes': [20, 30, 40],
    'min_samples_leaf': [10, 20, 30]
}

# Inner and outer cross-validation settings
inner_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing

best_scores = []
best_params = []

# Nested cross-validation
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


# Unique Encoding without LOF Outlier Dataset








# In[59]:


clf3 = setup(df_uni_without_lof_outliers, target = 'y_encoded')


# In[60]:


best_model3 = compare_models()


# In[61]:


best_model3


# In[62]:


# uni data without lof
X = df_uni_without_lof_outliers.drop('y_encoded', axis=1)
y = df_uni_without_lof_outliers['y_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=527)


# In[63]:


sgd= make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_accuracy = sgd.score(X_test, y_test)

print(f"Accuracy: {sgd_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[64]:


sgd_rbf = make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd_rbf.fit(X_train, y_train)

y_pred = sgd_rbf.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_rbf_accuracy = sgd_rbf.score(X_test, y_test)

print(f"Accuracy: {sgd_rbf_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[ ]:





# In[109]:


model = LGBMClassifier()
gridParams = {
    'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.02, 0.1],
    'n_estimators': [40, 80, 100, 200, 400],
    'num_leaves': [20, 30, 40, 50, 70, 100],
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

print(f"Mean Best Score: {mean_best_score:.4f} ± {std_best_score:.4f}")
print("Best parameters per fold:", best_params)


# In[112]:


model = LGBMClassifier()
gridParams = {
    'learning_rate': [0.02, 0.1],
    'n_estimators': [80, 200, 400],
    'num_leaves': [20, 30, 40],
}

inner_cv = KFold(n_splits=3, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=8, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

print(f"Mean Best Score: {mean_best_score:.4f} ± {std_best_score:.4f}")
print("Best parameters per fold:", best_params)


# In[113]:


model = LGBMClassifier()
gridParams = {
    'learning_rate': [0.02, 0.1],
    'n_estimators': [80, 200],
    'num_leaves': [20, 30],
}

inner_cv = KFold(n_splits=3, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

print(f"Mean Best Score: {mean_best_score:.4f} ± {std_best_score:.4f}")
print("Best parameters per fold:", best_params)


# In[117]:


optim_lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['l2', 'auc'],
    'num_leaves': 30,  
    'learning_rate': 0.1,  
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'n_estimators': 80 
}

# lgbm_model = LGBMClassifier(learning_rate=0.1, n_estimators=80, num_leaves=30)

# lgbm_model.fit(X_train, y_train)

# y_pred = lgbm_model.predict(X_test)

# optim_accuracy = accuracy_score(y_test, y_pred)

# optim_accuracy


# In[118]:


lgbm_model = lgb.train(
    optim_lgbm_params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)


# In[119]:


y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

lgbm_accuracy = accuracy_score(y_test, y_pred_binary)
lgbm_accuracy


# In[ ]:


model = RandomForestClassifier()

gridParams = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


model = HistGradientBoostingClassifier()

# Grid parameters for HistGradientBoostingClassifier
gridParams = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [50, 100, 200],
    'max_leaf_nodes': [20, 30, 40],
    'min_samples_leaf': [10, 20, 30]
}

# Inner and outer cross-validation settings
inner_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing

best_scores = []
best_params = []

# Nested cross-validation
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


# One-Hot Encoding Original Dataset








# In[65]:


clf4 = setup(df_oh_ori, target = 'y_encoded')


# In[66]:


best_model4 = compare_models()


# In[67]:


best_model4


# In[68]:


# uni data without lof
X = df_oh_ori.drop('y_encoded', axis=1)
y = df_oh_ori['y_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=527)


# In[69]:


sgd= make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_accuracy = sgd.score(X_test, y_test)

print(f"Accuracy: {sgd_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[70]:


sgd_rbf = make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd_rbf.fit(X_train, y_train)

y_pred = sgd_rbf.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_rbf_accuracy = sgd_rbf.score(X_test, y_test)

print(f"Accuracy: {sgd_rbf_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[ ]:





# In[ ]:


model = LGBMClassifier()
gridParams = {
    'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.02, 0.1],
    'n_estimators': [40, 80, 100, 200, 400],
    'num_leaves': [20, 30, 40, 50, 70, 100],
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

print(f"Mean Best Score: {mean_best_score:.4f} ± {std_best_score:.4f}")
print("Best parameters per fold:", best_params)


# In[ ]:


model = RandomForestClassifier()

gridParams = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


model = HistGradientBoostingClassifier()

# Grid parameters for HistGradientBoostingClassifier
gridParams = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [50, 100, 200],
    'max_leaf_nodes': [20, 30, 40],
    'min_samples_leaf': [10, 20, 30]
}

# Inner and outer cross-validation settings
inner_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing

best_scores = []
best_params = []

# Nested cross-validation
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


# One-Hot Encoding without LOF Outlier Dataset










# In[71]:


clf5 = setup(df_oh_without_lof_outliers, target = 'y_encoded')


# In[72]:


best_model5 = compare_models()


# In[73]:


best_model5


# In[74]:


# uni data without lof
X = df_oh_without_lof_outliers.drop('y_encoded', axis=1)
y = df_oh_without_lof_outliers['y_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=527)


# In[75]:


sgd= make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_accuracy = sgd.score(X_test, y_test)

print(f"Accuracy: {sgd_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[76]:


sgd_rbf = make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd_rbf.fit(X_train, y_train)

y_pred = sgd_rbf.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_rbf_accuracy = sgd_rbf.score(X_test, y_test)

print(f"Accuracy: {sgd_rbf_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[ ]:





# In[ ]:


model = LGBMClassifier()
gridParams = {
    'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.02, 0.1],
    'n_estimators': [40, 80, 100, 200, 400],
    'num_leaves': [20, 30, 40, 50, 70, 100],
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

print(f"Mean Best Score: {mean_best_score:.4f} ± {std_best_score:.4f}")
print("Best parameters per fold:", best_params)


# In[ ]:


model = RandomForestClassifier()

gridParams = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


model = HistGradientBoostingClassifier()

# Grid parameters for HistGradientBoostingClassifier
gridParams = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [50, 100, 200],
    'max_leaf_nodes': [20, 30, 40],
    'min_samples_leaf': [10, 20, 30]
}

# Inner and outer cross-validation settings
inner_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing

best_scores = []
best_params = []

# Nested cross-validation
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


# One-Hot Encoding without IF Outlier Dataset









# In[77]:


clf6 = setup(df_oh_without_iso_forest_outliers, target = 'y_encoded')


# In[78]:


best_model6 = compare_models()


# In[79]:


best_model6


# In[80]:


# uni data without if
X = df_uni_without_iso_forest_outliers.drop('y_encoded', axis=1)
y = df_uni_without_iso_forest_outliers['y_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=527)


# In[81]:


sgd= make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_accuracy = sgd.score(X_test, y_test)

print(f"Accuracy: {sgd_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[82]:


sgd_rbf = make_pipeline(RBFSampler(gamma=1, random_state=527), SGDClassifier(random_state=527))

sgd_rbf.fit(X_train, y_train)

y_pred = sgd_rbf.predict(X_test)
mcc = calculate_mcc(y_test, y_pred)

sgd_rbf_accuracy = sgd_rbf.score(X_test, y_test)

print(f"Accuracy: {sgd_rbf_accuracy}")
print(f"Matthews Correlation Coefficient: {mcc}")


# In[ ]:





# In[ ]:


model = LGBMClassifier()
gridParams = {
    'learning_rate': [0.001, 0.002, 0.005, 0.01, 0.02, 0.1],
    'n_estimators': [40, 80, 100, 200, 400],
    'num_leaves': [20, 30, 40, 50, 70, 100],
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

print(f"Mean Best Score: {mean_best_score:.4f} ± {std_best_score:.4f}")
print("Best parameters per fold:", best_params)


# In[ ]:


model = RandomForestClassifier()

gridParams = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=527)
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=527)

best_scores = []
best_params = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:


model = HistGradientBoostingClassifier()

# Grid parameters for HistGradientBoostingClassifier
gridParams = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [50, 100, 200],
    'max_leaf_nodes': [20, 30, 40],
    'min_samples_leaf': [10, 20, 30]
}

# Inner and outer cross-validation settings
inner_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing
grid = GridSearchCV(model, gridParams, cv=inner_cv, n_jobs=-1)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=527)  # Reduced number of splits for quicker processing

best_scores = []
best_params = []

# Nested cross-validation
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid.fit(X_train, y_train)
    best_scores.append(grid.best_score_)
    best_params.append(grid.best_params_)

# Aggregate and print the results
mean_best_score = np.mean(best_scores)
std_best_score = np.std(best_scores)

mean_best_score, std_best_score, best_params


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




