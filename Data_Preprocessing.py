
# coding: utf-8

# # Python Libraries

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp


# # Change Working Directory

# In[2]:

import os
os.chdir('F:')


# # Read .csv file

# In[3]:

datafile = pd.read_csv('MallCustomers.csv')
datafile.head()


# In[4]:

datafile.shape


# In[5]:

X = datafile.iloc[:,:-1].values


# In[6]:

Y = datafile.iloc[:,4]


# # Missing Value Detection and Imputation

# In[26]:

data = pd.read_csv('Sample_real_estate_data.csv')
print(data['ST_NUM'].isnull())


# In[27]:

print(data['NUM_BEDROOMS'].isnull())


# In[71]:

missing_value = ["n/a","na","--"]
data1 = pd.read_csv('Sample_real_estate_data.csv', na_values = missing_value)
data = data1


# In[29]:

print(data['NUM_BEDROOMS'].isnull())


# In[30]:

print(data['OWN_OCCUPIED'].isnull())


# In[31]:

count = 0
for row in data['OWN_OCCUPIED']:
    try:
        int(row)
        data.loc[count, 'OWN_OCCUPIED '] = np.nan
    except ValueError:
        pass
    count+=1


# In[39]:

print(data['OWN_OCCUPIED'].isnull())


# In[40]:

print(data.isnull().sum())


# In[45]:

print(data.isnull().values.any())


# In[61]:

from sklearn.preprocessing import Imputer

X = data.iloc[:,:-1].values
Y = data.iloc[:,6]
imput = Imputer(missing_values = 'NaN', strategy= 'mean', axis=0)
imput = imput.fit(X[:,1:2])
X[:,1:2] = imput.transform(X[:,1:2])
X[:,1:2]


# In[62]:

data


# In[72]:

median = data['NUM_BEDROOMS'].median()
data['NUM_BEDROOMS'].fillna(median, inplace=True)
data


# # Categorical Variable Encoding

# In[145]:

data = pd.read_csv('MallCustomers.csv')
data.head()


# In[146]:

X = data.iloc[:,:-1].values
Y = data.iloc[:,4].values


# In[147]:

from sklearn.preprocessing import LabelEncoder
lblencode = LabelEncoder()
X[:,1] = lblencode.fit_transform(X[:,1])
X


# ## One Hot Encoding

# In[148]:

from sklearn.preprocessing import OneHotEncoder
onehotencod = OneHotEncoder(categorical_features = [1])


# In[149]:

X = onehotencod.fit_transform(X).toarray()
X


# # Train Test Split

# In[9]:

dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,8].values
dataset.head()


# In[10]:

np.set_printoptions(edgeitems=127)


# In[11]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# # Feature Scaling

# In[ ]:

from sklearn.preprocessing import StandardScaler
stdscalar = StandardScaler()
X_train = stdscalar.fit_transform(X_train)
X_test = stdscalar.transform(X_test)


# # Outlier Detection

# In[23]:

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
print(boston.feature_names)


# In[24]:

boston = pd.DataFrame(boston.data)
boston.head()


# ## Outlier Detection through Boxplot

# In[26]:

get_ipython().magic('matplotlib inline')
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x=boston[7])


# In[27]:

# Outlier Detection Part 2
boston_c = boston


# ## Outlier Detection through Scatter Plot

# In[29]:

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(boston_c[2], boston_c[9])
ax.set_xlabel('proportion of non-retail business acres per town')
ax.set_ylabel('full-value property-tax rate per $10,000')
plt.show()


# ## Outlier Detection through Mathematical Method (Z-Score)

# In[31]:

from scipy import stats
zscore = np.abs(stats.zscore(boston_c))
print(zscore)


# In[32]:

threshold = 3
print(np.where(zscore > 3))


# In[33]:

print(zscore[102][11])


# ## Outlier Detection through Mathematical Method (Inter Quartile Range)

# In[35]:

boston_iqr = boston
Q1 = boston_iqr.quantile(0.25)
Q3 = boston_iqr.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[40]:

print(boston_iqr < (Q1 - 1.5 * IQR)) | (boston_iqr > (Q3 + 1.5 * IQR))


# ## Handle Outliers/Correct Outliers

# In[41]:

boston_clean = boston
boston_clean = boston_clean[(zscore < 3).all(axis=1)]


# In[42]:

boston.shape


# In[43]:

boston_clean.shape


# In[46]:

#Remove Outliers using IQR
boston_iqr_clean = boston_iqr[~((boston_iqr < (Q1 - 1.5 * IQR)) | (boston_iqr > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



