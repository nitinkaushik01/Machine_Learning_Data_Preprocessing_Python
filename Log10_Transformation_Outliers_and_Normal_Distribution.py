
# coding: utf-8

# # Log Transformation for Positive or Right Skewed Data

# In[46]:

# Import pandas library 
import pandas as pd
import numpy as np
import seaborn as sns

# create a list of data 
data = [1,1,10,10,15,15,20,20,30,50,120,130,120,50,30,30,25,20,20,15,15,13,11,9,7,6,6,5,5,5,4,4,4,4,3,3,3,3,2,2,2,2,2,1,1,1,1,1,1,
1] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Positive Skewed']) 
  
# print dataframe. 
df 


# In[47]:

#Boxplot showing three outliers
df.boxplot(column='Positive Skewed')
plt.show()


# In[48]:

#Right Skewed data
sns.distplot(df)


# In[49]:

#Creating input data from dataframe df on variable Positive Skewness with input values ranging from 1 to 130
inp_array = df 
print ("Input array : ", inp_array)


# In[50]:

#Applying log10 transformation with output values ranging from 0 to 2+
out_array = np.log10(inp_array) 
print ("Output array : ", out_array)


# In[51]:

#Boxplot showing No outliers with all of them treated by doing log10 transformation.
out_array.boxplot(column='Positive Skewed')
plt.show()


# In[52]:

#Right Skewed data transformed to Fairly or close to Normal Distribution using Log10 transformations
sns.distplot(out_array)


# In[53]:

#If wants to revert back log10 values to original value for interpretation purpose then just raise 10 to the power 
#log10 values as shown below.
original_val = (10**out_array) 
print ("Original Values : ", original_val)


# # Square Root Transformation for Positive or Right Skewed Data

# In[54]:

# Import pandas library 
import pandas as pd
import numpy as np
import seaborn as sns

# Create a list of data. Here, we have included zeros as well in the data
data = [0,0,1,1,10,10,15,15,20,20,30,50,120,130,120,50,30,30,25,20,20,15,15,13,11,9,7,6,6,5,5,5,4,4,4,4,3,3,3,3,2,2,2,2,2,1,1,1,1,1,1,
1,0,0] 
  
# Create the pandas DataFrame 
df2 = pd.DataFrame(data, columns = ['Positive Skewed']) 
  
# print dataframe. 
df2 


# In[55]:

#Boxplot showing three outliers
df2.boxplot(column='Positive Skewed')
plt.show()


# In[57]:

#Right Skewed data 
sns.distplot(df2)


# In[58]:

#Creating input data from dataframe df on variable Positive Skewness with input values ranging from 0 to 130
inp_array2 = df2 
print ("Input array : ", inp_array2)


# In[59]:

#Applying Square Root transformation with output values ranging from 0 to 11+
out_array2 = np.sqrt(inp_array2) 
print ("Output array : ", out_array2)


# In[60]:

#Boxplot showing only Two outliers now, with one of those treated by doing Square Root transformation.
out_array2.boxplot(column='Positive Skewed')
plt.show()


# In[61]:

#Right Skewed data transformed to Fairly or close to Normal Distribution using Square Root transformations, though not perfect
#Normal Distribution since this type of transformation has moderate effect on distribution shape
sns.distplot(out_array2)


# In[ ]:



