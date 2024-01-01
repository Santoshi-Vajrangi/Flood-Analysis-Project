#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

team = "A2 - TEAM 4"
print("FEMA's NATIONAL FLOOD INSURANCE POLICY ")
print(team.center(30))

data={'Roll_No' : pd.Series([148, 164, 159, 169, 150], index = ['1','2','3','4','5']),
 'Names' : pd.Series(["Santoshi", "Deepa", "Nidhi","Sneha","Mahima"], index = ['1','2','3','4','5']),
 'USN' :pd.Series(['01fe21bcs298','01fe21bcs297','01fe21bcs331','01fe21bcs362','01fe21bcs301'],index=['1','2','3','4','5'])}
df = pd.DataFrame(data)
df


# In[5]:


import numpy as np
import pandas as pd
data = pd.read_csv('C:\\Users\\Dell\\Desktop\\flood1.csv')
data 


# In[38]:


df_imputed_mean = df.fillna(df.mean())
print("DataFrame after imputing missing values with mean:")
print(df_imputed_mean)


# In[ ]:


df = pd.DataFrame(data)
d1 = df.fillna(df.mean())
d1


# In[ ]:





# In[34]:



data.dropna(subset=['cancellationdateoffloodpolicy'],inplace=True)


# In[17]:


data.dropna(subset=['elevationcertificateindicator'],inplace=True)


# In[29]:


data.isnull().sum()


# In[31]:


data = data.drop(['agriculturestructureindicator'], axis = 1)
data.isnull().sum()


# In[8]:


data.info()


# In[9]:


data.duplicated()


# In[14]:


data.dtypes


# In[15]:


data.columns


# In[18]:


data.describe()


# In[41]:


data['reportedcity'].unique()


# In[31]:


data[data["reportedcity"] == "MIAMI"]


# In[59]:


data[data["reportedcity"] == "DENHAM SPRINGS"]
data[data["reportedcity"] == "MYRTLE BEACH"]


# In[10]:


r1=data['reportedcity'].replace({'DENHAM SPRINGS':'MYRTLE BEACH'}, inplace=True)
r1
#data["educational_level"].replace({"Master": "Post Graduation"}, inplace=True)


# In[53]:


#data.loc[:6, ['countycode','floodzone']]
data.loc[:10, ['countycode','ratemethod']]


# In[63]:


display = data[['countycode','construction','floodzone','policycount']]
display


# In[ ]:


cat_cols=data.select_dtypes(include=['object']).columns
  
num_cols = data.select_dtypes(include=np.number).columns.tolist()
 
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)


# In[ ]:


data.info()


# In[64]:


data.isna()


# In[66]:


data.isna().sum()


# In[67]:


data.duplicated()


# In[76]:


(data.isnull().sum()/(len(data)))*100


# In[ ]:


df = pd.DataFrame(data)

# Replace null values in the 'category_column' with a default value, e.g., 'Unknown'
default_value = 'B'
df['floodzone '] = df['floodzone'].fillna(default_value)

print(df)


# In[4]:


data.iloc[:5,[2,5]]


# In[ ]:


(data.isnull().sum()/(len(data)))*100


# In[ ]:


import pandas as pd

# Assuming you have a DataFrame named 'df'
# Replace missing values in the 'basefloodelevation' column with the mean of the column

mean = df['countycode'].mean()
df['countycode '] = df['countycode'].fillna(mean)

print("DataFrame after filling missing values in 'basefloodelevation' column with mean:")
print(df)


# In[ ]:


import pandas as pd

# Assuming you have a DataFrame named 'df'
# Replace missing values in the 'basefloodelevation' column with the mean of the column

mean = df['basefloodelevation'].mean()
df['basefloodelevation'] = df['basefloodelevation'].fillna(mean)

print("DataFrame after filling missing values in 'basefloodelevation' column with mean:")
print(df)


# In[28]:


(data.isnull().sum()/(len(data)))*100


# In[ ]:


df = pd.DataFrame(data)

# Replace null values in the 'category_column' with a default value, e.g., 'Unknown'
default_value = 'B'
df['floodzone '] = df['floodzone'].fillna(default_value)

print(df)


# In[2]:


data


# In[11]:


import pandas as pd
import numpy as np

your_dataset = pd.DataFrame(data)
def min_max_normalization(data):
   try:
       data = np.array(data, dtype=float)
       min_val = np.min(data)
       max_val = np.max(data)
       normalized_data = (data - min_val) / (max_val - min_val)
       return normalized_data
   except Exception as e:
       print("Error:", e)
       return None

# Assuming 'your_dataset' is the pandas DataFrame containing your data
countycode_column = your_dataset['countycode']
crsdiscount_column = your_dataset['crsdiscount']

# Applying min-max normalization to 'countycode' and 'crsdiscount'
normalized_countycode = min_max_normalization(countycode_column)
normalized_crsdiscount = min_max_normalization(crsdiscount_column)

# Adding the normalized attributes back to the DataFrame
your_dataset['normalized_countycode'] = normalized_countycode
your_dataset['normalized_crsdiscount'] = normalized_crsdiscount

# Print the updated DataFrame
print(your_dataset)


# In[12]:


data


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data['latitude'], bins=20, kde=True)
plt.xlabel('latitude')
plt.ylabel('Frequency')
plt.title('Distribution of latitude and longitude')
plt.show()


# In[17]:


correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[18]:


from sklearn.linear_model import LinearRegression
import numpy as np

your_dataset = pd.DataFrame(data)

# Assuming 'your_dataset' is the pandas DataFrame containing your data
# Replace 'your_dataset' with the actual variable name that holds your dataset
X = your_dataset[['ratemethod']]  # Independent variable (feature)
y = your_dataset['totalbuildinginsurancecoverage']  # Dependent variable (target)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict using the trained model
X_test = np.array([[new_ratemethod_1], [new_ratemethod_2]])  # New input data for prediction
y_pred = model.predict(X_test)

print(y_pred)


# In[19]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Assuming 'your_dataset' is the pandas DataFrame containing your data
# Replace 'your_dataset' with the actual variable name that holds your dataset

# Extract 'ratemethod' and 'totalbuildinginsurancecoverage' columns
X = your_dataset[['ratemethod']]
y = your_dataset['totalbuildinginsurancecoverage']

# One-hot encode the 'ratemethod' column
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_encoded, y)

# Predict using the trained model
X_test = np.array([[new_ratemethod_1]])  # New input data for prediction
X_test_encoded = encoder.transform(X_test)
y_pred = model.predict(X_test_encoded)

print(y_pred)


# In[22]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Assuming 'your_dataset' is the pandas DataFrame containing your data
# Replace 'your_dataset' with the actual variable name that holds your dataset

# Extract 'ratemethod' and 'totalbuildinginsurancecoverage' columns
X = your_dataset[['ratemethod']]
y = your_dataset['totalbuildinginsurancecoverage']

# One-hot encode the 'ratemethod' column
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_encoded, y)

# Predict using the trained model
new_ratemethod = '4'  # Replace this with the actual value for prediction
X_test = np.array([[new_ratemethod]])  # New input data for prediction
X_test_encoded = encoder.transform(X_test)
y_pred = model.predict(X_test_encoded)

print(y_pred)


# In[26]:


attribute_mapping = {'censustract': 'House No'}

# Use the rename() function to replace the attribute names in the DataFrame
data.rename(columns=attribute_mapping, inplace=True)
print(data)


# In[25]:


data


# In[12]:


plt.title("RateMethod Vs its Frequency")
data['ratemethod'].value_counts() .head(10).plot(kind='bar')


# In[10]:


plt.title("Cities")
data['reportedcity'].value_counts() .head(10).plot(kind='bar')
 


# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.bar(data['ratemethod'].data['reportedcity'])
plt.ylabel("ratemethod")
plt.xlabel("reportedcity")
plt.title("ratemethod vs reportedcity")
plt.show()


# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
# If 'ratemethod' and 'reportedcity' are columns in the DataFrame
# then you can group by 'ratemethod' and count occurrences of 'reportedcity'
grouped_data = data.groupby('ratemethod')['reportedcity'].count().reset_index()

# Plot the data using a bar plot
plt.bar(grouped_data['ratemethod'], grouped_data['reportedcity'])
plt.ylabel("Number of reported cities")
plt.xlabel("Rate Method")
plt.title("Rate Method vs Reported City")
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
# If 'ratemethod' and 'reportedcity' are columns in the DataFrame
# then you can group by 'ratemethod' and count occurrences of 'reportedcity'
grouped_data = data.groupby('reportedcity')['ratemethod'].count().reset_index()

# Plot the data using a bar plot
plt.bar(grouped_data['reportedcity'], grouped_data['ratemethod'])
plt.ylabel("Number of reported cities")
plt.xlabel("Reported City")
plt.title("Reported City ")
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()


# In[ ]:





# In[20]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.boxplot(x=data['totalbuildinginsurancecoverage'])


# In[ ]:




