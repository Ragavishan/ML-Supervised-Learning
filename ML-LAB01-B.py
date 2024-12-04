#!/usr/bin/env python
# coding: utf-8

# # Predict whether someone like pizza or not

# ### Import Necessary libraries and Dataset

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import seaborn as sns


# In[4]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.ensemble import RandomForestClassifier


# In[5]:


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# ###   Example dataset

# In[7]:


data = {
    'Age': [25, 34, 45, 23],
    'Gender': ['Male', 'Female', 'Male', 'Female'],
    'Likes Fast Food': ['Yes', 'No', 'Yes', 'Yes'],
    'Likes Cheese': ['Yes', 'No', 'No', 'Yes'],
    'Likes Italian Cuisine': ['Yes', 'Yes', 'No', 'Yes'],
    'Likes Pizza': [1, 0, 0, 1]
}


# ###  Convert to DataFrame

# In[8]:


df = pd.DataFrame(data)


# ### Visualize the raw data (with Seaborn)

# In[9]:


plt.figure(figsize=(8, 6))


# In[12]:


sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()


# ### Preprocess categorical columns

# In[13]:


df = pd.get_dummies(df, drop_first=True)


# ### Features (X) and target (y)

# In[14]:


X = df.drop('Likes Pizza', axis=1) 
y = df['Likes Pizza']


# In[15]:


print("\nX Matrix (Features):")
print(X.head())   


# In[16]:


print("\nY Vector (Target):")
print(y.head())


# ###  Examine X and y

# In[18]:


print("\nX Matrix (Features):")
print(X.head())   
print("\nY Vector (Target):")
print(y.head())


# ### 1. Pairplot: Visualize relationships between features and target

# In[19]:


sns.pairplot(df, hue='Likes Pizza', palette='Set1')
plt.suptitle('Pairplot: Features vs Likes Pizza', y=1.02)
plt.show()


# ### 2. Heatmap of correlation to target variable

# In[20]:


corr = df.corr()


# In[21]:


plt.figure(figsize=(8, 6))


# In[22]:


sns.heatmap(corr[['Likes Pizza']], annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation of Features with Likes Pizza')
plt.show()


# ###  Split data into training and testing sets

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#  ### Initialize and train the model

# In[24]:


model = RandomForestClassifier()
model.fit(X_train, y_train)


# ###  Make predictions

# In[25]:


y_pred = model.predict(X_test)


# ### Evaluate the model

# In[26]:


accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ### Confusion Matrix visualization with Seaborn

# In[27]:


cm = confusion_matrix(y_test, y_pred)


# In[28]:


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Does Not Like', 'Likes'], yticklabels=['Does Not Like', 'Likes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ### Feature Importance visualization with Seaborn

# In[29]:


feature_importances = model.feature_importances_
features = X.columns


# In[30]:


plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=features, palette='Blues_d')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()


# In[ ]:




