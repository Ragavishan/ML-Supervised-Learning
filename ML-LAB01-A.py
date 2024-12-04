#!/usr/bin/env python
# coding: utf-8

# # SUPERVISED LEARNING

# ## CLASSIFY EMAILS AS SPAM OR NOT-SPAM

# ### Import Required Libraries

# In[1]:


import pandas as pd


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


from sklearn.naive_bayes import MultinomialNB


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer


# In[22]:


from sklearn.metrics import accuracy_score


# ### Sample email data 

# In[23]:


data = {
    'text': [
        "Free money, call now!", 
        "Hello, I hope you are doing well.", 
        "Get a loan in minutes, guaranteed!", 
        "Hi John, can we meet tomorrow?",
        "Earn cash from home, no experience needed!", 
        "Meeting at 3 PM today, please confirm.",
        "Congratulations! You've won a prize!", 
        "Are you available for a quick meeting?",
        "Get rich quick, limited time offer!", 
        "Reminder: Meeting at 3 PM tomorrow."
    ],
     'label': [
        1,  
        0,  
        1,
        0,  
        1,  
        0,  
        1,  
        0,  
        1,  
        0   
    ]
}


# ### Convert to DataFrame

# In[24]:


df = pd.DataFrame(data)


# ### Separate features (X) and labels (y)

# In[25]:


X = df['text']


# In[26]:


y = df['label']


# ### Split the data into training and testing sets (70% train, 30% test)

# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ### Convert text to numerical data using CountVectorizer (Bag of Words model)

# In[28]:


vectorizer = CountVectorizer(stop_words='english')


# In[29]:


X_train_vec = vectorizer.fit_transform(X_train)


# In[30]:


X_test_vec = vectorizer.transform(X_test)


# ### Initialize and train the Naive Bayes classifier

# In[31]:


model = MultinomialNB()
model.fit(X_train_vec, y_train)


# ### Make predictions on the test data

# y_pred = model.predict(X_test_vec)
# 

# ### Evaluate the model's performance

# In[33]:


accuracy = accuracy_score(y_test, y_pred)


# In[34]:


print(f'Accuracy: {accuracy * 100:.2f}%')


# ### Test the classifier with some new email samples

# In[35]:


test_emails = [
    "Claim your free iPhone now!", 
    "Can we reschedule the meeting?", 
    "Limited time offer for you, act now!"
]


# ### Vectorize the new test emails and make predictions

# In[36]:


test_vec = vectorizer.transform(test_emails)


# In[37]:


predictions = model.predict(test_vec)


# ### Output predictions

# In[38]:


for email, pred in zip(test_emails, predictions):
    print(f"Email: {email}")
    print(f"Predicted: {'Spam' if pred == 1 else 'Not Spam'}\n")


# In[ ]:




