#!/usr/bin/env python
# coding: utf-8

# # 10Alytics Capstone Project- Blossom Bank PLC

# ## Online Payments Fraud Detection

# ### Problem Definition
# 
# - Identification of transaction types that can lead to fraud.
# 
# - Identification of fraud in payment.
# 
# ### How will the business benefit from your solution
# 
# Given the insight gained from the data set, Blossom Bank can choose to adopt the use of any of the ML Algorithms tested to Block Fake Accounts, Detect Payment Fraud and Prevent Content Spam.

# In[1]:


# Import all the necessary libraries
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ## Data Inspection

# In[2]:


# Load the dataset - Online Payment Fraud Detection.csv

df = pd.read_csv(r'C:\Users\DANIEL\Downloads\DATA\CAPSTONE PROJECT\Online Payment Fraud Detection.csv')


# In[3]:


# View Data (first 5 rows of the data set)

df.head()


# In[4]:


# Buttom 5 rows

df.tail()


# ## Data Dictionary
# 
# • step: represents a unit of time where 1 step equals 1 hour
# 
# • type: type of online transaction
# 
# • amount: the amount of the transaction
# 
# • nameOrig: customer starting the transaction
# 
# • oldbalanceOrg: balance before the transaction
# 
# • newbalanceOrig: balance after the transaction
# 
# • nameDest: recipient of the transaction
# 
# • oldbalanceDest: initial balance of recipient before the transaction
# 
# • newbalanceDest: the new balance of the recipient after the transaction
# 
# • isFraud: fraud transaction

# In[5]:


# Rename the column headers

df.columns = ["step", "transaction type", "amount", "customer id", "previous balance", "present balance", "receiver id", "receiver previous balance", "receiver present balance", "fraudulent transaction"]

df.head()


# In[6]:


# Data verification

df.info()


# In[7]:


# Dimension of the data

df.shape


# In[8]:


# Statistical Analysis of the data

df.describe().astype(int)


# In[9]:


# Missing Value

df.isnull().sum()


# In[10]:


# Visualizing missing values

plt.figure(figsize = (10, 5))
plt.title("Visualizing missing values in dataset")
sns.heatmap(df.isnull(), cbar=True, cmap="YlGnBu_r")


# ## Exploratory Data Analysis

# ### Univariate Analysis

# In[11]:


# Drop redundant columns

df.drop(["customer id", "receiver id"], axis = 1, inplace = True)


# In[12]:


df.head()


# In[13]:


# Creating new columns for easy visualization


# In[14]:


# amount

# create a function that categorises amount

def amount_bracket(amount):
    if amount <= 2000000:
        return "<=2M"
    elif amount <=4000000:
        return "<=4M"
    elif amount <=6000000:
        return "<=6M"
    elif amount <= 8000000:
        return "<=8M"
    else: 
        return ">10M)"
    
# create a new function for amount category

df["amount_group"] = df["amount"].apply(amount_bracket)

df.head()


# In[15]:


# visualize amount group

plt.figure(figsize=(10,5))
plt.ticklabel_format(style='plain')
viz = sns.countplot(x = "amount_group", data = df)
viz.set_xticklabels(viz.get_xticklabels(), rotation=90)
for i in viz.patches:
    height = i.get_height()
    viz.text(i.get_x() + i.get_width()/2., height + 0.1, height, ha="center")
sns.countplot(x="amount_group", data=df)
plt.title("Catergory of Amount")
plt.xlabel("Amount Group")
plt.ylabel("Count of Amount Group")


# In[16]:


# previous balance

# create a function that categorises previous balance

def pb_bracket(previous_balance):
    if previous_balance <= 7780000:
        return "<=7.7M"
    elif previous_balance <=15560000:
        return "<=15.5M"
    elif previous_balance <=23340000:
        return "<=23.3M"
    elif previous_balance <= 31120000:
        return "<=31.1M"
    else: 
        return ">31.1M"
    
# create a new function for previous balance category

df["previous_balance_group"] = df["previous balance"].apply(pb_bracket)

df.head()


# In[17]:


# visualize previous balance

plt.figure(figsize=(10,5))
plt.ticklabel_format(style='plain')
viz = sns.countplot(x = "previous_balance_group", data = df)
viz.set_xticklabels(viz.get_xticklabels(), rotation=90)
for i in viz.patches:
    height = i.get_height()
    viz.text(i.get_x() + i.get_width()/2., height + 0.1, height, ha="center")
sns.countplot(x="previous_balance_group", data=df)
plt.title("Catergory of Previous Balance")
plt.xlabel("Previous Balance Group")
plt.ylabel("Count of Previous Balance Group")


# In[18]:


# present balance

# create a function that categorises present balance

def pb_bracket(present_balance):
    if present_balance <= 7780000:
        return "<=7.7M"
    elif present_balance <=15560000:
        return "<=15.5M"
    elif present_balance <=23340000:
        return "<=23.3M"
    elif present_balance <= 31120000:
        return "<=31.1M"
    else: 
        return ">31.1M"
    
# create a new function for present balance category

df["present_balance_group"] = df["present balance"].apply(pb_bracket)

df.head()


# In[19]:


# visualize present balance

plt.figure(figsize=(10,5))
plt.ticklabel_format(style='plain')
viz = sns.countplot(x = "present_balance_group", data = df)
viz.set_xticklabels(viz.get_xticklabels(), rotation=90)
for i in viz.patches:
    height = i.get_height()
    viz.text(i.get_x() + i.get_width()/2., height + 0.1, height, ha="center")
sns.countplot(x="present_balance_group", data=df)
plt.title("Catergory of Present Balance")
plt.xlabel("Present Balance Group")
plt.ylabel("Count of Present Balance Group")


# In[20]:


# receiver previous balance

# create a function that categorises receiver previous balance

def rpb_bracket(receiver_previous_balance):
    if receiver_previous_balance <= 8420000:
        return "<=8.4M"
    elif receiver_previous_balance <= 16840000:
        return "<=16.8M"
    elif receiver_previous_balance <= 25260000:
        return "<=25.2M"
    elif receiver_previous_balance <= 33680000:
        return "<=33.6M"
    else:
        return ">33.6M)"
    
# create a new function for receiver previous balance category

df["receiver_previous_balance_group"] = df["receiver previous balance"].apply(rpb_bracket)

df.head()


# In[21]:


# receiver previous balance

plt.figure(figsize=(10,5))
plt.ticklabel_format(style='plain')
viz = sns.countplot(x = "receiver_previous_balance_group", data = df)
viz.set_xticklabels(viz.get_xticklabels(), rotation=90)
for i in viz.patches:
    height = i.get_height()
    viz.text(i.get_x() + i.get_width()/2., height + 0.1, height, ha="center")
sns.countplot(x="receiver_previous_balance_group", data=df)
plt.title("Catergory of Receiver Previous Balance")
plt.xlabel("Receiver Previous Balance Group")
plt.ylabel("Count of Receiver Previous Balance Group")


# In[22]:


# receiver present balance

# create a function that categorises receiver present balance

def rpb_bracket(receiver_present_balance):
    if receiver_present_balance <= 8420000:
        return "<=8.4M"
    elif receiver_present_balance <= 16840000:
        return "<=16.8M"
    elif receiver_present_balance <= 25260000:
        return "<=25.2M"
    elif receiver_present_balance <= 33680000:
        return "<=33.6M"
    else:
        return ">33.6M)"
    
# create a new function for receiver present balance category

df["receiver_present_balance_group"] = df["receiver present balance"].apply(rpb_bracket)

df.head()


# In[23]:


# receiver present balance

plt.figure(figsize=(10,5))
plt.ticklabel_format(style='plain')
viz = sns.countplot(x = "receiver_present_balance_group", data = df)
viz.set_xticklabels(viz.get_xticklabels(), rotation=90)
for i in viz.patches:
    height = i.get_height()
    viz.text(i.get_x() + i.get_width()/2., height + 0.1, height, ha="center")
sns.countplot(x="receiver_present_balance_group", data=df)
plt.title("Catergory of Receiver Present Balance")
plt.xlabel("Receiver Present Balance Group")
plt.ylabel("Count of Receiver Present Balance Group")


# In[24]:


# step

plt.figure(figsize = (10,5))
sns.histplot(x = 'step', data = df)
plt.title('Steps Count')
plt.show()


# In[25]:


# create visualization for transaction type 

plt.figure(figsize = (10,5))
viz = sns.countplot(x="transaction type", data = df)
viz.set_xticklabels(viz.get_xticklabels(), rotation=90)
for i in viz.patches:
    height = i.get_height()
    viz.text(i.get_x() + i.get_width()/2., height + 0.1, height, ha="center")
sns.countplot(x="transaction type", data=df)
sns.countplot(x ='transaction type', data=df)
plt.title('Transaction Type')
plt.show()


# In[26]:


# fraudulent transaction

# create a function that categorises fraudulent transaction

def ft_group(fraudulent_transaction):
    if fraudulent_transaction == 1:
        return 'Yes'
    else:
        return 'No'
    
df['ft_group'] = df['fraudulent transaction'].apply(ft_group)

# create visualization for fraudulent transaction

plt.figure(figsize = (10,5))
viz = sns.countplot(x="fraudulent transaction", data = df)
viz.set_xticklabels(viz.get_xticklabels(), rotation=90)
for i in viz.patches:
    height = i.get_height()
    viz.text(i.get_x() + i.get_width()/2., height + 0.1, height, ha="center")
sns.countplot(x="fraudulent transaction", data=df)
sns.countplot(x ='fraudulent transaction', data=df)
plt.title('Fraudulent Transaction Type')
plt.show()


# In[27]:


#Create visualization for transaction type and fraudulent transaction

plt.figure(figsize = (10,5))
viz = sns.countplot(x="transaction type", data = df, hue="fraudulent transaction")
viz.set_xticklabels(viz.get_xticklabels(), rotation=90)
for i in viz.patches:
    height = i.get_height()
    viz.text(i.get_x() + i.get_width()/2., height + 0.1, height, ha="center")
sns.countplot(x="transaction type", data=df)
sns.countplot(x ="transaction type", data=df)
plt.title('Transaction Type')
plt.show()


# In[28]:


# fraudulent transaction

# create a function that categorises fraudulent transaction

def ft_group(fraudulent_transaction):
    if fraudulent_transaction == 1:
        return 'Yes'
    else:
        return 'No'
    
df['ft_group'] = df['fraudulent transaction'].apply(ft_group)

#Create visualization for fraudulent transaction

plt.figure(figsize = (10,5))
plt.title('Fraud Transaction Category')
df['ft_group'].value_counts(normalize=True).plot.pie(autopct="%.2f")


# ### Bivariate Analysis

# In[29]:


# transaction type by amount

fig, ax1 = plt.subplots(figsize = (10,5))
plt.ticklabel_format(style='plain')
sns.barplot( x = "transaction type", y = "amount", data = df, ci = None)
plt.title('Transaction type by Amount')


# ### Multivariate Analysis

# In[30]:


# Transaction Type by Amount Per Fraud Transaction

fig, ax1 = plt.subplots(figsize = (10,5))
plt.ticklabel_format(style='plain')
sns.barplot(ax =ax1, x = "transaction type", y = "amount", data = df, hue = "fraudulent transaction", ci = None)
plt.title('Transaction type by Amount Per Fraud Transaction')


# ### Correlation Analysis

# In[31]:


# correlation Analysis

correl = df.corr()


# num = 10

# cols = corel.nlargest(num, "target")["target"].index
sns.heatmap(correl, cbar=True, annot=True,fmt=".2f", annot_kws={'size': 12})


# - Strong postive relationship betweeen previous balance and present balance.
# - Weak positive relationship between fraudulent transaction and amount.

# ### Feature Engineering

# ### Encoding Categorical Variable- One Hot Code

# In[32]:


# conversion of catergorical column to numeric

transaction_type_num = pd.get_dummies(df["transaction type"])
transaction_type_num.head(2)


# In[33]:


# join the encoded variable back to the data frame

df = pd.concat([df, transaction_type_num], axis = 1)

# View data
print(df.shape)
df.head(2)


# In[34]:


# drop other categorical data columns used for visualization purpose

df.drop(['amount_group',
       'previous_balance_group', 'present_balance_group',
       'receiver_previous_balance_group', 'receiver_present_balance_group', 'ft_group'], axis = 1, inplace = True)

df.head(2)


# In[35]:


# drop transaction type column used for the one hot code

df.drop('transaction type', axis = 1, inplace = True)

df.head(2)


# ### Creating Target X and y

# In[36]:


y = df['fraudulent transaction'] 
X = df.drop('fraudulent transaction', axis=1)


# In[37]:


y.head()


# In[38]:


X.head()


# ### ML Algorithm 1

# In[48]:


# Machine Learning
from sklearn.model_selection import train_test_split

# ML Algorithms
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

# ML Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix


# To disable warnings

import warnings
warnings.filterwarnings("ignore")


# In[49]:


# Load three ML Algorithms

classifiers = [[RandomForestClassifier(), "Random Forest"], [KNeighborsClassifier(), "K-Nearest Neighbors"], [LogisticRegression(), "Logistic Regression"]]


# In[50]:


# Train Test Split (Training on 80% while Testing is 20%)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

accuracy_score_list = {}

for i in classifiers:
    model = i[0]
    model.fit(X_train, y_train)
    model_name= i[1]
    
    predict = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predict)
    
    accuracy_score_list[model_name] = ([str(round(accuracy * 100, 2)) + "%"])
    
    if model_name != classifiers[-1][1]:
        print('')


# In[51]:


accuracy_score_list


# In[52]:


print ("Accuracy Score of ML Algorithms")
as_df = pd.DataFrame(accuracy_score_list)
as_df


# In[66]:


# Machine Learning
from sklearn.model_selection import train_test_split


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[68]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()


# In[69]:


LR.fit(X_train,y_train)


# In[70]:


# Calculate the confusion matrix for LogisticRegressionClassifier

conf_matrix = confusion_matrix(y_true=y_test, y_pred=LR.predict(X_test))

# Print the confusion matrix

fig, ax = plt.subplots(figsize=(8, 4))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.2)
for i in range(conf_matrix.shape[1]):
    for j in range(conf_matrix.shape[0]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title(' LogisticRegression Confusion Matrix', fontsize=16)
plt.show()

print('Precision: %.2f' % precision_score(y_test, LR.predict(X_test)))
print('Recall: %.2f' % recall_score(y_test, LR.predict(X_test)))


# In[61]:


# Machine Learning
from sklearn.model_selection import train_test_split


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[63]:


from sklearn.neighbors import KNeighborsClassifier

KN = KNeighborsClassifier()


# In[64]:


KN.fit(X_train,y_train)


# In[65]:


# Calculate the confusion matrix for KNeighborClassifier

conf_matrix = confusion_matrix(y_true=y_test, y_pred=KN.predict(X_test))

# Print the confusion matrix

fig, ax = plt.subplots(figsize=(8, 4))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.2)
for i in range(conf_matrix.shape[1]):
    for j in range(conf_matrix.shape[0]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title(' KNeighbor Confusion Matrix', fontsize=16)
plt.show()

print('Precision: %.2f' % precision_score(y_test, KN.predict(X_test)))
print('Recall: %.2f' % recall_score(y_test, KN.predict(X_test)))


# In[76]:


# Machine Learning
from sklearn.model_selection import train_test_split


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[78]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()


# In[79]:


RF.fit(X_train,y_train)


# In[80]:


# Calculate the confusion matrix for RandomForestClassifier

conf_matrix = confusion_matrix(y_true=y_test, y_pred=RF.predict(X_test))

# Print the confusion matrix

fig, ax = plt.subplots(figsize=(8, 4))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.2)
for i in range(conf_matrix.shape[1]):
    for j in range(conf_matrix.shape[0]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title(' RandomForest Confusion Matrix', fontsize=16)
plt.show()

print('Precision: %.2f' % precision_score(y_test, RF.predict(X_test)))
print('Recall: %.2f' % recall_score(y_test, RF.predict(X_test)))


# ### Summary
# 
# Random Forest Classifier is selected as a better ML Algorithm to be deployed by Blossom Bank to predict online payment fraud as it has a higher recall of 0.82 and precision of 0.98 with 99.98% accuracy.
