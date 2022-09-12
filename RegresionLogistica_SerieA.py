#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime

s10 = pd.read_csv('./archive/season-0910.csv')
s10['Date']= pd.to_datetime(s10['Date'],infer_datetime_format=True)

s11 = pd.read_csv('./archive/season-1011.csv')
s11['Date']= pd.to_datetime(s11['Date'],infer_datetime_format=True)

s12 = pd.read_csv('./archive/season-1112.csv')
s12['Date']= pd.to_datetime(s12['Date'],infer_datetime_format=True)

s13 = pd.read_csv('./archive/season-1213.csv')
s13['Date']= pd.to_datetime(s13['Date'],infer_datetime_format=True)

s14 = pd.read_csv('./archive/season-1314.csv')
s14['Date']= pd.to_datetime(s14['Date'],infer_datetime_format=True)

s15 = pd.read_csv('./archive/season-1415.csv')
s15['Date']= pd.to_datetime(s15['Date'],infer_datetime_format=True)

s16 = pd.read_csv('./archive/season-1516.csv')
s16['Date']= pd.to_datetime(s16['Date'],infer_datetime_format=True)

s17 = pd.read_csv('./archive/season-1617.csv')
s17['Date']= pd.to_datetime(s17['Date'],infer_datetime_format=True)

s18 = pd.read_csv('./archive/season-1718.csv')
s18['Date']= pd.to_datetime(s18['Date'],infer_datetime_format=True)

s19 = pd.read_csv('./archive/season-1819.csv')
s19['Date']= pd.to_datetime(s19['Date'],infer_datetime_format=True)

Seasons = [s10,s11,s12,s13,s14,s15,s16,s17,s18,s19]

data = pd.concat(Seasons)
data.reset_index(inplace=True)


# In[2]:


data['HTR'] =np.where(data['HTR']=='H',3,np.where(data['HTR']=='D',1,0))

data['FTR'] =np.where(data['FTR']=='H',3,np.where(data['FTR']=='D',1,0))


# In[3]:


data['Season'] = data['Date'].dt.to_period('Y')


# In[4]:


print(list(data.columns))


# In[5]:


df = data[['Season','Date','AwayTeam','HomeTeam','AS','HS','AST','HST','HTAG','HTHG','HTR','FTAG','FTHG','FTR','AC','HC','AF','HF','AY','HY','AR','HR']]
df


# In[6]:


df.dropna(inplace=True)
df


# In[7]:


keys = df['HomeTeam'].unique()
values = list(range(len(keys)))

values
dictionary = dict(zip(keys, values))
dictionary


# In[8]:


df.replace({"AwayTeam": dictionary}, inplace =True)
df.replace({"HomeTeam": dictionary}, inplace =True)
df


# In[9]:


print(list(df.columns))


# In[10]:


df2 = df["AwayTeam"].values
df2 = pd.DataFrame(df2,columns = ['Away_Team'])

df3 = df["HomeTeam"].values
df3 = pd.DataFrame(df3,columns = ['Home_Team'])

df['Away_Team'] = df2
df['Home_Team'] = df3
df.drop(columns  = ['HomeTeam','AwayTeam'], axis = 1,inplace =True)
df.reset_index(inplace = True)
df


# In[11]:


df.drop(columns  = ['index'], axis = 1,inplace =True)
df


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

y = df["FTR"]
X = df.drop(["Season", "Date", "Home_Team", "Away_Team", "FTR"], axis=1)
print(len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)


# In[18]:


X_train.dropna(inplace=True)
X_test.dropna(inplace=True)
y_train.dropna(inplace=True)
y_test.dropna(inplace=True)

X_train.describe()


# In[19]:


from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression(random_state=11, multi_class="multinomial", solver="saga")
modelo.fit(X_train, y_train)


# In[20]:


y_pred = modelo.predict(X_test)


# In[21]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score


# Accuracy (Precisión)
acc_score = accuracy_score(y_test, y_pred)
print(acc_score)

baseline_acc = len(y[y == 0]) / len(y) 
cohens_score = cohen_kappa_score(y_test, y_pred) #Encuentra el score de Cohens; mayor significa menos aleatoreidad fuera de 1.
print(baseline_acc)
print(cohens_score)


# In[ ]:




