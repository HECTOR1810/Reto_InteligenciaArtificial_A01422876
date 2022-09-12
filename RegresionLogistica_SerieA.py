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
data.reset_index(drop = True, inplace=True)


# In[2]:


data['HTR'] =np.where(data['HTR']=='H',3,np.where(data['HTR']=='D',1,0))

data['FTR'] =np.where(data['FTR']=='H',3,np.where(data['FTR']=='D',1,0))


# In[3]:


#data['Season'] = data['Date'].dt.to_period('Y')
data['Season']= data['Date'].dt.year
data


# In[4]:


print(list(data.columns))


# In[5]:


print(data.isnull().sum())
df = data[['Season','Date','AwayTeam','HomeTeam','AS','HS','AST','HST','HTAG','HTHG','HTR','FTAG','FTHG','FTR','AC','HC','AF','HF','AY','HY','AR','HR']]
df


# In[6]:


df.dropna(inplace=True)
df.reset_index(drop = True, inplace=True)
df


# In[7]:


keys = df['HomeTeam'].unique()
values = list(range(len(keys)))

values
dictionary2 = dict(zip(values,keys))
dictionary = dict(zip(keys, values))
dictionary


# In[8]:


df.replace({"AwayTeam": dictionary}, inplace =True)
df.replace({"HomeTeam": dictionary}, inplace =True)
df


# In[9]:


df[df.columns].dtypes


# In[10]:


df[df.columns].isnull().sum()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

y = df["FTR"]
X = df.drop(["Season", "Date", "HomeTeam", "AwayTeam", "FTR"], axis=1)
print(len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[12]:


y_test.isnull().sum()


# In[13]:


from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression(random_state=11, multi_class="multinomial", solver="saga", penalty = 'l2', C = 0.01)
modelo.fit(X_train, y_train)


# In[14]:


y_pred = modelo.predict(X_test)


# In[15]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score


# Accuracy (Precisión)
acc_score = accuracy_score(y_test, y_pred)
print(acc_score)

peor_acc = len(y[y == 0]) / len(y) 
cohens_score = cohen_kappa_score(y_test, y_pred) #Encuentra el score de Cohens; mayor significa menos aleatoreidad fuera de 1.
print(peor_acc)
print(cohens_score)


# In[16]:


importance = modelo.coef_[0]
names = list(X.columns)
f_importance = pd.DataFrame(names, columns = ["Variable"])
f_importance["Importancia"] = importance
f_importance = f_importance.sort_values(by = ["Importancia"], ascending=False)
f_importance.reset_index(drop = True, inplace=True)
f_importance


# In[17]:


import matplotlib.pyplot as plt
ax = f_importance.plot.bar(x='Variable', y='Importancia')
plt.show()


# In[25]:


S19 = df[df['Season'] >= 2018]
S19.reset_index(drop=True,inplace=True)
S19


# In[27]:


PruebaF = S19.drop(["Season", "Date", "HomeTeam", "AwayTeam", "FTR"], axis=1)
PruebaF


# In[28]:


y_pred = modelo.predict(PruebaF)
y_pred


# In[29]:


PruebaF['FTR'] = y_pred
PruebaF['HomeTeam'] = S19['HomeTeam']
PruebaF['AwayTeam'] = S19['AwayTeam']
PruebaF


# In[45]:


dfF = PruebaF.groupby(['HomeTeam'])['FTR'].sum().reset_index()
dfF = dfF.sort_values(by = ["FTR"], ascending=False).reset_index(drop=True)
dfF["FTR_Pred"] = dfF["FTR"].reset_index(drop=True)
dfF.drop(['FTR'],axis = 1, inplace = True)
dfF


# In[46]:


S19 = S19.groupby(['HomeTeam'])['FTR'].sum().reset_index()
S19 = S19.sort_values(by = ["FTR"], ascending=False).reset_index(drop=True)
S19


# In[47]:


S19.replace({"HomeTeam": dictionary2},inplace=True)
S19


# In[48]:


dfF.replace({"HomeTeam": dictionary2},inplace=True)
dfF


# In[49]:


result = pd.merge(S19, dfF, on="HomeTeam")
result


# In[ ]:




