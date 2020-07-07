#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


# In[4]:


flight_prc=pd.read_excel('D:/Machine learning data set/Flight-Price-Prediction-master/Flight-Price-Prediction-master/Data_Train.xlsx')
flight_prc.head()


# In[5]:


flight_prc.dropna(inplace=True)


# In[6]:


flight_prc.query('Total_Stops == "2 stops" ' ).head()


# In[7]:


flight_prc['Date_of_Journey']=flight_prc['Date_of_Journey'].apply(lambda x : x.replace('/','-'))
flight_prc.head()


# In[8]:


flight_prc.Total_Stops=flight_prc.Total_Stops.apply(lambda x : x.split()[0] if x not in 'non-stop' else 0)
flight_prc.head()


# In[9]:


flight_prc.Total_Stops=flight_prc.Total_Stops.astype('int')


# In[10]:


import datetime
import re


# In[11]:


flight_prc['Duration']=flight_prc['Duration'].apply(lambda x : re.sub('[a-z]','',x))
flight_prc.head()


# In[12]:


#pd.to_datetime(flight_prc['Duration'])
def minute_conv(x):
    if len(x.split()) == 2:
        return (int(x.split()[0])*60)+(int(x.split()[1]))
    else:
        return int(x)*60
flight_prc['Duration_in_minute']=flight_prc['Duration'].apply(minute_conv)


# In[13]:


flight_prc.Total_Stops.unique()


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


flight_prc.Additional_Info=flight_prc.Additional_Info.apply(lambda x : x.replace('No info','No Info'))
flight_prc.head()


# In[16]:


flight_prc['Arrival_Time']=flight_prc['Arrival_Time'].apply(lambda x : re.sub('[a-zA-Z]','',x))
flight_prc.head()
#flight_prc.query('Airline == "IndiGo" ').head()


# In[17]:


flight_prc.Arrival_Time=flight_prc.Arrival_Time.apply(lambda x : x.split()[0])
flight_prc.head()


# In[18]:


flight_prc['Dep_Time_hr']=flight_prc['Dep_Time'].apply(lambda x :x.split(':')[0])
flight_prc['Dep_Time_min']=flight_prc['Dep_Time'].apply(lambda x :x.split(':')[1])
flight_prc['Arrival_Time_hr']=flight_prc['Arrival_Time'].apply(lambda x :x.split(':')[0])
flight_prc['Arrival_Time_min']=flight_prc['Arrival_Time'].apply(lambda x :x.split(':')[1])
flight_prc.head()


# In[19]:


le=LabelEncoder()
flight_prc['Source_']=le.fit_transform(flight_prc['Source'])
flight_prc['Destination_']=le.fit_transform(flight_prc['Destination'])
flight_prc['Additional_Info_']=le.fit_transform(flight_prc['Additional_Info'])
flight_prc['Airline_']=le.fit_transform(flight_prc['Airline'])


# In[20]:


flight_prc.head()


# In[21]:


flight_prc['date']=flight_prc.Date_of_Journey.apply(lambda x:x.split('-')[0])
flight_prc['month']=flight_prc.Date_of_Journey.apply(lambda x:x.split('-')[1])
flight_prc['year']=flight_prc.Date_of_Journey.apply(lambda x:x.split('-')[2])
flight_prc.head()


# In[27]:


flight_prc.head()


# In[77]:


sns.pairplot(flight_prc)


# In[22]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score


# In[23]:


flight_prc.columns


# In[24]:


x=flight_prc.iloc[:,[8,11,12,13,14,15,16,17,18,19,20,21]]
y=flight_prc.iloc[:,10]


# In[25]:


x.head()


# In[23]:


rfg=RandomForestRegressor(n_estimators=700)


# In[26]:


cros=cross_val_score(rfg,x,y,scoring='neg_mean_squared_error',cv=10)
cros


# In[27]:


cros.mean()


# In[26]:


from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor


# In[28]:


grad_boost=GradientBoostingRegressor(n_estimators=500)
cros=cross_val_score(grad_boost,x,y,scoring='r2',cv=10)
cros


# In[29]:


cros.mean()


# In[31]:


from sklearn.model_selection import RandomizedSearchCV


# In[33]:


param={'n_estimators':[100,200,400,600],
      'learning_rate':[0.1,0.15,0.2],
      'min_samples_split':[150,200,500],
      'min_samples_leaf':[50,100],
      'max_depth':[2,4,6],
      'max_features':['sqrt','log2']}


# In[34]:


gscv=RandomizedSearchCV(grad_boost,param_distributions=param,scoring='r2',cv=8,n_jobs=-1)
gscv.fit(x,y)


# In[37]:


gscv.best_params_
#gscv.best_score_


# In[38]:


grad_boost=GradientBoostingRegressor(n_estimators= 600,
 min_samples_split=200,
 min_samples_leaf=100,
 max_features='log2',
 max_depth=6,
 learning_rate= 0.15)


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.25)


# In[40]:


grad_boost.fit(x_train,y_train)
pred=grad_boost.predict(x_test)


# In[42]:


mean_squared_error(y_test,pred)
mean_absolute_error(y_test,pred)


# In[43]:


import pickle


# In[44]:


pickle.dump(grad_boost,open('grad_boost.pkl','wb'))
grad_boost=pickle.load(open('grad_boost.pkl','rb'))

