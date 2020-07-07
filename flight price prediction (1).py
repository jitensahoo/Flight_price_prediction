#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


# In[2]:


flight_prc=pd.read_excel('D:/Machine learning data set/Flight-Price-Prediction-master/Flight-Price-Prediction-master/Data_Train.xlsx')
flight_prc.head()


# In[3]:


flight_prc.dropna(inplace=True)


# In[4]:


flight_prc.query('Total_Stops == "2 stops" ' ).head()


# In[5]:


flight_prc['Date_of_Journey']=flight_prc['Date_of_Journey'].apply(lambda x : x.replace('/','-'))
flight_prc.head()


# In[6]:


flight_prc.Total_Stops=flight_prc.Total_Stops.apply(lambda x : x.split()[0] if x not in 'non-stop' else 0)
flight_prc.head()


# In[7]:


flight_prc.Total_Stops=flight_prc.Total_Stops.astype('int')


# In[8]:


import datetime
import re


# In[9]:


flight_prc['Duration']=flight_prc['Duration'].apply(lambda x : re.sub('[a-z]','',x))
flight_prc.head()


# In[10]:


#pd.to_datetime(flight_prc['Duration'])
def minute_conv(x):
    if len(x.split()) == 2:
        return (int(x.split()[0])*60)+(int(x.split()[1]))
    else:
        return int(x)*60
flight_prc['Duration_in_minute']=flight_prc['Duration'].apply(minute_conv)


# In[23]:


flight_prc.Total_Stops.unique()


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


flight_prc.Additional_Info=flight_prc.Additional_Info.apply(lambda x : x.replace('No info','No Info'))
flight_prc.head()


# In[13]:


flight_prc['Arrival_Time']=flight_prc['Arrival_Time'].apply(lambda x : re.sub('[a-zA-Z]','',x))
flight_prc.head()
#flight_prc.query('Airline == "IndiGo" ').head()


# In[14]:


flight_prc.Arrival_Time=flight_prc.Arrival_Time.apply(lambda x : x.split()[0])
flight_prc.head()


# In[15]:


flight_prc['Dep_Time_hr']=flight_prc['Dep_Time'].apply(lambda x :x.split(':')[0])
flight_prc['Dep_Time_min']=flight_prc['Dep_Time'].apply(lambda x :x.split(':')[1])
flight_prc['Arrival_Time_hr']=flight_prc['Arrival_Time'].apply(lambda x :x.split(':')[0])
flight_prc['Arrival_Time_min']=flight_prc['Arrival_Time'].apply(lambda x :x.split(':')[1])
flight_prc.head()


# In[16]:


le=LabelEncoder()
flight_prc['Source_']=le.fit_transform(flight_prc['Source'])
flight_prc['Destination_']=le.fit_transform(flight_prc['Destination'])
flight_prc['Additional_Info_']=le.fit_transform(flight_prc['Additional_Info'])
flight_prc['Airline_']=le.fit_transform(flight_prc['Airline'])


# In[17]:


flight_prc.head()


# In[18]:


flight_prc['date']=flight_prc.Date_of_Journey.apply(lambda x:x.split('-')[0])
flight_prc['month']=flight_prc.Date_of_Journey.apply(lambda x:x.split('-')[1])
flight_prc['year']=flight_prc.Date_of_Journey.apply(lambda x:x.split('-')[2])
flight_prc.head()


# In[27]:


flight_prc.head()


# In[77]:


sns.pairplot(flight_prc)


# In[19]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score


# In[20]:


flight_prc.columns


# In[21]:


x=flight_prc.iloc[:,[8,11,12,13,14,15,16,17,18,19,20,21]]
y=flight_prc.iloc[:,10]


# In[22]:


x.head()


# In[23]:


rfg=RandomForestRegressor(n_estimators=700)


# In[26]:


cros=cross_val_score(rfg,x,y,scoring='neg_mean_squared_error',cv=10)
cros


# In[27]:


cros.mean()


# In[1]:


from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor


# In[29]:


grad_boost=GradientBoostingRegressor(n_estimators=300)
cros=cross_val_score(rfg,x,y,scoring='r2',cv=10)
cros


# In[30]:


cros.mean()


# In[31]:


from sklearn.model_selection import RandomizedSearchCV


# In[2]:


param={'n_estimators':[100,200,400,600],
      'learning_rate':[0.1,0.15,0.2],
      'min_samples_split':[150,200,500],
      'min_samples_leaf':[50,100],
      'max_depth':[2,4,6],
      'max_features':'sqrt'}


# In[34]:


gscv=RandomizedSearchCV(grad_boost,param_distributions=param,scoring='r2',cv=8,n_jobs=-1)
gscv.fit(x,y)


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.25)


# In[40]:


rfg.fit(x_train,y_train)
pred=rfg.predict(x_test)


# In[110]:


mean_squared_error(y_test,pred)
mean_absolute_error(y_test,pred)


# In[45]:


import pickle


# In[47]:


pickle.dump(rfg,open('rfg.pkl','wb'))
rfg=pickle.load(open('rfg.pkl','rb'))


# In[48]:


rfg.predict([[1,2,3,7,11,8,2019,145]])

