#!/usr/bin/env python
# coding: utf-8

# In[89]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[90]:


train=pd.read_csv("C:/Users/SAJAN P MENON/Desktop/week-9-case_study/Hackerearth_attretion rate_regression/Dataset/Train.csv")
print(train.shape)


# In[91]:


#combining both train and test for common data cleaning
df1=pd.concat([train,test],axis=0)
df=train.drop(columns=['Employee_ID'],axis=1)   #as employee_id is having completely unique values, we are dropping it.
df.shape


# In[92]:


#we shall impute the null value in "age" feature with its median.
df['Age']=df['Age'].fillna(df['Age'].median())
df['Pay_Scale']=df['Pay_Scale'].fillna(0)


# In[93]:


# we shall impute with median
df['Time_of_service']=df['Time_of_service'].fillna(df['Time_of_service'].median())


# In[94]:


# we shall flag the missing value rather than imputing with mode
df['VAR2']=df['VAR2'].fillna('miss_value')


# In[95]:


# we shall flag the missing value rather than imputing with mode
df['VAR4']=df['VAR4'].fillna('miss_value')


# In[96]:


# we shall flag the missing value rather than imputing with mode
df['Work_Life_balance']=df['Work_Life_balance'].fillna(-1)


# In[ ]:





# # Converting the datatype of the features as per the data inside it

# In[97]:


df[["Education_Level",'Post_Level','Time_since_promotion','Travel_Rate','VAR1','VAR3','VAR5','VAR6',"VAR7"]] = df[["Education_Level",'Post_Level','Time_since_promotion','Travel_Rate','VAR1','VAR3','VAR5','VAR6',"VAR7"]].astype('object')


# In[ ]:





# # Label Encoding/ dummies

# In[98]:


df5=df.copy()


# In[99]:


df5['VAR2']=df5['VAR2'].apply(lambda x: 0 if x=="miss_value" else x )
#df5['VAR3']=df5['VAR3'].astype("float64")


# In[ ]:





# In[100]:


df5=pd.get_dummies(data=df5,columns=['Decision_skill_possess','Education_Level','Gender','Hometown','Post_Level','Relationship_Status',
      'Time_since_promotion','Travel_Rate',"VAR1","VAR4","VAR5","VAR6","VAR3","VAR7","Compensation_and_Benefits",
        "Pay_Scale","Unit","Work_Life_balance"],drop_first=True)


# # Feature transformation

# In[101]:


#feature transformation was done as we observed better model performance.


# In[102]:


df5['Age']=np.sqrt(df5['Age'])
df5['Time_of_service']=np.sqrt(df5['Time_of_service'])


# In[103]:


y=df5['Attrition_rate']
x=df5.drop(columns=['Attrition_rate'])
print("x_train shape is ",x.shape)
print("y_train shape is ",y.shape)


# In[ ]:





# In[104]:



#from sklearn.ensemble import GradientBoostingRegressor
#gboost=GradientBoostingRegressor()
#gboost.fit(x,y)


# In[ ]:





# imp=pd.DataFrame(gboost.feature_importances_,index=x.columns,columns=['sign'])
# 
# sign_df=imp.sort_values(by="sign",ascending=False)
# sign_df1=sign_df.reset_index()
# sign_df1
# 
# sign_df_final=sign_df1[sign_df1['sign']>=0.010609]
# 

# In[105]:



selected_features=['growth_rate',
 'Age',
 'Time_of_service',
 'Education_Level_2',
 'Unit_Sales',
 'VAR2',
 'Compensation_and_Benefits_type4',
 'VAR7_2',
 'Pay_Scale_4.0',
 'Unit_Security',
 'Unit_Human Resource Management',
 'VAR6_6',
 'VAR1_5',
 'Pay_Scale_9.0',
 'Unit_R&D',
 'Post_Level_2',
 'Unit_Purchasing',
 'Work_Life_balance_2.0',
 'VAR3_0.7075',
 'Time_since_promotion_1',
 'Gender_M',
 'Compensation_and_Benefits_type1',
 'Pay_Scale_2.0','Attrition_rate']


# In[106]:



df_fs=df5[selected_features]
print(df_fs.shape)


# In[107]:


y=df_fs['Attrition_rate']
x=df_fs.drop(columns=['Attrition_rate'])
print("x_train shape is ",x.shape)
print("y_train shape is ",y.shape)


# In[108]:


from sklearn.ensemble import GradientBoostingRegressor
gboost=GradientBoostingRegressor(n_estimators=3)     #  we got n_estimators=3 from hyperparameter tuning
gboost.fit(x,y)


# In[109]:


import pickle
# Saving model to disk
pickle.dump(gboost, open('model.pkl','wb'))


# In[110]:



# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[ ]:




