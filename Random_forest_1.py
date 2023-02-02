#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
from glob import glob # this will help us download the data in order to visualize it
#These functions will be vital in making the CNN
import tensorflow as tf 
from tensorflow import keras
from keras import models
from keras import layers
#For some image processing
import keras.preprocessing.image as kpi
#This will be important for processing the metadata
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error


# In[2]:


#Starting with down loading the meta data
data_location = '/fs/ess/PAS2038/PHYSICS5680_OSU/student_data/armitage'
Meta = pd.DataFrame()
Meta_name = data_location + '/train.csv'
Meta = pd.read_csv(Meta_name,header=0)


# In[3]:


Meta['Random'] = np.random.randint(0,1000,size = len(Meta))


# In[4]:


Meta_data = Meta.iloc[:,1:13]
Meta_data['Random'] = Meta['Random']
Meta_data['Pawpularity'] = Meta['Pawpularity']
Meta = Meta_data
Meta


# In[5]:


from sklearn.model_selection import train_test_split
#Splitting into X(Features) Y(Pawpularity)
X_Meta = Meta.iloc[:,1:13].to_numpy()
Y_Meta = Meta['Pawpularity']


X_train,X_test,Y_train,Y_test = train_test_split(X_Meta,Y_Meta.values, test_size=0.2, shuffle = True)


# In[6]:


X = Meta.iloc[:,1:13]
X


# In[7]:


Y_test = np.array(Y_test)
Y_train = np.array(Y_train)


# In[8]:


X_train


# In[9]:


from sklearn import metrics
from sklearn.metrics import auc

def calc_performance_multi(true_labels,predicted_labels):
#
# Get the total number of unique labels
    num_labels = len(set(true_labels))
    confusionMatrix = nested_defaultdict(int,num_labels)
#
# Initialize all columns and rows
    for true_class in set(true_labels):
        for pred_class in set(true_labels):
            confusionMatrix[true_class][pred_class] = 0
# 
# Now calculate the confusion matrix
    for i in range(len(predicted_labels)):
        true_class = true_labels[i]      # this is either 0 or 1
        pred_class = predicted_labels[i]
        confusionMatrix[true_class][pred_class] += 1
#
# Get the recall, precision, ands F1 for each individual label
# - return both the "string report" (which you can print)
# - and the "dictionary report" (which you can use for averages and so on)
    report = classification_report(true_labels,predicted_labels)
    report_dict = classification_report(true_labels,predicted_labels,output_dict=True)
#
    results = {"confusionMatrix":confusionMatrix,"report":report,"report_dict":report_dict}
    return results


# In[10]:


def run_fitter_multi(estimator,X_train,y_train,X_test,y_test):
#
# Now fit to our training set
    estimator.fit(X_train,y_train)
#
# Now predict the classes and get the score for our traing set
    y_train_pred = estimator.predict(X_train)
    #y_train_score = estimator.predict_prob(X_train)[:,1]   # NOTE: some estimators have a predict_prob method instead od descision_function
#
# Now predict the classes and get the score for our test set
    y_test_pred = estimator.predict(X_test)
    #y_test_score = estimator.predict_prob(X_test)[:,1]

#
# Now get the performance
    results_test = calc_performance_multi(y_test,y_test_pred)
    results_train = calc_performance_multi(y_train,y_train_pred)
#
    return results_train,results_test


# # Doing the random forest regressor here, but running to check max depth

# In[11]:


from sklearn.ensemble import RandomForestRegressor

dfError = []


for depth in range (5,20):
    estimator = RandomForestRegressor(n_estimators=100,random_state=42,max_depth= depth)
    estimator.fit(X_train,Y_train)
    
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    
    train_RMSE = np.sqrt(mean_squared_error(Y_train,y_train_pred))
    test_RMSE = np.sqrt(mean_squared_error(Y_test,y_test_pred))
    dfError.append({'max_depth':depth,
                             'train_RMSE':train_RMSE,
                             'test_RMSE':test_RMSE
                             })


# In[12]:


df = pd.DataFrame(dfError)


# In[13]:


df


# In[14]:


X_train


# In[15]:


#dffeatures['Features']= pd.DataFrame(['Eyes','Face','Near','Action','Accessory','Group','Collage','Human','Occlusion','Info','Blur','Random'])


# In[16]:


#dffeatures["Importance"] = estimator.feature_importances_


# In[17]:


#dffeatures


# import matplotlib.pyplot as plt
# 
# #plt.plot(X_train,Y_train)

# In[18]:

# In[19]:


#Now lets load in the features 
features_location = '/users/PAS2038/robinson2407/osc_classes/PHYSICS5680_OSU/materials/Project '
Pet_features = pd.DataFrame()
Features_name = features_location + '/Pet_Features.csv'
Features = pd.read_csv(Features_name,header=None)


# In[20]:


Features


# In[21]:


#Now lets load in the shuffled meta data 
Shuffled_meta_location = '/users/PAS2038/robinson2407/osc_classes/PHYSICS5680_OSU/materials/Project '
Shuffled_meta = pd.DataFrame()
Shuffled_meta_name = Shuffled_meta_location + '/Shuffled_Meta_Data.csv'
Shuffled_Meta_DF = pd.read_csv(Shuffled_meta_name,header=None, names = ['ID', 'Subject Focus','Eyes','Face','Near','Action','Accessory','Group','Collage','Human','Occlusion','Info','Blur','Pawpularity','Random','Location' ])


# In[22]:


Shuffled_Meta_DF


# In[23]:


Pet_features = Features


# In[24]:


# add a collumn for pawpularity 
Pet_features['Pawpularity'] = Shuffled_Meta_DF['Pawpularity'].values


# In[25]:


Pet_features


# In[26]:


X_Feat = Pet_features.iloc[:,1:4096].to_numpy()
Y_Feat = Pet_features['Pawpularity']


X_train,X_test,Y_train,Y_test = train_test_split(X_Feat,Y_Feat.values, test_size=0.2, shuffle = True)


# In[27]:


Y_test = np.array(Y_test)
Y_train = np.array(Y_train)


# In[ ]:


df_feat_Error = []


for depth in range (5,20):
    print(depth)
    estimator = RandomForestRegressor(n_estimators=100,random_state=42,max_depth= depth)
    estimator.fit(X_train,Y_train)
    
    print('estimator has been defined and fit')
    
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)
    
    print('Predictions made')
    
    train_RMSE = np.sqrt(mean_squared_error(Y_train,y_train_pred))
    test_RMSE = np.sqrt(mean_squared_error(Y_test,y_test_pred))
    df_feat_Error.append({'max_depth':depth,
                             'train_RMSE':train_RMSE,
                             'test_RMSE':test_RMSE
                             })
    print('Results:  ',
          'train_RMSE:  ',train_RMSE,
                             'test_RMSE:  ', test_RMSE)
# In[ ]:


Final_df = pd.DataFrame(df_feat_Error)


# In[ ]:

Other_df = pd.DataFrame()

Other_df["Importance"] = estimator.feature_importances_


# In[ ]:


fig = px.histogram(Other_df,x='Features', y="Importance",
                      title='Feature Importance')
fig.show()

