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


# example of loading the vgg16 model
from keras.applications.vgg16 import VGG16

#for layer in model.layers:
#    layer.trainable = False

# In[3]:


#Starting with down loading the meta data
data_location = '/fs/ess/PAS2038/PHYSICS5680_OSU/student_data/armitage'
Meta = pd.DataFrame()
Meta_name = data_location + '/train.csv'
Meta = pd.read_csv(Meta_name,header=0)


# In[4]:


#Then the image data
image_location = data_location + '/train/*.jpg'
images = glob(image_location)


# In[5]:


Meta['Random'] = np.random.randint(0,1000,size = len(Meta))


# In[6]:


#To test we want to consolidate these two into a single data frame
Meta['Path'] = images
print(Meta['Id'][0])
Meta['Path'][0]


# In[7]:


from sklearn.utils import shuffle
Meta_shuffled = shuffle(Meta, random_state=1)

np.savetxt("Shuffled_Meta_Data.csv", 
           Meta_shuffled,
           delimiter =", ", 
           fmt ='% s')
# In[8]:


#As we can see each of these above images are different sizes so we need to resize them
from PIL import Image

reshaped_images = []

for i in range(len(Meta_shuffled)):
    pillow_image = Image.open(Meta_shuffled['Path'][i])
    reshaped_images.append(pillow_image.resize((224,224)))


# In[9]:


image_array = []
for i in range(len(reshaped_images)):
    image_array.append(np.array(reshaped_images[i])/256)
    
image_array = np.array(image_array)

print('image array is now a numpy array, yay!')

# In[10]:


from sklearn.model_selection import train_test_split
#Splitting into X(Features) Y(Pawpularity)
X_Meta = Meta_shuffled.iloc[:,1:13]
Y_Meta = Meta_shuffled['Pawpularity']

print('Data has been shuffled, yay!')

X_train,X_test,Y_train,Y_test = train_test_split(X_Meta,Y_Meta.values, test_size=0.2, shuffle = False)
image_train,image_test,iy_train,iy_test = train_test_split(image_array,Y_Meta.values,test_size = 0.2, shuffle = False)

print('Data has been split into test and train, yay!')
# In[11]:


Meta_shuffled


# In[ ]:
#CUT- WE ARE NOT USING THIS AS A CLASSIFIER---------------------------------------------------------------------

#from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg16 import decode_predictions

# predict the probability across all output classes
#yhat = model.predict(image_array)
# convert the probabilities to class labels
#label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
#label = label[0][0]
#----------------------------------------------------------------------------------------------------------

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
from pickle import dump


# reshape data for the model
# image = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image_array)


# load model
model = VGG16(weights = 'imagenet')




#INSTEAD USING IT AS A FEATURE EXTRACTOR
# remove the output layer
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# get extracted features
features = model.predict(image)
print(features.shape)

print(model.summary())
# save to file
dump(features, open('dog.pkl', 'wb'))


print(features)

np.savetxt("Pet_Features.csv", 
           features,
           delimiter =", ", 
           fmt ='% s')
