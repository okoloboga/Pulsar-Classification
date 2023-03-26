#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from keras import models
from keras import layers

get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('/kaggle/input/playground-series-s3e10/train.csv')
train.head(10)


# Extract labels, delet Id column from dataset
# Scaling dataet

# In[2]:


label = train['Class']

del train['id']
del train['Class']

scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train),columns = train.columns)

train.head(5)


# Let's see what is Pulsars

# In[4]:


colors = {1: 'orange', 0: 'grey'}

def visual(df):
    
    scatter_matrix = pd.plotting.scatter_matrix(
        df, alpha = 0.5, figsize = (25, 25),
        grid = True, marker = '.',
        c = label.map(colors)) 

    for ax in scatter_matrix.ravel():
        ax.set_xlabel(ax.get_xlabel(), fontsize = 20, rotation = 45)
        ax.set_ylabel(ax.get_ylabel(), fontsize = 20, rotation = 45)
    
    return scatter_matrix

visual(train)


# Dimensionality reduction by concatenating EK and Skewness, also their DMSNR Curves
# 
# 

# In[5]:


SDC = np.array(train['Skewness_DMSNR_Curve']).reshape(-1, 1)
EDC = np.array(train['EK_DMSNR_Curve']).reshape(-1, 1)

EK = np.array(train['EK']).reshape(-1, 1)
SK = np.array(train['Skewness']).reshape(-1, 1)

SDC_EDC = np.concatenate((SDC, EDC), axis=1)
EK_SK = np.concatenate((EK, SK), axis=1)

sdc_edc = PCA(n_components = 1).fit(SDC_EDC)
curves = sdc_edc.transform(SDC_EDC)

ek_sk = PCA(n_components = 1).fit(EK_SK)
eksk = ek_sk.transform(EK_SK)

curves = pd.DataFrame(curves, columns = ['Curves'])
eksk = pd.DataFrame(eksk, columns = ['EKSK'])

del train['Skewness_DMSNR_Curve']
del train['EK_DMSNR_Curve']
del train['EK']
del train['Skewness']

train_new = pd.concat([train, curves, eksk], axis = 1)
train_new.head(10)


# Let's see new picture of that Pulsars

# In[6]:


visual(train_new)


# And some of them in 3D

# In[7]:


class scatter_3D:
    
    def __init__(self, x, y, z):
        
        self.x = x
        self.y = y
        self.z = z
        
    def show(self):
        
        x = self.x
        y = self.y
        z = self.z
        
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d') 
        ax.scatter(x, y, z, alpha = 0.2,
               c = label.map(colors)) 
        ax.set_xlabel('X label') 
        ax.set_ylabel('Y label') 
        ax.set_zlabel('Z label')
        
        return plt.show()


# In[8]:


pack_0 = scatter_3D(train_new['Mean_Integrated'],
                    train_new['SD'], 
                    train_new['Mean_DMSNR_Curve'])
pack_0.show()


# In[9]:


pack_1 = scatter_3D(train_new['SD_DMSNR_Curve'],
                    train_new['SD'], 
                    train_new['Mean_DMSNR_Curve'])
pack_1.show()


# In[10]:


pack_3 = scatter_3D(train_new['SD_DMSNR_Curve'],
                    train_new['EKSK'], 
                    train_new['Curves'])
pack_3.show()


# Oh, i guess, i find difficult pair,
# I will test my models on that pair.
# But, at first, i'll split dataset

# In[11]:


x_train, x_test, y_train, y_test = train_test_split(train_new, label, 
                                                    test_size = 0.2, random_state = 17)
x_train


# In[12]:


sub_train = pd.DataFrame(x_train, columns = ['Mean_DMSNR_Curve', 'SD_DMSNR_Curve'])
sub_test = pd.DataFrame(x_test, columns = ['Mean_DMSNR_Curve', 'SD_DMSNR_Curve'])

b = plt.scatter(sub_test['Mean_DMSNR_Curve'], sub_test['SD_DMSNR_Curve'],
            s = 5, alpha = 0.7, marker = 'o', c = y_test)
plt.show(b)


# Oh yeah, it looks like fontain, let test model  on that spread

# In[17]:


def result(model):
    
    pred = model.predict(sub_test)
    accuracy = log_loss(y_test, pred)
    print(f':loss sub_test {accuracy}')

    a = plt.scatter(sub_test['Mean_DMSNR_Curve'], sub_test['SD_DMSNR_Curve'],
            s = 5, alpha = 0.7, marker = 'o', c = pred)
    plt.show(a)

    b = plt.scatter(sub_test['Mean_DMSNR_Curve'], sub_test['SD_DMSNR_Curve'],
            s = 5, alpha = 0.7, marker = 'o', c = y_test)
    plt.show(b)


# First - CatBoostClassifier

# In[18]:


model_0 = CatBoostClassifier(
    iterations = 618, 
    learning_rate = 0.345,
    random_seed = 43,
    loss_function = 'Logloss'
)

model_0.fit(
    sub_train, y_train,
    eval_set = (sub_test, y_test),
    verbose = False,
    )

result(model_0)


# Now, classic KNeighbors

# In[19]:


model_1 = KNeighborsClassifier(n_neighbors=13).fit(sub_train, y_train)

result(model_1)


# RandomForest

# In[20]:


model_2 = RandomForestClassifier(n_estimators=285,     
                               criterion='gini',
                               max_depth=17,                    
                               max_features=None,
                               min_weight_fraction_leaf=0.001,
                               random_state=1337,
                               ).fit(sub_train, y_train)

result(model_2)


# Ok, of that 3 guys, Random forest is the best, but...
# Lets test DenseNN, i belive in that

# In[43]:


model = keras.Sequential([
        layers.Dense(256, activation = "relu"),
        layers.Dropout(0.25),
        layers.Dense(128, activation = "relu"),
        layers.Dropout(0.25),
        layers.Dense(64, activation = "relu"),
        layers.Dense(32, activation = "relu"),
        layers.Dense(16, activation = "relu"),
        layers.Dense(1, activation = "sigmoid")
    ])

model.compile(optimizer = "Adam",
             loss = "binary_crossentropy",
             metrics = ["accuracy"])

model.fit(sub_train, y_train,
          epochs = 10, batch_size = 512)
    
pred = model.predict(sub_test)
accuracy = log_loss(y_test, pred)
print(f':loss sub_test {accuracy}')

a = plt.scatter(sub_test['Mean_DMSNR_Curve'], sub_test['SD_DMSNR_Curve'],
                s = 5, alpha = 0.7, marker = 'o', c = pred)
plt.show(a)

b = plt.scatter(sub_test['Mean_DMSNR_Curve'], sub_test['SD_DMSNR_Curve'],
                s = 5, alpha = 0.7, marker = 'o', c = y_test)
plt.show(b)


# Good, let test it on full dataset

# In[45]:


model_f = keras.Sequential([
        layers.Dense(256, activation = "relu"),
        layers.Dropout(0.25),
        layers.Dense(128, activation = "relu"),
        layers.Dropout(0.25),
        layers.Dense(64, activation = "relu"),
        layers.Dense(32, activation = "relu"),
        layers.Dense(16, activation = "relu"),
        layers.Dense(1, activation = "sigmoid")
    ])

model_f.compile(optimizer = "Adam",
                loss = "binary_crossentropy",
                metrics = ["accuracy"])

model_f.fit(x_train, y_train,
          epochs = 10, batch_size = 512)
    
pred_f = model_f.predict(x_test)
accuracy_f = log_loss(y_test, pred_f)
print(f':loss sub_test {accuracy}')


# Good, let make prediction on test dataset

# In[42]:


#Preparing test data

test = pd.read_csv('/kaggle/input/playground-series-s3e10/test.csv')

del test['id']

scaler = StandardScaler()
test = pd.DataFrame(scaler.fit_transform(test),columns = test.columns)

SDC_T = np.array(test['Skewness_DMSNR_Curve']).reshape(-1, 1)
EDC_T = np.array(test['EK_DMSNR_Curve']).reshape(-1, 1)

EK_T = np.array(test['EK']).reshape(-1, 1)
SK_T = np.array(test['Skewness']).reshape(-1, 1)

SDC_EDC_T = np.concatenate((SDC_T, EDC_T), axis=1)
EK_SK_T = np.concatenate((EK_T, SK_T), axis=1)

sdc_edc_T = PCA(n_components = 1).fit(SDC_EDC_T)
curves_T = sdc_edc_T.transform(SDC_EDC_T)

ek_sk_T = PCA(n_components = 1).fit(EK_SK_T)
eksk_T = ek_sk_T.transform(EK_SK_T)

curves_T = pd.DataFrame(curves_T, columns = ['Curves'])
eksk_T = pd.DataFrame(eksk_T, columns = ['EKSK'])

del test['Skewness_DMSNR_Curve']
del test['EK_DMSNR_Curve']
del test['EK']
del test['Skewness']

test = pd.concat([test, curves_T, eksk_T], axis = 1)
test.head(10)


# Finally...

# In[73]:


prediction = model_f.predict(test)

submission = pd.read_csv('/kaggle/input/playground-series-s3e10/test.csv')
submission = pd.DataFrame(submission['id'])
Class = pd.DataFrame(data=prediction, columns = ['Class'])
submission = pd.concat([submission, Class], axis = 1)
submission.to_csv('/kaggle/working/submission.csv', index = False)
submission.head(5)

