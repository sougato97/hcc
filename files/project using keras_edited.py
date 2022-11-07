# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:09:26 2018

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:02:42 2018

@author: USER
"""

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.cross_validation import train_test_split


#Reading the dataset
hcc = pd.read_csv("hcc-data.csv")

#Oversampling the data 2 times
hcc = pd.concat([hcc]*2, ignore_index = True)

#Handling the values for continuous attributes
continuous_x = hcc.iloc[:,23:49]
continuous_x.head()
y_value = hcc.iloc[: , 49]

#type(y_value)
imputer = Imputer(missing_values = "NaN" ,strategy = 'mean', axis = 0)  
imputer = imputer.fit(continuous_x)
continuous_x = imputer.transform(continuous_x)
continuous_x = pd.DataFrame(data = continuous_x)
continuous_x = continuous_x.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#continuous_x.head()
continuous_x.columns = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

#Handling the values for nominal values
nom_x = hcc.iloc[:,0:23]
nom_x.head()
imputer = Imputer(missing_values = "NaN" ,strategy = 'most_frequent', axis = 0) 
imputer = imputer.fit(nom_x) 
nom_x = imputer.transform(nom_x)
nom_x = pd.DataFrame(data = nom_x)
nom_x.columns = ['aa','ab','ac','ad','ae','af','ag','ah','ai','aj','ak','al','am','an','ao','ap','aq','ar','as','at','au','av','aw']
nom_x.head()
frames = [nom_x,continuous_x]
x_data = pd.concat(frames, axis=1)
x_data.head()
X_train, X_test, y_train, y_test = train_test_split(x_data,y_value,test_size=0.3, random_state=100)

X_train.columns = ['aa','ab','ac','ad','ae','af','ag','ah','ai','aj','ak','al','am','an','ao','ap','aq','ar','as','at','au','av','aw','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
X_test.columns = ['aa','ab','ac','ad','ae','af','ag','ah','ai','aj','ak','al','am','an','ao','ap','aq','ar','as','at','au','av','aw','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#X_train.columns

'''
#Oversampling the X_train 

result = pd.concat([X_train, y_train], axis = 1)
result = result.sample(frac=1) #randamizing the result
'''
'''
The frac keyword argument specifies the fraction of rows to return in the random sample, 
so frac=1 means return all rows (in random order).
'''
'''
result = pd.concat([result]*10, ignore_index = True)
#repeat the values and not the index
result = result.sample(frac = 1) #again randamizing the result
X_train = result.iloc[: , 0:49]
y_train = result.iloc[: , 49]

'''
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu', input_dim = 49))
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 100)


#Creating the confusion matrix
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc = ((cm[0][0]+cm[1][1])/(cm[1][0]+cm[1][1]+cm[0][0]+cm[0][1]))*100
