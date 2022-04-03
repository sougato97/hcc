# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:00:18 2018

@author: USER
"""

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import Imputer
import numpy as np


hcc = pd.read_csv("hcc-data.csv")

hcc.head()

continuous_x = np.array(hcc.iloc[:,23:49])
y_value = np.asarray(hcc.iloc[: , 49])
imputer = Imputer(missing_values = "NaN" ,strategy = 'mean', axis = 0)  
imputer = imputer.fit(continuous_x[:,23:49])
continuous_x[:,23:49] = imputer.transform(continuous_x[:,23:49])
