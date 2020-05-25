# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:47:52 2020

@author: du
"""


import numpy as np
import pandas as pd
from sklearn.ensamble import RandomForestClassifier

train_data = pd.read_csv('./train.csv')
train_ans = pd.read_csv('./train_answers.csv')

train_ans = train_ans.iloc[:,1]
y = np.repeat(train_ans.to_numpy(), 4)
Tortuosity features


train_tort_data = train_data.iloc[:, 4:44]

x = train_tort_data.to_numpy()

print(np.shape(x))
print(np.shape(y))

print(f'\nThere are {(y==0).sum()} male writers.')
print(f'\nThere are {(y==1).sum()} male writers.')


model = RandomForestClassifier()
model.fit(X=x, y=y)

answers = model.predict(X=x)

print((answers != y).sum())
print(answers[:32])
print(y[:32])