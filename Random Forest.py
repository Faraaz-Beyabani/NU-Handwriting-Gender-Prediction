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

train_chain_data = train_data.iloc[:, 901:5020]
x = train_chain_data.to_numpy()


model = RandomForestClassifier()
model.fit(X=x, y=y)
answers = model.predict(X=x)
print((answers != y).sum())


chaincode_features = (train_data.iloc[:, 901:5020]).to_numpy()
test_idx = np.arange(3, 1128, 4)
train_idx = np.delete(np.arange(1128), test_idx)

x_test = chaincode_features[test_idx, :]
x_train = chaincode_features[train_idx, :]
y_train = np.repeat(train_ans.to_numpy(), 3)

y_test = train_ans
model.fit(X=x_train, y=y_train)


print(model.score(X=x_train, y=y_train))
print((model.predict(X=x_train) != y_train).sum())
print('\n')
print(model.score(X=x_test, y=y_test))
print((model.predict(X=x_test) != y_test).sum())
