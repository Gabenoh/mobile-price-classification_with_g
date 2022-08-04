import pandas as pd
import numpy as np
from utils import Encoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


train_x = pd.read_csv('data/train_x.csv')
train_y = pd.read_csv('data/train_y.csv')
test_x = pd.read_csv('data/test_x.csv')
test_y = pd.read_csv('data/test_y.csv')

# import data for kangle-test

'''
train_x = pd.read_csv('kangle_data/train.csv')
train_y = train_x.price_range

test_x = pd.read_csv('kangle_data/test.csv')
'''

# encode data
dec = Encoder()

dec.fit(train_x)
train_x = dec.transform()

dec.fit(test_x)
test_x = dec.transform()

# search best features for model

grid_table = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': range(2, 10),
    'min_samples_split': range(2, 5),
    'splitter': ['best', 'random'],
    'max_features': [None, 'sqrt', 'log2']
}
Gs = GridSearchCV(DecisionTreeClassifier(), grid_table)
Gs.fit(train_x, train_y)
print(Gs.best_params_)
# {'criterion': 'gini', 'max_depth': 5, 'max_features': None, 'min_samples_split': 2, 'splitter': 'best'}

model_tree = DecisionTreeClassifier(criterion=Gs.best_params_['criterion'], max_depth=Gs.best_params_['max_depth'],
                                    min_samples_split=Gs.best_params_['min_samples_split'],
                                    splitter=Gs.best_params_['splitter'], max_features=Gs.best_params_['max_features'])

# fit and predict model

model_tree.fit(train_x, train_y)
pred_model_tree = model_tree.predict(test_x)

test_y = test_y.to_numpy()
print('Accuracy tree: ', accuracy_score(test_y, pred_model_tree))
# Accuracy score - 0.8
