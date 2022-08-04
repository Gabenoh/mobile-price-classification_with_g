import pandas as pd

from sklearn.model_selection import train_test_split
from settings.constants import TRUE_COL
data = pd.read_csv('../kangle_data/train.csv')

train_x, test_x, train_y, test_y = train_test_split(data[TRUE_COL], data.price_range)

train_x.to_csv('train_x.csv', index=False)
train_y.to_csv('train_y.csv', index=False)
test_x.to_csv('test_x.csv', index=False)
test_y.to_csv('test_y.csv', index=False)
