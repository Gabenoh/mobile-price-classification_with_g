import pandas as pd
import sklearn
import numpy

from sklearn.preprocessing import LabelEncoder, minmax_scale
from settings.constants import TRUE_COL


class Encoder:
    def __init__(self):
        self.data = None

    def fit(self, data):
        print(data.head(10))
        self.data = pd.DataFrame(data[TRUE_COL])
        print(self.data.head(10))

    def transform(self):
        """
        :return: transform data
        """
        # cur or qcut data

        self.data.battery_power = pd.cut(self.data.battery_power, 10)

        self.data.n_cores = pd.cut(self.data.n_cores, 10)

        self.data.pc = pd.cut(self.data.pc, 10)

        self.data.ram = pd.cut(self.data.ram, 10)

        # encode labels

        La = LabelEncoder()

        La.fit(self.data.battery_power)
        self.data.battery_power = La.transform(self.data.battery_power)

        La.fit(self.data.n_cores)
        self.data.n_cores = La.transform(self.data.n_cores)

        La.fit(self.data.pc)
        self.data.pc = La.transform(self.data.pc)

        La.fit(self.data.ram)
        self.data.ram = La.transform(self.data.ram)

        # normalize data

        self.data.battery_power = minmax_scale(self.data.battery_power, axis=0)
        self.data.n_cores = minmax_scale(self.data.n_cores, axis=0)
        self.data.pc = minmax_scale(self.data.pc, axis=0)
        self.data.ram = minmax_scale(self.data.ram, axis=0)

        print(self.data.head(10))
        return self.data

