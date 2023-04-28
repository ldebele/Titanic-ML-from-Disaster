import os
import time 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

BASE_DIR = './tmp/data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train.csv')
TEST_DIR = os.path.join(BASE_DIR, 'test.csv')


class TitanicSurvival():
    def __init__(self):
        self.df = None


    def wrangle(self, path):
        self.df = pd.read_csv(path).set_index("PassengerId")

        # Drop high-cardinality categorical features
        self.df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

        # Fill NaN values
        self.df["Age"].fillna(self.df["Age"].median(), inplace=True)
        self.df["Embarked"].fillna("S", inplace=True)
        
        # age
        self.df["Age"] = pd.cut(self.df["Age"], bins=5)

        # total family size
        self.df["TotFamily"] = self.df["Parch"] + self.df["SibSp"] + 1
        self.df.drop(columns=["SibSp", "Parch"], inplace=True)


        # OneHotEncoding
        self.label_encoding("Sex")
        self.label_encoding("Embarked")

        return self.df

    
    def label_encoding(self, item):
        encoder = LabelEncoder()
        self.df[item] = encoder.fit_transform(self.df[item])
        



if __name__ == '__main__':
    titanic_survival = TitanicSurvival()

    train = titanic_survival.wrangle(TRAIN_DIR)
    # test = titanic_survival.wrangle(TEST_DIR)
    print(train.head())
