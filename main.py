import os
import time 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = './tmp/data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train.csv')
TEST_DIR = os.path.join(BASE_DIR, 'test.csv')


class TitanicSurvival():
    def __init__(self):
        pass 


    def wrangle(self, path):
        df = pd.read_csv(path).set_index("PassengerId")

        # Drop high-cardinality categorical features
        df = df.drop(columns=["Name", "Ticket"], inplace=True)

        # Fill NaN values
        df["Age"].fillna(df["Age"].median(), inplace=True)
        df["Embarded"].fillna("S", inplace=True)
        

        return df


if __name__ == '__main__':
    titanic_survival = TitanicSurvival()

    train = titanic_survival.wrangle(TRAIN_DIR)
    # test = titanic_survival.wrangle(TEST_DIR)
