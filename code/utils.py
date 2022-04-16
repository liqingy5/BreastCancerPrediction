import os
import random

_random_seed_ = 5
random.seed(_random_seed_)

def load_standard_data():
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
        
    pth = "../data/train_image.csv"
   

    data = read_csv(pth)
    y, X = data['label'], data['image_array']
    return train_test_split(X, y, test_size=0.33, random_state=_random_seed_)