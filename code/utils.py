import os
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
import random
import numpy as np
import pandas as pd
_random_seed_ = 5
random.seed(_random_seed_)

def load_standard_data():
    import pickle
    from sklearn.model_selection import train_test_split
        
    image_path = "../data/images.npy"
    label_path = "../data/labels.npy"
   

    X = np.load(image_path)
    y = np.load(label_path)
    X = X/255.0
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=_random_seed_)

    return flatten(x_train),flatten(x_test), y_train, y_test


def flatten(data):
    x=[]
    for i in data:
        x.append(i.flatten())
    return x