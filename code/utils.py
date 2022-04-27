import os
from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
import random
import numpy as np
import pandas as pd
_random_seed_ = 5
random.seed(_random_seed_)

def load_standard_data(data_type="standard",isAlexnet=False):
    import pickle
    from sklearn.model_selection import train_test_split

    post_fix=""
    if isAlexnet:
        post_fix="_Alexnet"
        
    image_path = f"../data/images{post_fix}.npy"
    label_path = f"../data/labels{post_fix}.npy"
    if data_type == "oversampled":
        image_path = f"../data/images_oversampled{post_fix}.npy"
        label_path = f"../data/labels_oversampled{post_fix}.npy"
   

    X = np.load(image_path)
    y = np.load(label_path)
    X = X/255.0
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=_random_seed_)
    if not isAlexnet:
        return flatten(x_train),flatten(x_test), y_train, y_test
    else:
        return x_train, x_test, y_train, y_test


def flatten(data):
    x=[]
    for i in data:
        x.append(i.flatten())
    return np.array(x)