import os
import random

_random_seed_ = 5
random.seed(_random_seed_)

def load_standard_data():
    import pickle
    from sklearn.model_selection import train_test_split
        
    pth = "../data/train_images.pkl"
   

    data = pickle.load(open(pth, "rb"))
    y, X = data['labels'], data['images']
    return train_test_split(X, y, test_size=0.33, random_state=_random_seed_)