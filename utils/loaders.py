import pickle
import os

#from keras.datasets import mnist, cifar100,cifar10
#from keras.preprocessing.image import ImageDataGenerator, load_img, save_img, img_to_array

#import pandas as pd

#import numpy as np
#from os import walk, getcwd
#import h5py

#import scipy
#from glob import glob

#from keras.applications import vgg19
#from keras import backend as K
#from keras.utils import to_categorical

#import pdb




def load_model(model_class, folder):
    
    with open(os.path.join(folder, 'params.pkl'), 'rb') as f:
        params = pickle.load(f)

    model = model_class(*params)

    model.load_weights(os.path.join(folder, 'weights/weights.h5'))

    return model





