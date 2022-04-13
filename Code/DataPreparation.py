
import pandas as pd
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def getValidation(M = 10954 - 1):
    path_y = "Data/fairface_label_val.csv"
    path_data = "Data/validation_data.npy"
    return getData(path_y , path_data, M)

def getTrain(M = 86744 - 1):
    path_y = "Data/fairface_label_train.csv"
    path_data = "Data/train_data.npy" 
    return getData(path_y , path_data, M)

def getData(path_y , path_data, M):

    y = pd.read_csv(path_y, nrows = M , usecols = ["race"])
    print()
    assert len(y.race.unique()) == 7, 'InvalidRaceGroupSize'

    cleanup_nums = {"race": {
                                'East Asian': 0,
                                'White': 1,
                                'Latino_Hispanic' : 2,
                                'Southeast Asian' : 3,
                                'Black' : 4,
                                'Indian' : 5,
                                'Middle Eastern' : 6
                            }
                    }
        
    y = y.replace(cleanup_nums)
    y =  to_categorical(y["race"])
    data = np.load(path_data, mmap_mode='r', )
    data = data[: M]


    return (data, y)


def plotResaluts(training, critaria = 'accuracy', title = 'Accuracy:'):  
    plt.plot(training.history[critaria],  label = 'train')
    plt.plot(training.history['val_' + critaria], label = 'val') # shows overfitting
    plt.legend(title = title)
    plt.show()                                     
                
