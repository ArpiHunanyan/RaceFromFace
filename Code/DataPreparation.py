
from pkgutil import get_data
import pandas as pd
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#print(pd.__version__)
def getValidation():
    path = "Data/fairface_label_val.csv"
    suffix = "Data/val/" 
    return getData(path, suffix)

def getTrain():
    path = "Data/fairface_label_train.csv"
    suffix = "Data/train/" 
    return getData(path, suffix)

def getData(path, suffix):
    M = 30
    yy = pd.read_csv(path, nrows = M )
    y_index = [re.split('/', i)[1] for i in yy["file"]]

        
    data = np.empty((yy.shape[0], 224, 224, 3))
    index = np.array([])
        
    for i in range(yy.shape[0]):

        img = Image.open( suffix + y_index[i] )
        index = np.append(index, yy["file"][i]) 
        imageToMatrice = np.asarray(img)
        data[i] = imageToMatrice


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
        
    yy = yy.replace(cleanup_nums)
    y =  to_categorical(yy["race"])

    
    return (data, y)


def plotResaluts(training, critaria = 'accuracy', title = 'Accuracy:'):  
    plt.plot(training.history[critaria],  label = 'train')
    plt.plot(training.history['val_' + critaria], label = 'val') # shows overfitting
    plt.legend(title = title)
    plt.show()                                     
                
