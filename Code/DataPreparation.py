
import pandas as pd
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
import skimage

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def getValidation(M = 10954, age = False, gender = False):
    path_y = "Data/fairface_label_val.csv"
    path_data = "Data/validation_data.npy"
    return getData(path_y , path_data, M,  age, gender)

def getTrain(M = 86744, age = False, gender = False):
    path_y = "Data/fairface_label_train.csv"
    path_data = "Data/train_data.npy" 
    return getData(path_y , path_data, M,  age, gender)

def getData(path_y , path_data, M, age, gender):

    usecols = ["race"]
    if  age :
        usecols.append('age')

    if  gender :
        usecols.append('gender')

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
    #data = skimage.color.rgb2gray(data)

    inputArray = []

    if  age :
        ageDF = pd.read_csv(path_y, nrows = M , usecols = ['age'])

        
        cleanup_nums = {"age": {
                        '0-2': 0,
                        '3-9': 1,
                        '10-19' : 2,
                        '20-29' : 3,
                        '30-39' : 4,
                        '40-49' : 5,
                        '50-59' : 6 ,
                        '60-69' : 7 ,
                        'more than 70' : 8,
                        }
        }
        
        ageDF = ageDF.replace(cleanup_nums)
        inputArray.append(data)
        inputArray.append(ageDF.values)


    
    if  gender :

        genderDF = pd.read_csv(path_y, nrows = M , usecols = ['gender'])

        
        cleanup_nums = {"gender": {
                        'Female': 0,
                        'Male': 1 }
        }
        
        genderDF = genderDF.replace(cleanup_nums)
        inputArray.append(genderDF.values)


    if  len(inputArray) > 0 :

        return (inputArray, y)

    return (data, y)


def plotResaluts(training, critaria = 'accuracy', title = 'Accuracy:'):  
    plt.plot(training.history[critaria],  label = 'train')
    plt.plot(training.history['val_' + critaria], label = 'val') # shows overfitting
    plt.legend(title = title)
    plt.show()                                     
                
