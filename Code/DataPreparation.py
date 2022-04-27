
from this import d
import pandas as pd
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
import skimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def getValidation(M = 10954, age = False, gender = False, tensor = False):
    path_y = "Data/fairface_label_val.csv"
    path_data = "Data/validation_data.npy"
    return getOutput(path_y , path_data, M,  age, gender, tensor)

def getTrain(M = 86744, age = False, gender = False, tensor = False):
    path_y = "Data/fairface_label_train.csv"
    path_data = "Data/train_data.npy" 
    return getOutput(path_y , path_data, M,  age, gender, tensor)

def getOutput(path_y , path_data, M,  age, gender, tensor):

    if (tensor) :
        return tansorData(path = path_y, M = M) 
    print("I am using numpy.") 
    
    y = getLables(path_y, M)
    data = getData(path_data, M)
    inputArray = getInputArray(age, gender, M, path_y, data)


    if  len(inputArray) > 0 :
        return (inputArray, y)

    return (data, y)

def getInputArray(age, gender, M, path_y, data):
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
    return inputArray


def getLables(path_y, M):

    y = pd.read_csv(path_y, nrows = M , usecols = ["race"])
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
    return y

def getData(path_data, M):
        data = np.load(path_data, mmap_mode='r')
        data = data[: M]
        return data
    

def plotResaluts(training, critaria = 'accuracy', title = 'Accuracy:'):  
    plt.plot(training.history[critaria],  label = 'train')
    plt.plot(training.history['val_' + critaria], label = 'val') # shows overfitting
    plt.legend(title = title)
    plt.show() 

def tansorData(path, M):                                
            
    dataframe = pd.read_csv(path , usecols = ["file", "race"], header=0, nrows = M)
    y = pd.read_csv(path, usecols = ["race"], nrows = M)
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
            
    y = dataframe.replace(cleanup_nums)
    y =  to_categorical(y["race"])


    dataframe['East Asian'] = y[:,0]
    dataframe['White'] = y[:,1]
    dataframe['Latino_Hispanic'] = y[:,2]
    dataframe['Southeast Asian'] = y[:,3]
    dataframe['Black'] = y[:,4]
    dataframe['Indian'] = y[:,5]
    dataframe['Middle Eastern'] = y[:,6]


    train_generator = ImageDataGenerator().flow_from_dataframe(     
        dataframe = dataframe,  
        directory = "Data",
        x_col = "file", # name of col in data frame that contains file names
        y_col = ['East Asian', 'White', 'Latino_Hispanic', 'Southeast Asian', 'Black', 'Indian', 'Middle Eastern'] ,# name of col with labels
        batch_size = 16,
        # save_to_dir = "saveDir",
        target_size = (224, 224),
        class_mode = "raw" # for classification task
        )

    return train_generator




