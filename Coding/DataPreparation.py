# Author: ArpiHunanyan
# Created: 29 April,2022, 14:58 PM
# Email: arpi_hunanyan@edu.aua.am

import os
import sys
import pandas as pd
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder

import numpy as np

def getValidation(M = 10954, masked = False):
    path = "Data/fairface_label_val.csv"
    return getOutput(path, M, masked)

def getTrain(M = 86744, masked = False):
    path = "Data/fairface_label_train.csv"
    return getOutput(path, M, masked)

def getOutput(path, M, masked):

    if masked :
        file = "file_masked"
    else :
        file = "file"

        
    dataframe = pd.read_csv(path , usecols = [file, "race"], header = 0, nrows = M)

    # for i in np.unique(dataframe.race):
    #     print(i, dataframe.loc[dataframe["race"] == i, file].count())
    #     print(dataframe.loc[dataframe["race"] == i, file].count())

       
    assert len(dataframe.race.unique()) == 7, 'InvalidRaceGroupSize'
    
    # encoding
    encoder = OneHotEncoder(handle_unknown = 'ignore')
    encoded = pd.DataFrame(encoder.fit_transform(dataframe[["race"]]).toarray())
    dataframe = dataframe.join(encoded)
    #clean up
    dataframe.drop("race", axis = 1, inplace = True)
    dataframe.columns = [file, 'Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White'] # alphabetically


    # DataFrameIterator
    sys.stdout = open(os.devnull, 'w')
    data = ImageDataGenerator().flow_from_dataframe(dataframe = dataframe,  
                                                    directory = 'Data',
                                                    x_col = file, 
                                                    y_col =  [ 'Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White'], 
                                                    batch_size = 16,
                                                    target_size = (224, 224),
                                                    class_mode = 'raw')

    sys.stdout = sys.__stdout__


    return data








