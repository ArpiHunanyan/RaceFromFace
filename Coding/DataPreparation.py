# Author: ArpiHunanyan
# Created: 29 April,2022, 14:58 PM
# Email: arpi_hunanyan@edu.aua.am

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder

import numpy as np

def getValidation(M = 10954):
    path = "Data/fairface_label_val.csv"
    return getOutput(path, M)

def getTrain(M = 86744):
    path = "Data/fairface_label_train.csv"
    return getOutput(path, M)

def getOutput(path, M):
        
    dataframe = pd.read_csv(path , usecols = ["file", "race"], header = 0, nrows = M)

    # for i in np.unique(dataframe.race):
    #     print(i, dataframe.loc[dataframe["race"] == i, "file"].count())
    #     print(dataframe.loc[dataframe["race"] == i, "file"].count())

       
    assert len(dataframe.race.unique()) == 7, 'InvalidRaceGroupSize'
    
    # encoding
    encoder = OneHotEncoder(handle_unknown = 'ignore')
    encoded = pd.DataFrame(encoder.fit_transform(dataframe[["race"]]).toarray())
    dataframe = dataframe.join(encoded)
    #clean up
    dataframe.drop("race", axis = 1, inplace = True)
    dataframe.columns = ['file', 'Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White'] # alphabetically

    # DataFrameIterator
    data = ImageDataGenerator().flow_from_dataframe(dataframe = dataframe,  
                                                    directory = 'Data',
                                                    x_col = 'file', 
                                                    y_col =  [ 'Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White'], 
                                                    batch_size = 16,
                                                    target_size = (224, 224),
                                                    class_mode = 'raw')

    return data





