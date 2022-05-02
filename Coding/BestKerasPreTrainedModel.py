# Author: ArpiHunanyan
# Created: 1 May,2022, 8:00 PM
# Email: arpi_hunanyan@edu.aua.am

import sys
import pandas as pd
from tqdm import tqdm
from  DataPreparation import getValidation, getTrain
from Model import kerasModelNames, Classifier

import matplotlib.pyplot as plt
from  Visualisation import Plot






# model names
model_names = kerasModelNames()


# Download the training and validation data
train_data =  getTrain(18)
validation_data =  getValidation(18)

# Number of training examples and labels
batch_size = 16
input_shape=(224, 224,3)
num_train = 86744
num_validation = 10954 
num_classes = 7
num_iterations = int(num_train/batch_size)

# Print important info
print(f'Num train images: {num_train} \
        \nNum validation images: {num_validation} \
        \nNum classes: {num_classes} \
        \nNum iterations per epoch: {num_iterations}')






# Loop over each model available in Keras
model_benchmarks = {'model_name': [], 'num_model_params': [], 'validation_accuracy': [] , 'val_recall': [], 'val_specificity_at_sensitivity': []}


num = 1
for model_name in tqdm(model_names):

    #  "NASNetLarge" requires input images with size (331,331)
    if 'NASNetLarge' in model_name:
        continue 

    print("------------------------------------------------")
    print("N: ", num)
    print("Model Name: ", model_name)
    num = num + 1
    print("------------------------------------------------")
        
    clf_model = Classifier( modelName = model_name)
    # unfreazing
    clf_model.unfreazeLastLayers(5);
    clf_model.compile()
    history = clf_model.fit(train_data, epochs = 3, validation_data = validation_data)
    
    # Calculate all relevant metrics
    model_benchmarks['model_name'].append(model_name)
    model_benchmarks['num_model_params'].append(clf_model.baseModelParamsCount())
    model_benchmarks['validation_accuracy'].append(history.history['val_accuracy'][-1])
    model_benchmarks['val_recall'].append(history.history['val_recall'][-1])
    model_benchmarks['val_specificity_at_sensitivity'].append(history.history['val_specificity_at_sensitivity'][-1])



benchmark_df = pd.DataFrame(model_benchmarks)
benchmark_df.sort_values(['validation_accuracy', 'val_recall', 'val_specificity_at_sensitivity'], inplace = True)
benchmark_df.to_csv('benchmark_df.csv', index = False)

#plot = Plot()
# plot.parameterBar()
# plot.metricBar()
#plot.scaterplotWithParameters()
