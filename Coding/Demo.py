# Author: ArpiHunanyan
# Created: 1 May,2022, 8:00 PM
# Email: arpi_hunanyan@edu.aua.am

from unicodedata import name
from tensorflow import keras

from  DataPreparation import getValidation, getTrain
from Model import Classifier
import json
import os

# set up
unfreezeedLayersTraining = 10 # 10 from pre-traind model + classification layer
learningRateTrainig = 0.001
epochsTraining = 15


learningRateFineTuning = 1e-7
epochsFineTuning = 10



print()
print("Data Preperation")
print()
tarin_data = getTrain()
validation_data  = getValidation()


model = Classifier(modelName = EfficientNetV2L) #set pre-traind's layers freazed 


model.unfreazeLastLayers(unfreezeedLayersTraining)  # 10 layer 
model.compile(optimizer = keras.optimizers.Adam(learningRateTrainig)) # alpha 0.001

# model.summary()


print()

#  Train the top layer/s (for classifaction) 
print("Training the top layer/s (for classifaction)  ")
trainingFirst = model.fit(tarin_data,   epochs = epochsTraining, validation_data = validation_data)

  
with open('Results/trainingResultsEfficientnetV2_2', 'w') as convert_file:
     convert_file.write(json.dumps(trainingFirst.history))
print()
print("History is suuccsesfuly saved.")
print()
os.mkdir( "Model/TrainedModel" )
model.save("Model/TrainedModel");

print()
## Fine - Tuning
print("Fine - Tuning ")
model.setTrainable(True)
model.compile(optimizer = keras.optimizers.Adam(learningRateFineTuning)) # 1e-5 => 1e-7

# model
trainingSecond = model.fit(tarin_data , epochs =  epochsFineTuning, validation_data = validation_data)

with open('Results/tuningResultsEfficientnetV2_2', 'w') as convert_file:
     convert_file.write(json.dumps(trainingSecond.history))
print()
print("History is suuccsesfuly saved.")
print()

model.save("Model/TunedModel");