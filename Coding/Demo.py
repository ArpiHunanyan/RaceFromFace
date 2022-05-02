# Author: ArpiHunanyan
# Created: 1 May,2022, 8:00 PM
# Email: arpi_hunanyan@edu.aua.am

from tensorflow import keras
from  DataPreparation import getValidation, getTrain
from Model import Classifier
import json


print("Data Preperation")

tarin_data = getTrain()
validation_data  = getValidation()


model = Classifier() #set pre-traind's layers freazed 


model.unfreazeLastLayers()  # 10 layer 
model.compile() # alpha 0.001

# model.summary()


print()

#  Train the top layer/s (for classifaction) 
print("Training the top layer/s (for classifaction)  ")
trainingFirst = model.fit(tarin_data,   epochs = 200, validation_data = validation_data)

  
with open('trainingResults', 'w') as convert_file:
     convert_file.write(json.dumps(trainingFirst.history))


print()
## Fine - Tuning
print("Fine - Tuning ")
model.setTrainable(True)
model.compile(optimizer = keras.optimizers.Adam(1e-7)) # 1e-5 => 1e-7

# model
trainingSecond = model.fit(tarin_data , epochs =  20, validation_data = validation_data)

with open('fineTunigResults', 'w') as convert_file:
     convert_file.write(json.dumps(trainingSecond.history))

model.save();