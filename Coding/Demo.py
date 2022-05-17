# # Author: ArpiHunanyan
# # Created: 14 May,2022, 9:00 PM
# # Email: arpi_hunanyan@edu.aua.am

from tensorflow import keras
from  DataPreparation import getValidation, getTrain
from Model import Classifier
import sys
import json
import os




########################################### set up
modelName = "MobileNetV3Large"
path_running_model = "Model/newMobileNetV3LargeMasked/Tuning"

trainMask = True
valMask = trainMask


trainM = 86744
valM = 10954

# # training
# unfreezeedLayersTraining = 10 # 10 from pre-traind model + classification layer
# learningRateTrainig = 0.001
# epochsTraining = 1

# tuning
learningRateFineTuning = 1e-3
epochsFineTuning = 200




########################################### set up
MaskFlag = "Masked" if trainMask else ""

path_model = "Model/" + "new2"+ modelName + MaskFlag
path_results = "Results/" + "new2" + modelName + MaskFlag

if os.path.exists(path_model) :
     print()
     print("Clean ", path_model)
     print()
     sys.exit(0)

if os.path.exists(path_results) :
     print()
     print("Clean ", path_results)
     print()
     sys.exit(0)

os.mkdir(path_model)
os.mkdir(path_results)


print()
print("Data Preperation")
print()


tarin_data = getTrain( M = trainM, masked = trainMask )
validation_data  = getValidation( M = valM, masked =  valMask)

model = Classifier( modelName = modelName , createModel = False, path = path_running_model)
# model = Classifier( modelName = modelName, path = "Model/TrainedModel" + modelName ) #set pre-traind's layers freazed 

# ----------------------------------------------------------------------------------: Training

# print()
# print("Training the top layer/s (for classifaction)  ")
# print()
# # model.unfreazeLastLayers(unfreezeedLayersTraining)  # 10 layer 
# # model.compile(optimizer = keras.optimizers.Adam(learningRateTrainig)) # alpha 0.001

# print()

# trainingFirst = model.fit(tarin_data,   epochs = epochsTraining, validation_data = validation_data)

   
# with open(path_results + "/Training", 'w') as convert_file:
#      convert_file.write(json.dumps(trainingFirst.history))
# print()
# print("History is suuccsesfuly saved.")
# print()

# os.mkdir(path_model + "/Training")
# model.save(path_model + "/Training");
# ----------------------------------------------------------------------------------: Tuning


## Fine - Tuning
print()
print("Fine - Tuning ")
print()

model.setTrainable(True)
model.compile(optimizer = keras.optimizers.Adam(learningRateFineTuning)) 

print()

trainingSecond = model.fit(tarin_data , epochs =  epochsFineTuning, validation_data = validation_data)

with open(path_results + "/Tuning", 'w') as convert_file:
     convert_file.write(json.dumps(trainingSecond.history))

print()
print("History is suuccsesfuly saved in " + path_results + "/Tuning")
print()

print()
os.mkdir(path_model + "/Tuning")
model.save(path_model + "/Tuning");
print()

