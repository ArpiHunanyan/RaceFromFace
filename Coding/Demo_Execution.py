# # Author: ArpiHunanyan
# # Created: 14 May,2022, 9:00 PM
# # Email: arpi_hunanyan@edu.aua.am

from setuptools_scm import Configuration
from tensorflow import keras
from  DataPreparation import getValidation, getTrain
from Model import Classifier
import sys
import json
import os

# Set up___________________________________________________________________________________________________________________________________________________
print()
print("Demo.py started execution...")

print()
print('Do you want to generate new model?')
print("Options : True, False")
generateNewModel = True if input() == "True" else False

if(generateNewModel):
     print()
     print('Enter modelName')
     print("Options: InceptionV3, InceptionResNetV2, NASNetMobile, Xception, MobileNet, ResNet152V2, MobileNetV2, DenseNet169, DenseNet201, VGG19, ResNet101V2, DenseNet121,ResNet50V2, VGG16, EfficientNetB3, EfficientNetB4, EfficientNetB2, EfficientNetB0, EfficientNetB1, ResNet50, EfficientNetB6, MobileNetV3Small, EfficientNetB5, MobileNetV3Large, EfficientNetB7, ResNet152, ResNet101, EfficientNetV2B1, EfficientNetV2S, EfficientNetV2M, EfficientNetV2B2, EfficientNetV2B0, EfficientNetV2B3, EfficientNetV2L")
     modelName = input()   


else:
     print()
     print('Enter  modelName')
     print("Options:  ResNet50, MobileNetV3Large")
     modelName = input() 


     print('Enter the path of the existic model:')
     print('Enter the path of the existic model:')
     path = ["Model/3.MobileNetV3Large/Training", 
          "Model/3.MobileNetV3Large/Tuning", 
          "Model/4.ResNet50/Training", 
          "Model/4.ResNet50/Tuning", 
          "Model/5.MobileNetV3Large_cont/Tuning", 
          "Model/6.MobileNetV3LargeMasked/Tuning"]

     print("Options:")

     _ = [print(str ) for str in path if modelName in str]
     path_running_model = input()

print()
print('Do you want to inclued masked images ?')
print("Options : True, False")
trainMask = True if input() == "True" else False
valMask = trainMask

print()
print('How many images to inclued in training data set(maximum 86744)?')
trainM = float(input())

print()
print('How many images to inclued in validation data set(maximum 10954)?')
valM = float(input())

print()
print("Training Configuration")

print("How many of the top layes should be in training mode? ")
unfreezeedLayersTraining = int(input())
print("Enter the learning rate ")

learningRateTrainig = float(input())
print("How many epochs?")
epochsTraining = int(input())

print()
print("Fine-Tuning Configuration")

print("Enter the learning rate? ")
learningRateFineTuning  = float(input())

print("How many epochs?")
epochsFineTuning = int(input())
print()

# New path genaration _________________________________________________________________________________________________________________________________

MaskFlag = "Masked" if trainMask else ""

path_model = "Model/" + "new"+ modelName + MaskFlag
path_results = "Results/" + "new" + modelName + MaskFlag

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

# Data Preperation_________________________________________________________________________________________________________________________________________________________

print()
print("Data Preperation...")
print()

tarin_data = getTrain( M = trainM, masked = trainMask )
validation_data  = getValidation( M = valM, masked =  valMask)

# Model generation_________________________________________________________________________________________________________________________________________________________

if ( generateNewModel ):
     model = Classifier( modelName = modelName , createModel = True)
else :    
     model = Classifier( modelName = modelName , createModel = False, path = path_running_model)
 

# ----------------------------------------------------------------------------------: Training

print()
print("Training the top layer/s (for classifaction)  ")
print()

model.unfreazeLastLayers(unfreezeedLayersTraining)  
model.compile(optimizer = keras.optimizers.Adam(learningRateTrainig)) 

print()

trainingFirst = model.fit(tarin_data,   epochs = epochsTraining, validation_data = validation_data)

   
with open(path_results + "/Training", 'w') as convert_file:
     convert_file.write(json.dumps(trainingFirst.history))
print()
print("History is suuccsesfuly saved.")
print()

os.mkdir(path_model + "/Training")
model.save(path_model + "/Training");
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

