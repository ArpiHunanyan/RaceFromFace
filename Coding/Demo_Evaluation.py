# # Author: ArpiHunanyan
# # Created: 14 May,2022, 9:00 PM
# # Email: arpi_hunanyan@edu.aua.am

from tensorflow import keras
from  DataPreparation import getValidation
from ResultInterpretation import Execution
from Model import Classifier
import sys
import json
import os




# Set up___________________________________________________________________________________________________________________________________________________

print()
print("Demo_Evaluation.py started execution...")
print()


print('Enter  modelName')
print("Options:  ResNet50, MobileNetV3Large")
modelName = input() 


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
valMask = True if input() == "True" else False


print()
print('How many images to inclued in validation data set(maximum 10954)?')
valM = float(input())


# New path genaration _________________________________________________________________________________________________________________________________
MaskFlag = "Masked" if valMask else ""
path_results = "Results/" + "new" + modelName  + MaskFlag + "Evaluation"


if os.path.exists(path_results) :
     print()
     print("Clean ", path_results)
     print()
     sys.exit(0)

os.mkdir(path_results)


# Data Preperation_________________________________________________________________________________________________________________________________________________________
print()
print("Data Preperation...")
print()

validation_data  = getValidation(M = valM, masked = valMask )

# Model generation_________________________________________________________________________________________________________________________________________________________
model = Classifier( modelName = modelName , createModel = False, path = path_running_model)

# ----------------------------------------------------------------------------------: Evaluation

print()
print("Evaluations")
print()


results = model.evaluate(validation_data)



with open(path_results + "/Evaluation", 'w') as convert_file:
     convert_file.write(json.dumps(results))

print()
print("History is suuccsesfuly saved in " + path_results + "/Evaluation")
print()


# demostrating results
exicution = Execution(path = path_results + "/Evaluation")
exicution.lastValues(matric = ['loss', 'accuracy',  'recall', 'precision',  'specificity_at_sensitivity'] )