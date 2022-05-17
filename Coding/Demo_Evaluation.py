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
path_running_model = "Model/newMobileNetV3Large/Tuning"

valMask = True
valM =  10954
########################################### set up

MaskFlag = "Masked" if valMask else ""
path_results = "Results/" + "new" + modelName  + MaskFlag + "Evaluation"


if os.path.exists(path_results) :
     print()
     print("Clean ", path_results)
     print()
     sys.exit(0)

os.mkdir(path_results)


print()
print("Data Preperation")
print()



validation_data  = getValidation(M = valM, masked = valMask )

model = Classifier( modelName = modelName , createModel = False, path = path_running_model)

# ----------------------------------------------------------------------------------: Training

print()
print("Evaluations")
print()


results = model.evaluate(validation_data)

print ( results)

with open(path_results + "/Evaluation", 'w') as convert_file:
     convert_file.write(json.dumps(results))

print()
print("History is suuccsesfuly saved in " + path_results + "/Evaluation")
print()