from numpy import matrix
from tensorflow import keras
from  DataPreparation import getValidation
from Model import Classifier
import json

print()
print("Set Up")
print()

# set up
modelName = "ResNet50"


print("modelName: ", modelName)

print()
print("Data Preperation")
print()
validation_data  = getValidation( masked = True)

print()
print("Evaluations")
print()
model = Classifier(createModel = False, path = "Model/TrainedModel" + modelName) 
# results = model.evaluate(validation)


results = model.evaluate(validation_data)

with open('Results/evaluationMasks' + modelName, 'w') as convert_file:
     convert_file.write(json.dumps(results.history))
print()
print("History is suuccsesfuly saved.")
print()

matric = ['loss', 'accuracy',  'Recall', 'Precision',  'SpecificityAtSensitivity']

for i, j in zip(matric, results):
    print(i," : ", j)
