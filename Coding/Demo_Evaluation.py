from numpy import matrix
from tensorflow import keras
from  DataPreparation import getValidation
from Model import Classifier


validation_data  = getValidation(masked = True)

model = keras.models.load_model('Model')
print("Evaluate:")
results = model.evaluate(validation_data, batch_size = 16)

matric = ['loss', 'accuracy',  'Recall', 'Precision',  'SpecificityAtSensitivity']

for i, j in zip(matric, results):
    print(i," : ", j)
