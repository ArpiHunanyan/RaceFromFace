
from tensorflow import keras
from  DataPreparation import getValidation, getTrain, plotResaluts
from DenseNetImplementation import DenceNetClassifier
import numpy as np
import json


train_data, train_labels = getTrain()
validation_data = getValidation()

model = DenceNetClassifier()
model.compile()
print()
# ## Train the top layer (for classifaction) 
print("Training the top layer (for classifaction) ")
trainingFirst = model.fit(x = train_data, y = train_labels,  epochs = 1, validation_data = validation_data)

  
with open('trainingFirst', 'w') as convert_file:
     convert_file.write(json.dumps(trainingFirst.history))


#plotResaluts(trainingFirst, 'loss', 'The top layer : Loss')
#plotResaluts(trainingFirst, 'accuracy', 'The top layer : Accuracy')

print()
## Fine - Tuning
print("Fine - Tuning ")
model.setTrainable(True)
model.compile(optimizer = keras.optimizers.Adam(1e-5))

# #model
trainingSecond = model.fit(x = train_data, y = train_labels,  epochs =  1, validation_data = validation_data)

with open('trainingSecond', 'w') as convert_file:
     convert_file.write(json.dumps(trainingSecond.history))

#plotResaluts(trainingSecond , 'loss', 'Fine - Tuning: Loss')
#plotResaluts(trainingSecond , 'accuracy', 'Fine - Tuning: Accuracy')
