from tensorflow import keras
from  DataPreparation import getValidation, getTrain
from DenseNetImplementation import DenceNetClassifier
import json


print("Data Preperation")
train_data, train_labels = getTrain(30, age = True, gender = True)
validation_data = getValidation(30, age = True, gender = True)

model = DenceNetClassifier()
model.MultipleInputsModel()
model.compile()
print()

# ## Train the top layer (for classifaction) 
print("Training the top layer (for classifaction) ")
trainingFirst = model.fit(x = train_data, y = train_labels,  epochs = 16, validation_data = validation_data, batch_size = 16)

  
with open('trainingFirst_1', 'w') as convert_file:
     convert_file.write(json.dumps(trainingFirst.history))


print()
## Fine - Tuning
print("Fine - Tuning ")
model.setTrainable(True)
model.compile(optimizer = keras.optimizers.Adam(1e-7)) # 1e-5 => 1e-7

#model
trainingSecond = model.fit(x = train_data, y = train_labels,  epochs =  4, validation_data = validation_data)

with open('trainingSecond_1', 'w') as convert_file:
     convert_file.write(json.dumps(trainingSecond.history))

